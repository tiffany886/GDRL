"""B5: flat off-policy MATD3-style baseline (CTDE, shared actor).

NOTE (W4): the deterministic MATD3-flat variant below was implemented and
abandoned for the main table: its continuous sigmoid-threshold offload head
cannot represent the middle LEO class once it saturates, and the greedy
deterministic actor collapses to a full-speed corner run (value
overestimation), so it never reaches a usable plateau. The shipped B5 row is
``sac_flat.SacFlat`` (stochastic member of the same off-policy family). This
module is kept as the reference implementation and hosts shared building
blocks (critic / replay / Gumbel helpers) used by ``sac_flat.py``.

Deterministic velocity actor (parameter-tied across homogeneous UAVs) plus a
*discrete* per-terminal-slot offload head {UAV, LEO, wait}. Twin centralized
critics see the concatenated global observation and the joint action; the
offload head is trained with a Gumbel-Softmax straight-through relaxation so
the critic's policy gradient reaches every class logit (a continuous sigmoid
threshold cannot represent the middle LEO band once it saturates). At eval the
head is an argmax. No embedded ScheduleDAG decision rule is used (spec
§8.1 B5: flat off-policy MADDPG/MATD3 family).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def gumbel_noise(shape, device):
    u = torch.rand(shape, device=device).clamp_min(1e-6)
    return -torch.log(-torch.log(u))


def gumbel_softmax(logits: torch.Tensor, tau: float,
                   hard: bool = False) -> torch.Tensor:
    """Soft relaxation of argmax with straight-through option."""
    y = F.softmax((logits + gumbel_noise(logits.shape, logits.device)) / tau, dim=-1)
    if not hard:
        return y
    idx = y.argmax(dim=-1)
    y_hard = torch.zeros_like(y).scatter_(-1, idx.unsqueeze(-1), 1.0)
    return y_hard - y.detach() + y


def one_hot(idx, n: int, dtype=np.float32):
    out = np.zeros(idx.shape + (n,), dtype=dtype)
    rows = np.arange(idx.shape[0])[:, None]
    cols = np.arange(idx.shape[1])[None, :]
    out[rows, cols, idx] = 1.0
    return out


class Td3Actor(nn.Module):
    """Per-UAV head: 2-d tanh velocity + per-slot 3-class offload logits."""

    def __init__(self, obs_dim: int, num_slots: int, hidden: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.vel_head = nn.Linear(hidden, 2)
        self.off_head = nn.Linear(hidden, 3 * num_slots)

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.tanh(self.vel_head(h))
        logits = self.off_head(h).reshape(*h.shape[:-1], self.num_slots, 3)
        return mu, logits


class Td3Critic(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, act):
        return self.net(torch.cat([state, act], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, cap: int = 20000):
        self.cap = cap
        self.obs = deque(maxlen=cap)
        self.act = deque(maxlen=cap)
        self.rew = deque(maxlen=cap)
        self.done = deque(maxlen=cap)
        self.nobs = deque(maxlen=cap)

    def add(self, obs, act, rew, done, nobs):
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(float(rew))
        self.done.append(float(done))
        self.nobs.append(nobs)

    def sample(self, batch: int, rng):
        n = len(self.obs)
        idx = rng.integers(0, n, size=batch)
        return {
            "obs": np.stack([self.obs[i] for i in idx]).reshape(batch, -1),
            "act": np.stack([self.act[i] for i in idx]).reshape(batch, -1),
            "rew": np.array([self.rew[i] for i in idx], dtype=float),
            "done": np.array([self.done[i] for i in idx], dtype=float),
            "nobs": np.stack([self.nobs[i] for i in idx]).reshape(batch, -1),
        }


class Td3Flat:
    def __init__(self, obs_dim: int, num_uavs: int, num_slots: int, seed: int = 0,
                 gamma: float = 0.99, tau: float = 0.005, policy_noise: float = 0.3,
                 noise_clip: float = 0.5, expl_noise: float = 0.35,
                 lr: float = 1e-4, hidden: int = 128, device: str = "cpu",
                 warmup: int = 500, batch: int = 128):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_slots = num_slots
        # per-UAV action encoding: [vx, vy] + one slot-hot x (slots*3)
        self.act_dim = 2 + 3 * num_slots
        torch.manual_seed(seed)
        self.device = device
        self.actor = Td3Actor(obs_dim, num_slots, hidden).to(device)
        self.actor_target = Td3Actor(obs_dim, num_slots, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        s_dim = num_uavs * obs_dim
        a_dim = num_uavs * self.act_dim
        self.critics = nn.ModuleList(
            [Td3Critic(s_dim, a_dim, hidden).to(device) for _ in range(2)]
        )
        self.critic_targets = nn.ModuleList(
            [Td3Critic(s_dim, a_dim, hidden).to(device) for _ in range(2)]
        )
        for c, ct in zip(self.critics, self.critic_targets):
            ct.load_state_dict(c.state_dict())
        params = list(self.actor.parameters())
        for c in self.critics:
            params += list(c.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise
        self.warmup = warmup
        self.batch = batch
        self.buffer = ReplayBuffer()
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0
        self._rew_hist = deque(maxlen=2048)
        self._rew_mean = 0.0
        self._rew_std = 1.0

    # ------------------------------------------------------------------
    def _decode(self, act_enc: np.ndarray) -> np.ndarray:
        """(K, act_dim) encoding -> (K, num_slots) discrete decisions."""
        one = act_enc[:, 2:].reshape(self.num_uavs, self.num_slots, 3)
        return np.argmax(one, axis=-1)

    def select_action(self, obs: np.ndarray, explore: bool = True,
                      noise_scale: float = 1.0) -> tuple:
        """Return (act_encoding (K, act_dim), dec (K, num_slots))."""
        with torch.no_grad():
            mu, logits = self.actor(_to_t(obs, self.device))
            mu = mu.cpu().numpy()
            logits = logits.cpu().numpy()
        vel = np.clip(mu + (self.rng.normal(0.0, self.expl_noise * noise_scale,
                                            size=mu.shape)
                            if explore else 0.0), -1.0, 1.0)
        if explore:
            g = self.rng.gumbel(size=logits.shape)
            dec = np.argmax(logits + g, axis=-1)
        else:
            dec = np.argmax(logits, axis=-1)
        enc = np.concatenate([vel, one_hot(dec, 3).reshape(self.num_uavs, -1)],
                             axis=-1).astype(np.float32)
        return enc, dec

    def act_env(self, obs: np.ndarray, explore: bool = False) -> dict:
        a, dec = self.select_action(obs, explore=explore, noise_scale=1.0)
        return {
            "vel": a[:, :2],
            "dec": dec,
            "logp_vel": np.zeros(self.num_uavs),
            "logp_cat": None,
            "entropy": 0.0,
            "value": 0.0,
        }

    def _temp(self, episodes: int, ep: int) -> float:
        frac = min(max(ep / max(episodes - 1, 1), 0.0), 1.0)
        return 1.0 - 0.85 * frac  # Gumbel temperature 1.0 -> 0.15

    # ------------------------------------------------------------------
    def step_update(self, batch: dict, tau_gs: float):
        obs_b = _to_t(batch["obs"], self.device)          # (B, K*D)
        act_b = _to_t(batch["act"], self.device)          # (B, K*A)
        rew_b = _to_t(batch["rew"], self.device)
        done_b = _to_t(batch["done"], self.device)
        nobs_b = _to_t(batch["nobs"], self.device)

        with torch.no_grad():
            mu_t, logits_t = self.actor_target(
                nobs_b.reshape(-1, self.num_uavs, self.obs_dim))
            vel_t = torch.clamp(
                mu_t + torch.normal(0.0, self.policy_noise, size=mu_t.shape,
                                    device=self.device).clamp(
                                        -self.noise_clip, self.noise_clip),
                -1.0, 1.0)
            off_t = gumbel_softmax(logits_t, tau=max(tau_gs, 0.2), hard=False)
            na = torch.cat(
                [vel_t, off_t.reshape(vel_t.shape[0], self.num_uavs, -1)],
                dim=-1).reshape(nobs_b.shape[0], -1)
            q1 = self.critic_targets[0](nobs_b, na)
            q2 = self.critic_targets[1](nobs_b, na)
            self._rew_hist.append(float(batch["rew"].mean()))
            if len(self._rew_hist) >= 64:
                arr = np.asarray(self._rew_hist, dtype=float)
                self._rew_mean = float(arr.mean())
                self._rew_std = max(float(arr.std()), 1.0)
            y = ((rew_b - self._rew_mean) / self._rew_std
                 + self.gamma * (1.0 - done_b) * torch.min(q1, q2))

        loss_c = sum(((c(obs_b, act_b) - y) ** 2).mean() for c in self.critics)
        self.opt.zero_grad()
        loss_c.backward()
        for c in self.critics:
            for p in c.parameters():
                p.grad.data.clamp_(-10.0, 10.0)
        self.opt.step()
        self.train_steps += 1

        loss_a = None
        if self.train_steps % 2 == 0:
            mu, logits = self.actor(obs_b.reshape(-1, self.num_uavs, self.obs_dim))
            off = gumbel_softmax(logits, tau=tau_gs, hard=False)
            ca = torch.cat([mu, off.reshape(mu.shape[0], self.num_uavs, -1)],
                           dim=-1).reshape(obs_b.shape[0], -1)
            loss_a = -self.critics[0](obs_b, ca).mean()
            self.opt.zero_grad()
            loss_a.backward()
            for p in self.actor.parameters():
                p.grad.data.clamp_(-10.0, 10.0)
            self.opt.step()
            self._soft_update()
        return {"loss_c": float(loss_c.item()),
                "loss_a": None if loss_a is None else float(loss_a.item())}

    def _soft_update(self):
        with torch.no_grad():
            for src, dst in (
                (self.actor, self.actor_target),
                (self.critics[0], self.critic_targets[0]),
                (self.critics[1], self.critic_targets[1]),
            ):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------
    def fit(self, env, episodes: int, eval_every: int = 0,
            eval_seeds: tuple = ()) -> list:
        """Off-policy training loop with replay + per-episode log rows."""
        vmax = env.cfg.uav_speed_max_mps
        rows = []
        for ep in range(episodes):
            obs = env.reset(seed=(ep * 7 + 7) % 1000)
            tau_gs = self._temp(episodes, ep)
            ep_rew = 0.0
            steps = 0
            done = False
            while not done:
                if len(self.buffer.obs) < self.warmup:
                    vel = self.rng.uniform(-1, 1, size=(self.num_uavs, 2))
                    dec = self.rng.integers(0, 3, size=(self.num_uavs,
                                                        self.num_slots))
                    act = np.concatenate(
                        [vel, one_hot(dec, 3).reshape(self.num_uavs, -1)],
                        axis=-1).astype(np.float32)
                else:
                    frac = 1.0 - 0.9 * ep / max(episodes - 1, 1)
                    act, dec = self.select_action(obs, explore=True,
                                                  noise_scale=frac)
                nobs, rew, done, info = env.step(act[:, :2] * vmax, dec)
                self.buffer.add(obs, act, rew, done, nobs)
                if len(self.buffer.obs) >= self.warmup and len(
                        self.buffer.obs) % 4 == 0:
                    self.step_update(self.buffer.sample(self.batch, self.rng),
                                     tau_gs)
                ep_rew += float(rew)
                obs = nobs
                steps += 1
            rows.append({"kind": "train", "episode": ep, "reward": ep_rew})
            if eval_every and (ep + 1) % eval_every == 0:
                for s in eval_seeds:
                    rows.append({"kind": "eval", "episode": ep,
                                 "eval_seed": int(s),
                                 **self._eval_one(env, int(s))})
        return rows

    def _eval_one(self, env, seed: int) -> dict:
        obs = env.reset(seed)
        vmax = env.cfg.uav_speed_max_mps
        done = False
        steps = 0
        while not done:
            out = self.act_env(obs, explore=False)
            obs, rew, done, info = env.step(out["vel"] * vmax, out["dec"])
            steps += 1
            if steps > 2000:
                raise RuntimeError("no termination")
        m = env.metrics
        done_n = int(m["apps_done"])
        arrived = int(m["apps_arrived"])
        return {
            "reward": float(m["reward_sum"]),
            "success_rate": done_n / max(arrived, 1),
            "apps_arrived": arrived,
            "apps_done": done_n,
            "apps_failed": int(m["apps_failed"]),
            "mean_delay_s": m["done_delay_sum_s"] / max(done_n, 1),
            "energy_j": float(m["energy_total_j"]),
            "leo_placements": int(m["leo_placements"]),
            "leo_invalid_attempts": int(m["leo_invalid_attempts"]),
        }
