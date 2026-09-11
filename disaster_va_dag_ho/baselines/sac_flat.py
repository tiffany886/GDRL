"""B5: flat off-policy SAC-flat baseline (CTDE, shared actor).

SAC-flat is the stochastic member of the off-policy MADDPG/MATD3 family used
for B5. The shared per-UAV actor has a tanh-squashed Gaussian velocity head
and a per-terminal-slot categorical {UAV, LEO, wait} head (trained through a
straight-through Gumbel-Softmax relaxation: a plain continuous sigmoid
threshold cannot represent the middle LEO class once it saturates). Twin
centralized critics and an entropy-regularised soft target make exploration
self-sustaining so the deterministic-policy collapse seen in MATD3-flat does
not occur. At evaluation the heads are greedy (mean velocity / argmax offload).
No embedded ScheduleDAG decision rule is used (spec §8.1 B5).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .td3_flat import Td3Critic, ReplayBuffer, gumbel_softmax, one_hot, _to_t

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class SacActor(nn.Module):
    """Per-UAV head: 2-d tanh-Normal velocity + per-slot 3-class logits."""

    def __init__(self, obs_dim: int, num_slots: int, hidden: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.vel_mu = nn.Linear(hidden, 2)
        self.vel_logstd = nn.Linear(hidden, 2)
        self.off_head = nn.Linear(hidden, 3 * num_slots)

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.tanh(self.vel_mu(h))
        log_std = torch.clamp(self.vel_logstd(h), LOG_STD_MIN, LOG_STD_MAX)
        logits = self.off_head(h).reshape(*h.shape[:-1], self.num_slots, 3)
        return mu, log_std, logits


def tanh_normal_logp(u: torch.Tensor, mu: torch.Tensor,
                     std: torch.Tensor) -> torch.Tensor:
    """Log-density of a=tanh(u), u~N(mu, std)."""
    z = (u - mu) / (std + 1e-8)
    logp = (-0.5 * z.pow(2) - std.log() - 0.5 * np.log(2.0 * np.pi)).sum(-1)
    logp = logp - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(-1)
    return logp


class SacFlat:
    def __init__(self, obs_dim: int, num_uavs: int, num_slots: int,
                 seed: int = 0, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.05, lr: float = 3e-4, hidden: int = 128,
                 device: str = "cpu", warmup: int = 500, batch: int = 128,
                 gs_tau: float = 0.5, offload_only: bool = False,
                 mover=None):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_slots = num_slots
        self.act_dim = 2 + 3 * num_slots  # per UAV
        torch.manual_seed(seed)
        self.device = device
        self.gs_tau = gs_tau
        self.actor = SacActor(obs_dim, num_slots, hidden).to(device)
        s_dim = num_uavs * obs_dim
        a_dim = num_uavs * self.act_dim
        self.critics = nn.ModuleList(
            [Td3Critic(s_dim, a_dim, hidden).to(device) for _ in range(2)])
        self.critic_targets = nn.ModuleList(
            [Td3Critic(s_dim, a_dim, hidden).to(device) for _ in range(2)])
        for c, ct in zip(self.critics, self.critic_targets):
            ct.load_state_dict(c.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            [p for c in self.critics for p in c.parameters()], lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.warmup = warmup
        self.batch = batch
        self.alpha_start = alpha
        self.alpha = alpha
        self.offload_only = offload_only   # v2 B5: velocity supplied externally
        self.mover = mover                 # callable(env) -> (K,2) m/s (T-patrol)
        self.buffer = ReplayBuffer(cap=30000)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

    def _set_alpha(self, ep: int, episodes: int):
        """Exponential alpha decay: high early entropy, near-greedy late."""
        self.alpha = self.alpha_start * (0.996 ** ep)

    # ------------------------------------------------------------------
    def _sample_actions(self, obs: torch.Tensor, explore: bool):
        """Return encodings/log-probs; gradients flow for actor updates."""
        mu, log_std, logits = self.actor(obs)
        std = log_std.exp()
        if explore:
            u = mu + std * torch.randn_like(mu)
        else:
            u = mu
        vel = torch.tanh(u)
        logp_vel = tanh_normal_logp(u, mu, std)
        if self.offload_only:
            # v2 B5: velocity head is inert; the fixed trajectory moves the UAV
            vel = torch.zeros_like(vel)
            logp_vel = torch.zeros_like(logp_vel)
        # discrete offload: straight-through Gumbel relaxation
        probs = F.softmax(logits, dim=-1)
        if explore:
            off = gumbel_softmax(logits, tau=self.gs_tau, hard=True)
        else:
            off = F.one_hot(logits.argmax(-1), 3).float()
        cls = off.argmax(-1)
        logp_off = torch.log(probs.gather(-1, cls.unsqueeze(-1)).squeeze(-1)
                             + 1e-8).sum(-1)
        return vel, off, logp_vel, logp_off

    def act_enc(self, obs: np.ndarray, explore: bool = True) -> tuple:
        """Return (enc (K, act_dim), dec (K, num_slots), logp_sum)."""
        with torch.no_grad():
            vel, off, lp_v, lp_o = self._sample_actions(
                _to_t(obs, self.device), explore)
            enc = torch.cat(
                [vel, off.reshape(self.num_uavs, -1)], dim=-1).cpu().numpy()
            dec = off.argmax(-1).cpu().numpy()
            logp = float((lp_v + lp_o).sum().item())
        return enc.astype(np.float32), dec.astype(np.int64), logp

    def act_env(self, obs: np.ndarray, explore: bool = False) -> dict:
        enc, dec, _ = self.act_enc(obs, explore=explore)
        return {
            "vel": enc[:, :2],
            "dec": dec,
            "logp_vel": np.zeros(self.num_uavs),
            "logp_cat": None,
            "entropy": 0.0,
            "value": 0.0,
        }

    # ------------------------------------------------------------------
    def _soft_update(self):
        with torch.no_grad():
            for src, dst in zip(self.critics, self.critic_targets):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def step_update(self, batch: dict):
        obs_b = _to_t(batch["obs"], self.device)      # (B, K*D)
        act_b = _to_t(batch["act"], self.device)      # (B, K*A)
        rew_b = _to_t(batch["rew"], self.device)
        done_b = _to_t(batch["done"], self.device)
        nobs_b = _to_t(batch["nobs"], self.device)
        k = self.num_uavs

        with torch.no_grad():
            mu_t, log_std_t, logits_t = self.actor(
                nobs_b.reshape(-1, k, self.obs_dim))
            std_t = log_std_t.exp()
            u_t = mu_t + std_t * torch.randn_like(mu_t)
            vel_t = torch.tanh(u_t)
            lp_v_t = tanh_normal_logp(u_t, mu_t, std_t)
            if self.offload_only:
                vel_t = torch.zeros_like(vel_t)
                lp_v_t = torch.zeros_like(lp_v_t)
            probs_t = F.softmax(logits_t, dim=-1)
            off_t = gumbel_softmax(logits_t, tau=self.gs_tau, hard=True)
            cls_t = off_t.argmax(-1)
            lp_o_t = torch.log(probs_t.gather(
                -1, cls_t.unsqueeze(-1)).squeeze(-1) + 1e-8).sum(-1)
            na = torch.cat(
                [vel_t, off_t.reshape(-1, k, 3 * self.num_slots)],
                dim=-1).reshape(nobs_b.shape[0], -1)
            q1 = self.critic_targets[0](nobs_b, na)
            q2 = self.critic_targets[1](nobs_b, na)
            y = rew_b + self.gamma * (1.0 - done_b) * (
                torch.min(q1, q2)
                - self.alpha * (lp_v_t + lp_o_t).sum(-1))

        loss_c = sum(((c(obs_b, act_b) - y) ** 2).mean()
                     for c in self.critics)
        self.critic_opt.zero_grad()
        loss_c.backward()
        for c in self.critics:
            for p in c.parameters():
                if p.grad is not None:
                    p.grad.data.clamp_(-10.0, 10.0)
        self.critic_opt.step()

        vel, off, lp_v, lp_o = self._sample_actions(
            obs_b.reshape(-1, k, self.obs_dim), explore=True)
        ca = torch.cat([vel, off.reshape(vel.shape[0], k, 3 * self.num_slots)],
                       dim=-1).reshape(obs_b.shape[0], -1)
        q = torch.min(self.critics[0](obs_b, ca),
                      self.critics[1](obs_b, ca))
        loss_a = (self.alpha * (lp_v + lp_o).sum(-1) - q).mean()
        self.actor_opt.zero_grad()
        loss_a.backward()
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-10.0, 10.0)
        self.actor_opt.step()
        self._soft_update()
        self.train_steps += 1
        return {"loss_c": float(loss_c.item()), "loss_a": float(loss_a.item())}

    # ------------------------------------------------------------------
    def fit(self, env, episodes: int, eval_every: int = 0,
            eval_seeds: tuple = ()) -> list:
        vmax = env.cfg.uav_speed_max_mps
        mover = self.mover if self.offload_only else None
        rows = []
        for ep in range(episodes):
            self._set_alpha(ep, episodes)
            obs = env.reset(seed=(ep * 7 + 7) % 1000)
            ep_rew = 0.0
            steps = 0
            done = False
            while not done:
                if len(self.buffer.obs) < self.warmup:
                    vel = (np.zeros((self.num_uavs, 2), dtype=float)
                           if self.offload_only
                           else self.rng.uniform(-1, 1,
                                                 size=(self.num_uavs, 2)))
                    dec = self.rng.integers(0, 3, size=(self.num_uavs,
                                                        self.num_slots))
                    act = np.concatenate(
                        [vel, one_hot(dec, 3).reshape(self.num_uavs, -1)],
                        axis=-1).astype(np.float32)
                else:
                    act, dec, _ = self.act_enc(obs, explore=True)
                vel_env = (mover.act(env) if hasattr(mover, "act")
                           else mover(env)) if mover is not None \
                    else act[:, :2] * vmax
                nobs, rew, done, info = env.step(vel_env, dec)
                self.buffer.add(obs, act, rew, done, nobs)
                if len(self.buffer.obs) >= self.warmup and len(
                        self.buffer.obs) % 4 == 0:
                    self.step_update(self.buffer.sample(self.batch, self.rng))
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
        mover = self.mover if self.offload_only else None
        done = False
        steps = 0
        while not done:
            out = self.act_env(obs, explore=False)
            vel_env = (mover.act(env) if hasattr(mover, "act")
                       else mover(env)) if mover is not None \
                else out["vel"] * vmax
            obs, rew, done, info = env.step(vel_env, out["dec"])
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
            "frac_unassociated": float(env.frac_unassociated),
            "fail_uncovered": int(m["fail_uncovered"]),
        }
