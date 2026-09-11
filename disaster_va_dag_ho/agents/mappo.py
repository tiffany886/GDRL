"""MAPPO (CTDE, parameter sharing across homogeneous UAVs) for VA-DAG-HO.

Main method (``mode="embed"``): the actor only outputs the horizontal
velocity; placement/offloading is decided by the embedded ScheduleDAG.

Baseline B4 (``mode="flat"``): the actor additionally outputs, for every
assigned-terminal slot, a categorical decision {0=UAV, 1=LEO, 2=wait} that
selects the device for that terminal's most urgent app with a ready node —
no embedded cost heuristic is used for offloading.

v2 B4 (``mode="flat_no_vel"``): like ``flat`` but the velocity head is inert
(velocity is supplied externally by the fixed T-patrol trajectory), so the
policy learns *only* the offload decisions (spec v2 §1.5 / §3.1 B4).

CTDE critic consumes the concatenated global observation; rewards are shared
per slot. GAE + clipped PPO with one update per collected episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


@dataclass
class StepData:
    obs: np.ndarray             # (K, obs_dim)
    next_obs: np.ndarray        # (K, obs_dim)
    vel: np.ndarray             # (K, 2) scaled actions in [-1, 1]
    dec: Optional[np.ndarray]   # (K, slots) ints, flat mode only
    rew: float
    done: bool
    logp_vel: np.ndarray        # (K,)
    logp_cat: Optional[np.ndarray]  # (K,), flat mode only
    value: float
    next_value: float


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, use_cat: bool, num_slots: int, hidden: int = 128):
        super().__init__()
        self.trunk = MLP(obs_dim, hidden)
        self.vel_head = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2) - 0.6)
        self.use_cat = use_cat
        self.num_slots = num_slots
        if use_cat:
            self.cat_head = nn.Linear(hidden, num_slots * 3)

    def distributions(self, x: torch.Tensor):
        feat = self.trunk(x)
        mean = self.vel_head(feat)
        std = torch.exp(self.log_std.clamp(-2.3, 0.5))
        vel_dist = Normal(mean, std)
        cat_dist = None
        if self.use_cat:
            logits = self.cat_head(feat).view(-1, self.num_slots, 3)
            cat_dist = Categorical(logits=logits)
        return vel_dist, cat_dist


class CriticNet(nn.Module):
    def __init__(self, global_obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Mappo:
    def __init__(self, obs_dim: int, num_uavs: int, num_slots: int,
                 mode: str = "embed", seed: int = 0,
                 gamma: float = 0.99, lam: float = 0.95, lr: float = 3e-4,
                 clip: float = 0.2, epochs: int = 4, entropy_coef: float = 0.01,
                 hidden: int = 128, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.num_uavs = num_uavs
        self.num_slots = num_slots
        self.mode = mode
        self.use_cat = mode in ("flat", "flat_no_vel")
        self.offload_only = mode == "flat_no_vel"
        torch.manual_seed(seed)
        self.actor = ActorNet(obs_dim, self.use_cat, num_slots, hidden).to(device)
        self.critic = CriticNet(num_uavs * obs_dim, hidden).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.device = device

    # ------------------------------------------------------------------
    # acting / rollouts
    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> dict:
        """Sample one-step actions. ``obs``: (K, obs_dim) current observation."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vel_dist, cat_dist = self.actor.distributions(obs_t)
            value = self._critic_value(obs)
            if self.use_cat:
                dec = cat_dist.sample()
                logp_cat = cat_dist.log_prob(dec).sum(-1).cpu().numpy()
                dec_np = dec.cpu().numpy().astype(np.int64)
                if self.offload_only:
                    # v2: velocity comes from the fixed trajectory, not the head
                    vel_np = np.zeros((self.num_uavs, 2), dtype=float)
                    logp_vel = np.zeros(self.num_uavs, dtype=float)
                    entropy = cat_dist.entropy().sum(-1).mean().item()
                else:
                    vel = vel_dist.sample()
                    logp_vel = vel_dist.log_prob(vel).sum(-1).cpu().numpy()
                    vel_np = torch.clamp(vel, -1.0, 1.0).cpu().numpy()
                    entropy = (vel_dist.entropy().sum(-1).mean()
                               + cat_dist.entropy().sum(-1).mean()).item()
            else:
                vel = vel_dist.sample()
                logp_vel = vel_dist.log_prob(vel).sum(-1).cpu().numpy()
                vel_np = torch.clamp(vel, -1.0, 1.0).cpu().numpy()
                entropy = vel_dist.entropy().sum(-1).mean().item()
                logp_cat = None
                dec_np = None
        return {
            "vel": vel_np,
            "dec": dec_np,
            "logp_vel": logp_vel,
            "logp_cat": logp_cat,
            "entropy": entropy,
            "value": value,
        }

    def act_deterministic(self, obs: np.ndarray) -> dict:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vel_dist, cat_dist = self.actor.distributions(obs_t)
            value = self._critic_value(obs)
            if self.use_cat:
                dec = cat_dist.logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
                if self.offload_only:
                    vel = np.zeros((self.num_uavs, 2), dtype=float)
                else:
                    vel = torch.clamp(vel_dist.mean, -1.0, 1.0).cpu().numpy()
            else:
                vel = torch.clamp(vel_dist.mean, -1.0, 1.0).cpu().numpy()
                dec = None
            return {
                "vel": vel,
                "dec": dec,
                "logp_vel": np.zeros(self.num_uavs),
                "logp_cat": None,
                "entropy": 0.0,
                "value": value,
            }

    def _critic_value(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            return self.critic(
                torch.as_tensor(obs.reshape(1, -1),
                                dtype=torch.float32, device=self.device)
            ).item()

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def update(self, traj: list) -> dict:
        """One PPO epoch loop over a collected episode (list of StepData)."""
        T = len(traj)
        n = T * self.num_uavs
        obs = np.stack([s.obs for s in traj]).reshape(n, -1)
        vel = np.stack([s.vel for s in traj]).reshape(n, -1)
        lp_vel = np.stack([s.logp_vel for s in traj]).reshape(n)
        rew = np.array([s.rew for s in traj], dtype=float)
        done = np.array([s.done for s in traj], dtype=float)
        value = np.array([s.value for s in traj], dtype=float)
        next_value = np.array([s.next_value for s in traj], dtype=float)
        adv = self._gae(value, next_value, rew, done)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = adv + value

        obs_f = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        vel_f = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        adv_f = torch.as_tensor(np.repeat(adv, self.num_uavs),
                                dtype=torch.float32, device=self.device)
        lp_vel_f = torch.as_tensor(lp_vel, dtype=torch.float32, device=self.device)
        obs_g = torch.as_tensor(np.stack([s.obs for s in traj]).reshape(T, -1),
                                dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        dec_f = None
        lp_cat_f = None
        if self.use_cat:
            dec_f = torch.as_tensor(
                np.stack([s.dec for s in traj]).reshape(-1),
                dtype=torch.int64, device=self.device,
            )
            lp_cat_f = torch.as_tensor(
                np.stack([s.logp_cat for s in traj]).reshape(-1),
                dtype=torch.float32, device=self.device,
            )

        losses = []
        for _ in range(self.epochs):
            vel_dist, cat_dist = self.actor.distributions(obs_f)
            loss_actor = torch.zeros((), dtype=torch.float32, device=self.device)
            entropy = torch.zeros((), dtype=torch.float32, device=self.device)
            if not self.offload_only:
                new_lp = vel_dist.log_prob(vel_f).sum(-1)
                ratio = torch.exp(new_lp - lp_vel_f)
                surr1 = ratio * adv_f
                surr2 = torch.clamp(ratio, 1.0 - self.clip,
                                    1.0 + self.clip) * adv_f
                loss_actor = -torch.min(surr1, surr2).mean()
                entropy = entropy + vel_dist.entropy().sum(-1).mean()
            if self.use_cat:
                logits = cat_dist.logits.reshape(-1, 3)
                cdist = Categorical(logits=logits)
                cat_lp = cdist.log_prob(dec_f).reshape(-1, self.num_slots).sum(-1)
                cat_ratio = torch.exp(cat_lp - lp_cat_f)
                cs1 = cat_ratio * adv_f
                cs2 = torch.clamp(cat_ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_f
                loss_actor = loss_actor - torch.min(cs1, cs2).mean()
                entropy = entropy + cat_dist.entropy().sum(-1).mean()
            value_pred = self.critic(obs_g)
            loss_critic = ((value_pred - ret_t) ** 2).mean()
            loss = loss_actor + loss_critic - self.entropy_coef * entropy
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 5.0
            )
            self.opt.step()
            losses.append(float(loss.item()))
        return {
            "loss": float(np.mean(losses)),
            "entropy": float(entropy.item()),
            "mean_adv": float(adv.mean()),
        }

    def _gae(self, value, next_value, rew, done):
        T = len(rew)
        adv = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * next_value[t] * (1.0 - done[t]) - value[t]
            gae = delta + self.gamma * self.lam * (1.0 - done[t]) * gae
            adv[t] = gae
        return adv
