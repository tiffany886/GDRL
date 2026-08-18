"""Decentralized parameter-shared DRL baselines for the multi-UAV scene.

Each UAV is controlled by the *same* policy network (parameter sharing, the
standard trick for homogeneous multi-agent systems) but observes only its own
local state: own position/queue/battery, the nearest LEO, hotspot, and the
load-balanced subset of users assigned to it.  At each slot the K policies are
evaluated on their local observations, and the per-UAV moves/offloading
decisions are merged into the global action consumed by the multi-UAV env.

Supported baselines:
- DQN (discrete per-user offloading grid, TEA heuristic move), name "m_dqn"
- TD3 (continuous move + continuous offloading scores), name "m_td3"
- SAC (same continuous action space as TD3), name "m_sac"
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .learning import DEVICE, _append_csv
from .multi_uav import associate_users, uav_pos_array
from .physics import rate_uav_leo_vec, rate_user_uav_vec

RATIO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
_PER_USER_FEATS = 11


def lmax_for(config):
    return int(np.ceil(config.users / max(int(getattr(config, "uavs", 1)), 1)))


def local_obs_dim(config):
    lmax = lmax_for(config)
    hotspot_extra = 5 if getattr(config, "hotspot_motion", False) else 0
    return 8 + hotspot_extra + _PER_USER_FEATS * lmax + lmax


def make_local_obs(obs, config, k, assoc, lmax):
    """Fixed-size local observation vector for UAV k (normalized)."""
    area = config.area_size
    p_all = uav_pos_array(obs, config)
    p_k = p_all[k]
    user_pos = np.asarray(obs["user_pos"], dtype=float)
    idx = np.where(assoc == k)[0]
    U = len(obs["task_bits"])

    backlogs = np.asarray(obs["uav_backlog"], dtype=float)
    if backlogs.ndim == 0:
        backlogs = np.full(len(p_all), float(backlogs))
    batteries = np.asarray(obs["uav_battery"], dtype=float)
    if batteries.ndim == 0:
        batteries = np.full(len(p_all), float(batteries))
    backlog_norm = backlogs / max(config.uav_cpu_cycles_per_s * config.slot_seconds, 1.0)
    battery_norm = batteries / max(config.uav_battery_per_slot * config.horizon, 1.0)
    leo_backlog_norm = float(obs["leo_backlog"]) / max(
        config.leo_cpu_cycles_per_s * config.slot_seconds, 1.0)

    leo_xy = np.asarray(obs["leo_pos"], dtype=float)[:, :2]
    d_leo = np.linalg.norm(leo_xy - p_k[None, :], axis=-1)
    nearest_leo = leo_xy[int(np.argmin(d_leo))]

    feats = [
        p_k / area,                                   # 2
        (nearest_leo - p_k) / area,                   # 2
        np.array([float(backlog_norm[k])]),           # 1
        np.array([leo_backlog_norm]),                 # 1
        np.array([float(battery_norm[k])]),           # 1
        np.array([float(obs["t"]) / config.horizon]), # 1
    ]
    if getattr(config, "hotspot_motion", False) and obs.get("hotspot_pos") is not None:
        hp = np.asarray(obs["hotspot_pos"], dtype=float)
        hv = np.asarray(obs["hotspot_vel"], dtype=float)
        if hp.ndim == 1:
            hp = hp[None, :]
            hv = hv[None, :]
        hp_n = hp / area
        hv_n = hv / max(config.hotspot_speed, 1.0)
        d_h = np.linalg.norm(hp - p_k[None, :], axis=-1)
        feats.append(hp_n.mean(axis=0))               # 2 (mean hotspot)
        feats.append(hv_n.mean(axis=0))               # 2
        feats.append(np.array([float(d_h.min()) / area]))  # 1

    bits = np.asarray(obs["task_bits"], dtype=float)
    cpb = np.asarray(obs["cycles_per_bit"], dtype=float)
    has_task = (bits > 1.0e-6).astype(float)
    # per-user feature rows (only subset users are non-zero)
    rows = []
    for u in range(lmax):
        if u < len(idx):
            uu = int(idx[u])
            dist = float(np.linalg.norm(user_pos[uu] - p_k))
            rate_uu = float(np.asarray(
                rate_user_uav_vec(config, user_pos[uu][None, :], p_k[None, :])).reshape(-1)[0])
            relay = float(np.minimum(
                np.asarray(rate_user_uav_vec(config, user_pos[uu][None, :], p_k[None, :])).reshape(-1)[0],
                np.asarray(rate_uav_leo_vec(config, p_k[None, :], nearest_leo[None, :])).reshape(-1)[0]))
            local_lat = (bits[uu] * cpb[uu]) / config.user_cpu_cycles_per_s / config.success_deadline_s
            rows.append(np.concatenate([
                user_pos[uu] / area,                          # 2
                np.asarray(obs["user_vel"])[uu] / max(config.user_speed_max, 1.0),  # 2
                np.array([bits[uu] / max(config.task_bits_max, 1.0),
                          cpb[uu] / max(config.cycles_per_bit_max, 1.0),
                          has_task[uu]]),                     # 3
                np.array([dist / area,
                          rate_uu / max(config.bandwidth_hz, 1.0),
                          relay / max(config.bandwidth_hz, 1.0),
                          min(local_lat, 1.0e3)]),            # 4
            ]))
        else:
            rows.append(np.zeros(_PER_USER_FEATS, dtype=np.float32))
    valid = np.zeros(lmax, dtype=np.float32)
    valid[: len(idx)] = 1.0
    return np.concatenate(feats + rows + [valid]).astype(np.float32)


# ---------------------------------------------------------------------------
# DQN (discrete offloading + TEA heuristic move)
# ---------------------------------------------------------------------------
class LocalDQN(nn.Module):
    def __init__(self, obs_dim, lmax, n_actions=15, hidden=256):
        super().__init__()
        self.lmax = lmax
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, lmax * n_actions)

    def forward(self, obs):
        return self.head(self.net(obs))


class DQNTrainerM:
    name = "m_dqn"

    def __init__(self, config, lr=1.0e-3, gamma=0.99, batch_size=128,
                 buffer_size=100000, hidden=256):
        self.config = config
        self.lmax = lmax_for(config)
        self.obs_dim = local_obs_dim(config)
        self.q_net = LocalDQN(self.obs_dim, self.lmax).to(DEVICE)
        self.target_net = LocalDQN(self.obs_dim, self.lmax).to(DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.steps = 0
        self.update_target_every = 500

    def _move_heuristic(self, obs, config, k, assoc):
        user_pos = np.asarray(obs["user_pos"], dtype=float)
        workload = np.asarray(obs["task_bits"], dtype=float) * np.asarray(obs["cycles_per_bit"], dtype=float)
        idx = np.where(assoc == k)[0]
        p_k = uav_pos_array(obs, config)[k]
        if len(idx) > 0 and workload[idx].sum() > 1.0e-9:
            target = np.average(user_pos[idx], axis=0, weights=workload[idx])
        elif len(idx) > 0:
            target = user_pos[idx].mean(axis=0)
        else:
            target = p_k
        mv = target - p_k
        norm = float(np.linalg.norm(mv))
        if norm > config.uav_speed_max and norm > 1.0e-9:
            mv = mv / norm * config.uav_speed_max
        return mv

    def act(self, obs_vec, epsilon=0.05, rng=None):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.q_net(obs_t).squeeze(0).cpu().numpy().reshape(self.lmax, -1)
        if rng is not None and rng.random() < epsilon:
            choices = np.stack([rng.integers(0, len(RATIO_GRID), size=self.lmax),
                                rng.integers(0, 3, size=self.lmax)], axis=-1)
        else:
            grid = np.stack([np.tile(RATIO_GRID, 3), np.repeat(np.arange(3), len(RATIO_GRID))], axis=-1)
            choices = grid[np.argmax(q, axis=1)]
        targets = choices[:, 1].astype(int)
        ratios = choices[:, 0].astype(float)
        return targets, ratios

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        actions = np.stack([b[1] for b in batch])  # (B, lmax, 2)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)
        q = self.q_net(states).view(-1, self.lmax, 15)
        act_idx = (actions[:, :, 1] * len(RATIO_GRID) + actions[:, :, 0]).astype(np.int64)
        q_sa = q.gather(2, torch.from_numpy(act_idx).unsqueeze(-1).to(DEVICE)).squeeze(-1)
        with torch.no_grad():
            q_next = self.target_net(next_states).view(-1, self.lmax, 15).max(dim=2).values
            target = rewards.unsqueeze(1) + self.gamma * (1.0 - dones).unsqueeze(1) * q_next
        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


# ---------------------------------------------------------------------------
# TD3 (continuous move + offloading scores)
# ---------------------------------------------------------------------------
class TD3ActorM(nn.Module):
    def __init__(self, obs_dim, lmax, hidden=256):
        super().__init__()
        self.lmax = lmax
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 4 * lmax + 2)

    def forward(self, obs):
        return torch.tanh(self.head(self.net(obs)))


class TD3CriticM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net1(x), self.net2(x)


class TD3TrainerM:
    name = "m_td3"

    def __init__(self, config, lr=3.0e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, buffer_size=200000, batch_size=128,
                 exploration_noise=0.1, hidden=256):
        self.config = config
        self.lmax = lmax_for(config)
        self.obs_dim = local_obs_dim(config)
        self.action_dim = 4 * self.lmax + 2
        self.actor = TD3ActorM(self.obs_dim, self.lmax, hidden).to(DEVICE)
        self.actor_target = TD3ActorM(self.obs_dim, self.lmax, hidden).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic1_target = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2 = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic2_target = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.steps = 0

    def _raw_action(self, obs_vec, noise=0.0):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise > 0.0:
            raw = raw + np.random.normal(0.0, noise, size=raw.shape)
        return raw

    def _decode(self, raw):
        lmax = self.lmax
        scores = raw[2:2 + 3 * lmax].reshape(lmax, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = np.clip((raw[2 + 3 * lmax:2 + 4 * lmax] + 1.0) / 2.0, 0.0, 1.0)
        move = raw[:2]
        return targets, ratios, move

    def act(self, obs_vec, noise=0.0):
        raw = self._raw_action(obs_vec, noise=noise)
        targets, ratios, move = self._decode(raw)
        return targets, ratios, move, raw

    def store(self, state, raw_action, reward, next_state, done):
        self.buffer.append((state, raw_action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        actions = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(DEVICE)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.clamp(torch.normal(0.0, self.policy_noise, size=next_actions.shape).to(DEVICE),
                                -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            n1a, n1b = self.critic1_target(next_states, next_actions)
            q_next = torch.min(n1a, n1b).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * q_next
        q1a, q1b = self.critic1(states, actions)
        loss = F.mse_loss(q1a.squeeze(-1), target) + F.mse_loss(q1b.squeeze(-1), target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.steps += 1
        if self.steps % self.policy_freq == 0:
            aa, ab = self.critic1(states, self.actor(states))
            actor_loss = -torch.min(aa, ab).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        return float(loss.item())


# ---------------------------------------------------------------------------
# SAC (same continuous action space as TD3)
# ---------------------------------------------------------------------------
class SACActorM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs, with_logprob=False):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -20.0, 2.0)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if with_logprob:
            action = dist.rsample()
            logp = dist.log_prob(action).sum(-1)
        else:
            action = mean
            logp = None
        return torch.tanh(action), logp


class SACTrainerM:
    name = "m_sac"

    def __init__(self, config, lr=3.0e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=200000, batch_size=128, hidden=256):
        self.config = config
        self.lmax = lmax_for(config)
        self.obs_dim = local_obs_dim(config)
        self.action_dim = 4 * self.lmax + 2
        self.actor = SACActorM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic1 = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic2 = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic1_target = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic2_target = TD3CriticM(self.obs_dim, self.action_dim, hidden).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.steps = 0
        self.target_entropy = -float(self.action_dim)

    def _decode(self, raw):
        lmax = self.lmax
        scores = raw[2:2 + 3 * lmax].reshape(lmax, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = np.clip((raw[2 + 3 * lmax:2 + 4 * lmax] + 1.0) / 2.0, 0.0, 1.0)
        move = raw[:2]
        return targets, ratios, move

    def act(self, obs_vec, noise=0.0):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if noise > 0.0:
                action, _ = self.actor(obs_t, with_logprob=True)
            else:
                action, _ = self.actor(obs_t)
            raw = action.squeeze(0).cpu().numpy()
        targets, ratios, move = self._decode(raw)
        return targets, ratios, move, raw

    def store(self, state, raw_action, reward, next_state, done):
        self.buffer.append((state, raw_action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        actions = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(DEVICE)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            next_actions, next_logp = self.actor(next_states, with_logprob=True)
            n1a, n1b = self.critic1_target(next_states, next_actions)
            q_next = torch.min(n1a, n1b).squeeze(-1) - self.log_alpha.exp() * next_logp
            target = rewards + self.gamma * (1.0 - dones) * q_next
        q1a, q1b = self.critic1(states, actions)
        loss = F.mse_loss(q1a.squeeze(-1), target) + F.mse_loss(q1b.squeeze(-1), target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        actions_pi, logp_pi = self.actor(states, with_logprob=True)
        pa, pb = self.critic1(states, actions_pi)
        q_pi = torch.min(pa, pb).squeeze(-1)
        actor_loss = (self.log_alpha.exp().detach() * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha.exp() * (logp_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        self.steps += 1
        return float(loss.item())


# ---------------------------------------------------------------------------
# Shared training / evaluation loop
# ---------------------------------------------------------------------------
def make_trainer(method, config):
    if method == "m_dqn":
        return DQNTrainerM(config)
    if method == "m_td3":
        return TD3TrainerM(config)
    if method == "m_sac":
        return SACTrainerM(config)
    raise ValueError(f"unknown method {method}")


def act_global(trainer, obs, config, assoc, lmax, train=True):
    """Evaluate the shared policy for all K UAVs and merge into a global action."""
    K = int(config.uavs)
    U = len(obs["task_bits"])
    moves = np.zeros((K, 2), dtype=float)
    targets = np.zeros(U, dtype=int)
    ratios = np.zeros(U, dtype=float)
    states, raw_actions = [], []
    for k in range(K):
        local = make_local_obs(obs, config, k, assoc, lmax)
        states.append(local)
        idx = np.where(assoc == k)[0]
        if trainer.name == "m_dqn":
            noise = 0.05 if train else 0.0
            t_k, r_k = trainer.act(local, epsilon=noise,
                                   rng=np.random.default_rng(np.random.randint(0, 2**31)))
            move_k = trainer._move_heuristic(obs, config, k, assoc)
            raw_actions.append(np.stack([r_k, t_k.astype(float)], axis=-1))
        elif trainer.name == "m_sac":
            t_k, r_k, move_k, raw = trainer.act(local, noise=1.0 if train else 0.0)
            raw_actions.append(raw)
        else:
            noise = trainer.exploration_noise if train else 0.0
            t_k, r_k, move_k, raw = trainer.act(local, noise=noise)
            raw_actions.append(raw)
        moves[k] = move_k * config.uav_speed_max
        n = len(idx)
        targets[idx] = t_k[:n]
        ratios[idx] = r_k[:n]
    return {"move": moves, "targets": targets, "ratios": ratios}, states, raw_actions


def train_multi_uav_drl(method, config, steps, eval_env=None, eval_every=5000,
                        log_path=None, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = make_trainer(method, config)
    from .env import UavLeoEnv
    env = UavLeoEnv(config)
    lmax = lmax_for(config)
    rng = np.random.default_rng(seed)
    obs = env.reset(seed=config.seed + int(rng.integers(0, 1000000)))
    total_reward = 0.0
    ep_count = 0
    best_score = float("-inf")
    if log_path is not None:
        log_path = Path(log_path)
        log_path.mkdir(parents=True, exist_ok=True)
    for step in range(1, steps + 1):
        assoc = associate_users(obs, config)
        action, states, raws = act_global(trainer, obs, config, assoc, lmax, train=True)
        next_obs, reward, done, _ = env.step(action)
        next_assoc = associate_users(next_obs, config)
        for k in range(len(states)):
            next_local = make_local_obs(next_obs, config, k, next_assoc, lmax)
            trainer.store(states[k], raws[k], reward, next_local, done)
        trainer.train_step()
        total_reward += reward
        obs = next_obs
        if done:
            if log_path is not None and ep_count % 10 == 0:
                _append_csv(log_path / "train_progress.csv",
                            {"step": step, "train_episode_reward": float(total_reward)})
            obs = env.reset(seed=config.seed + ep_count + 1)
            total_reward = 0.0
            ep_count += 1
        if eval_env is not None and step % eval_every == 0:
            score = evaluate_multi_uav(trainer, eval_env, config, episodes=5)
            if log_path is not None:
                _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
            if score > best_score:
                best_score = score
                torch.save({"state_dict": _state_dict(trainer), "step": step,
                            "eval_reward": score}, log_path / "model_best.pt")
            print(f"[{method}] step {step}: eval_reward = {score:.1f} (best {best_score:.1f})", flush=True)
    return trainer


def _state_dict(trainer):
    if trainer.name == "m_dqn":
        return trainer.q_net.state_dict()
    return trainer.actor.state_dict()


def evaluate_multi_uav(trainer, env, config, episodes=5, seed=None):
    from .multi_uav import associate_users as _assoc
    lmax = lmax_for(config)
    scores = []
    for ep in range(episodes):
        obs = env.reset(seed=(seed if seed is not None else config.seed + 1000 + ep))
        ep_reward = 0.0
        done = False
        while not done:
            assoc = _assoc(obs, config)
            action, _, _ = act_global(trainer, obs, config, assoc, lmax, train=False)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        scores.append(ep_reward)
    return float(np.mean(scores))
