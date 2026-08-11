"""Deep reinforcement learning agents for the UAV-LEO offloading experiment.

Implements two standard DRL baselines used across the MEC / V2X literature:

- :class:`PPOTrainer` — proximal policy optimization with a hybrid discrete
  (offload target) + continuous (offload ratio, UAV move) actor.
- :class:`DQNTrainer`  — deep Q-network over the per-user (target, ratio)
  action grid; UAV trajectory follows the TEA heuristic (standard DQN
  offloading baseline in the literature).

Both agents are wrapped as :class:`BasePolicy` subclasses so they plug into the
existing ``run_experiment.py`` evaluation pipeline.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import BasePolicy, OptimizedPartialPolicy
from .physics import nearest_leo_index, rate_uav_leo, rate_user_uav, rate_uav_leo_vec, rate_user_uav_vec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Shared TEA expert used to build the action-prior observation features.
_TEA_EXPERT = OptimizedPartialPolicy()

# Fixed user dimension so a single trained model can serve any difficulty.
MAX_USERS = 16

# Discrete offload-ratio grid shared by the expert policies and the actor head.
RATIO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)


def _ratio_bin(ratio):
    """Map a continuous ratio to the nearest grid bin index."""
    return int(np.argmin(np.abs(RATIO_GRID - float(np.clip(ratio, 0.0, 1.0)))))


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
def make_obs_vec(config, obs):
    """Normalized flat observation vector."""
    area = config.area_size
    uav = np.asarray(obs["uav_pos"], dtype=float) / area
    nearest_leo = _nearest_leo(config, obs)
    feats = [
        uav,                                     # 2
        (nearest_leo - np.asarray(obs["uav_pos"], dtype=float)) / area,  # 2
        np.array([obs["uav_backlog_norm"]]),     # 1
        np.array([obs["leo_backlog_norm"]]),     # 1
        np.array([obs["uav_battery_norm"]]),     # 1
        np.array([obs["t"] / config.horizon]),   # 1
    ]
    if config.hotspot_motion and obs.get("hotspot_pos") is not None:
        hotspot_pos = np.asarray(obs["hotspot_pos"], dtype=float) / area
        hotspot_vel = np.asarray(obs["hotspot_vel"], dtype=float) / max(config.hotspot_speed, 1.0)
        feats.append(hotspot_pos)                       # 2
        feats.append(hotspot_vel)                       # 2
        feats.append(np.array([np.linalg.norm(uav - hotspot_pos)]))  # 1
    user_pos = np.asarray(obs["user_pos"], dtype=float) / area
    vel = np.asarray(obs["user_vel"], dtype=float) / max(config.user_speed_max, 1.0)
    bits = np.asarray(obs["task_bits"], dtype=float) / config.task_bits_max
    cpb = np.asarray(obs["cycles_per_bit"], dtype=float) / config.cycles_per_bit_max
    has_task = (np.asarray(obs["task_bits"], dtype=float) > 1.0e-6).astype(float)
    dist_uav = np.linalg.norm(user_pos - np.asarray(obs["uav_pos"], dtype=float) / area, axis=1)
    rates_uav = rate_user_uav_vec(config, obs["user_pos"], obs["uav_pos"]) / config.bandwidth_hz
    d_to_leo = np.linalg.norm(np.asarray(obs["user_pos"], dtype=float)[:, None, :]
                              - np.asarray(obs["leo_pos"], dtype=float)[None, :, :], axis=-1)
    leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
    rates_leo = np.minimum(rate_user_uav_vec(config, obs["user_pos"], obs["uav_pos"]),
                           rate_uav_leo_vec(config, obs["uav_pos"], leo_for_user)) / config.bandwidth_hz
    local_lat = (bits * config.task_bits_max * cpb * config.cycles_per_bit_max) / config.user_cpu_cycles_per_s / config.success_deadline_s
    for u in range(config.users):
        feats.append(np.concatenate([
            user_pos[u],            # 2
            vel[u],                 # 2
            [bits[u], cpb[u], has_task[u], dist_uav[u], rates_uav[u], rates_leo[u], local_lat[u]],
        ]))                          # 7 -> 11 per user
    # pad to MAX_USERS and mark validity
    for _ in range(MAX_USERS - config.users):
        feats.append(np.zeros(11, dtype=np.float32))
    valid = np.zeros(MAX_USERS, dtype=np.float32)
    valid[: config.users] = 1.0
    feats.append(valid)
    # TEA expert action prior (3U target one-hot + U ratio + 2 move)
    tea_action = _tea_action_fast(obs, config)
    tea_t = np.zeros(MAX_USERS, dtype=int)
    tea_t[: config.users] = np.asarray(tea_action["targets"], dtype=int)
    tea_r = np.zeros(MAX_USERS, dtype=np.float32)
    tea_r[: config.users] = np.asarray(tea_action["ratios"], dtype=np.float32)
    tea_targets = np.eye(3, dtype=np.float32)[tea_t]
    tea_ratios = tea_r
    tea_move = np.asarray(tea_action["move"], dtype=np.float32) / max(config.uav_speed_max, 1.0)
    feats.append(tea_targets.reshape(-1))
    feats.append(tea_ratios)
    feats.append(tea_move)
    return np.concatenate(feats).astype(np.float32)


def _tea_action_fast(obs, config):
    """Cheap TEA suggestion: one-shot per-user target/ratio grid + centroid move."""
    from .baselines import _limited_move
    targets, ratios, _ = _TEA_EXPERT._choose_targets(obs, config, obs["uav_pos"])
    workload = obs["task_bits"] * obs["cycles_per_bit"]
    if workload.sum() > 1.0e-9:
        target = np.average(obs["user_pos"], axis=0, weights=workload)
    else:
        target = obs["user_pos"].mean(axis=0)
    move = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
    return {"targets": targets, "ratios": ratios, "move": move}


def _nearest_leo(config, obs):
    uav = np.asarray(obs["uav_pos"], dtype=float)
    d = np.linalg.norm(np.asarray(obs["leo_pos"], dtype=float) - uav, axis=1)
    return np.asarray(obs["leo_pos"], dtype=float)[int(np.argmin(d))]


def obs_dim(config):
    hotspot_extra = 5 if config.hotspot_motion else 0
    return 8 + hotspot_extra + 15 * MAX_USERS + 2 + MAX_USERS


def action_dim(config):
    return 4 * MAX_USERS + 2


def decode_action(config, target_logits, ratio_logits, move_raw):
    """Convert raw network outputs to an env action dict (numpy)."""
    U = config.users
    targets = np.argmax(target_logits, axis=-1).astype(int)
    ratio_bins = np.argmax(ratio_logits, axis=-1)
    ratios = RATIO_GRID[ratio_bins]
    move = np.tanh(move_raw) * config.uav_speed_max
    return {"targets": targets, "ratios": ratios, "move": move}


# ---------------------------------------------------------------------------
# Actor-critic network (PPO)
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, U, hidden=256):
        super().__init__()
        self.U = MAX_USERS
        self.real_users = U
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, 3 * self.U + 5 * self.U + 2)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        features = self.net(obs)
        raw = self.policy_head(features)
        target_logits = raw[:, : 3 * self.U].view(-1, self.U, 3)
        ratio_logits = raw[:, 3 * self.U : 8 * self.U].view(-1, self.U, 5)
        move_raw = raw[:, 8 * self.U : 8 * self.U + 2]
        value = self.value_head(features).squeeze(-1)
        return target_logits, ratio_logits, move_raw, value

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            target_logits, ratio_logits, move_raw, _ = self.forward(obs)
            t_dist = torch.distributions.Categorical(logits=target_logits)
            r_dist = torch.distributions.Categorical(logits=ratio_logits)
            if deterministic:
                targets = torch.argmax(target_logits, dim=-1)
                ratio_bins = torch.argmax(ratio_logits, dim=-1)
                move = torch.tanh(move_raw)
            else:
                targets = t_dist.sample()
                ratio_bins = r_dist.sample()
                move_std = torch.ones_like(move_raw) * 0.25
                move_dist = torch.distributions.Normal(move_raw, move_std)
                move = torch.tanh(move_dist.sample())
            grid_t = torch.from_numpy(RATIO_GRID).to(DEVICE)
            ratios = grid_t[ratio_bins]
            return targets.cpu().numpy(), ratios.cpu().numpy(), move.cpu().numpy()


# ---------------------------------------------------------------------------
# PPO trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    def __init__(self, config, hidden=256, lr=3.0e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.003, epochs=4,
                 minibatch=128, rollout_steps=512, move_std=0.25):
        self.config = config
        self.U = config.users
        self.hidden = hidden
        self.policy = self._make_policy().to(DEVICE)
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.rollout_steps = rollout_steps
        self.move_std = move_std
        self.reward_rms = 1.0
        self.reward_mean = 0.0

    def _make_policy(self):
        """Factory for the actor-critic network (overridden by graph variants)."""
        return ActorCritic(obs_dim(self.config), self.U, hidden=self.hidden)

    @torch.no_grad()
    def _rollout(self, env, steps):
        config = self.config
        obs = env.reset(seed=config.seed + int(np.random.randint(0, 1000000)))
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        total_reward = 0.0
        grid_t = torch.from_numpy(RATIO_GRID).to(DEVICE)
        for _ in range(steps):
            obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
            target_logits, ratio_logits, move_raw, value = self.policy(obs_t)
            t_dist = torch.distributions.Categorical(logits=target_logits)
            r_dist = torch.distributions.Categorical(logits=ratio_logits)
            targets = t_dist.sample()
            ratio_bins = r_dist.sample()
            move_dist = torch.distributions.Normal(move_raw, torch.ones_like(move_raw) * self.move_std)
            move_raw_sample = move_dist.sample()
            moves = torch.tanh(move_raw_sample)
            mask_t = torch.from_numpy(self.user_mask).to(DEVICE)
            lp = ((t_dist.log_prob(targets) * mask_t).sum(-1)
                  + (r_dist.log_prob(ratio_bins) * mask_t).sum(-1)
                  + move_dist.log_prob(move_raw_sample).sum(-1))
            targets_np = targets.squeeze(0).cpu().numpy()
            ratio_bins_np = ratio_bins.squeeze(0).cpu().numpy()
            ratios_np = grid_t[ratio_bins.squeeze(0)].cpu().numpy()
            moves_np = torch.tanh(move_raw_sample).squeeze(0).cpu().numpy()
            action = {
                "targets": targets_np[: self.U].astype(int),
                "ratios": ratios_np[: self.U],
                "move": (moves_np * config.uav_speed_max)[: 2],
            }
            next_obs, reward, done, _ = env.step(action)
            states.append(make_obs_vec(config, obs))
            actions.append(np.concatenate([
                targets_np.astype(np.float32),
                ratio_bins_np.astype(np.float32),
                move_raw_sample.squeeze(0).cpu().numpy(),
            ]))
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze(0).cpu().numpy())
            log_probs.append(lp.squeeze(0).cpu().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                obs = env.reset(seed=config.seed + int(np.random.randint(0, 100000)))
        return states, actions, rewards, dones, values, log_probs, total_reward

    def _compute_advantages(self, rewards, dones, values, last_value):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        returns = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]
        return advantages, returns

    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0,
              kl_anchor=None, kl_coef=0.0, critic_warmup_updates=0):
        config = self.config
        torch.manual_seed(seed)
        np.random.seed(seed)
        anchor_policy = None
        if kl_anchor is not None:
            anchor_policy = self._make_policy()
            anchor_policy.load_state_dict(kl_anchor)
            anchor_policy.to(DEVICE)
            anchor_policy.eval()
        self.critic_warmup_updates = critic_warmup_updates
        progress = []
        step = 0
        update = 0
        while step < steps:
            states, actions, rewards, dones, values, log_probs, ep_reward = self._rollout(env, self.rollout_steps)
            step += self.rollout_steps
            last_obs = torch.from_numpy(make_obs_vec(config, env.observation())).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                last_value = self.policy(last_obs)[3].squeeze(0).cpu().numpy()
            advantages, returns = self._compute_advantages(rewards, dones, values, last_value)
            # normalize advantages
            adv_std = float(np.std(advantages)) + 1.0e-6
            advantages = (advantages - float(np.mean(advantages))) / adv_std
            returns = returns / max(float(np.std(returns)), 1.0e-6)

            states_np = np.stack(states)
            actions_np = np.stack(actions)
            targets_np = actions_np[:, : MAX_USERS].astype(np.int64)
            ratio_bins_np = actions_np[:, MAX_USERS : 2 * MAX_USERS].astype(np.int64)
            moves_np = actions_np[:, 2 * MAX_USERS : 2 * MAX_USERS + 2]

            for _ in range(self.epochs):
                perm = np.random.permutation(len(states_np))
                for i in range(0, len(states_np), self.minibatch):
                    idx = perm[i : i + self.minibatch]
                    obs_b = torch.from_numpy(states_np[idx]).to(DEVICE)
                    old_lp = torch.from_numpy(np.stack(log_probs)[idx]).to(DEVICE)
                    adv_b = torch.from_numpy(advantages[idx]).to(DEVICE)
                    ret_b = torch.from_numpy(returns[idx]).to(DEVICE)
                    tgt_b = torch.from_numpy(targets_np[idx]).to(DEVICE)
                    rat_b = torch.from_numpy(ratio_bins_np[idx]).to(DEVICE)
                    mov_b = torch.from_numpy(moves_np[idx]).to(DEVICE)

                    target_logits, ratio_logits, move_raw, value = self.policy(obs_b)
                    mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
                    t_dist = torch.distributions.Categorical(logits=target_logits)
                    logp_t = (t_dist.log_prob(tgt_b) * mask_b).sum(-1)
                    r_dist = torch.distributions.Categorical(logits=ratio_logits)
                    logp_r = (r_dist.log_prob(rat_b) * mask_b).sum(-1)
                    move_dist = torch.distributions.Normal(move_raw, torch.ones_like(move_raw) * self.move_std)
                    logp_m = move_dist.log_prob(mov_b).sum(-1)
                    logp = logp_t + logp_r + logp_m
                    ratio = torch.exp(logp - old_lp)
                    ratio = torch.clamp(ratio, 0.0, 10.0)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, ret_b)
                    entropy = t_dist.entropy().mean() + r_dist.entropy().mean() + move_dist.entropy().mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    kl_term = 0.0
                    if anchor_policy is not None:
                        with torch.no_grad():
                            a_t_logits, a_r_logits, a_m_raw, _ = anchor_policy(obs_b)
                        kld = (F.kl_div(F.log_softmax(target_logits, -1),
                                        F.softmax(a_t_logits, -1), reduction="none").sum(-1) * mask_b).sum(-1) / self.U
                        kld = kld + (F.kl_div(F.log_softmax(ratio_logits, -1),
                                              F.softmax(a_r_logits, -1), reduction="none").sum(-1) * mask_b).sum(-1) / self.U
                        kld = kld + ((move_raw - a_m_raw) ** 2).sum(-1) / (2.0 * 0.25 ** 2)
                        kl_term = kl_coef * kld.mean()
                    if getattr(self, "critic_warmup_updates", 0) > 0 and update < self.critic_warmup_updates:
                        # Keep the policy anchored to the BC reference while the
                        # value head learns the return scale, then release it.
                        loss = value_loss
                        if anchor_policy is not None:
                            loss = loss + max(kl_coef, 1.0) * kld.mean()
                    else:
                        loss = loss + kl_term
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            update += 1
            if log_path is not None and update % 5 == 0:
                progress.append({"step": step, "train_episode_reward": float(ep_reward)})
                _append_csv(log_path / "train_progress.csv", progress[-1])
            if eval_env is not None and (step % eval_every < self.rollout_steps):
                score = evaluate_policy(self, eval_env, episodes=5, deterministic=True)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float('inf')
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.policy.state_dict(), "U": self.U,
                                    "eval_reward": score}, best_path)
        return self

    def pretrain_expert(self, env, steps=3000, lr=1.0e-3, batch_size=128, seed=0,
                        expert=None):
        """Behavior-clone an expert policy (default TEA) to bootstrap the actor."""
        config = self.config
        if expert is None or isinstance(expert, str):
            from .baselines import (DeadlineAwarePartialPolicy, EnergyGuardedPartialPolicy,
                                    FollowCentroidTeaPolicy, FullOffloadTeaPolicy, OptimizedPartialPolicy,
                                    PredictTeaPolicy)
            expert_pool = {
                "tea": OptimizedPartialPolicy,
                "deadline_tea": DeadlineAwarePartialPolicy,
                "energy_guarded_tea": EnergyGuardedPartialPolicy,
                "full_offload_tea": FullOffloadTeaPolicy,
                "follow_tea": FollowCentroidTeaPolicy,
                "predict_tea": PredictTeaPolicy,
            }
            expert = expert_pool.get(expert, OptimizedPartialPolicy)()
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=config.seed + seed)
        states, tgt, rat, mov = [], [], [], []
        pad_t = np.zeros(MAX_USERS, dtype=np.int64)
        pad_r = np.zeros(MAX_USERS, dtype=np.int64)
        for _ in range(steps):
            action = expert.act(obs, rng, config)
            states.append(make_obs_vec(config, obs))
            t_t = pad_t.copy()
            t_t[: self.U] = np.asarray(action["targets"], dtype=np.int64)
            r_t = pad_r.copy()
            for u in range(self.U):
                r_t[u] = _ratio_bin(action["ratios"][u])
            m_t = np.zeros(2, dtype=np.float32)
            m_t[:] = np.asarray(action["move"], dtype=np.float32) / max(config.uav_speed_max, 1.0)
            tgt.append(t_t)
            rat.append(r_t)
            mov.append(m_t)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset(seed=config.seed + seed + int(np.random.randint(0, 100000)))
        states = np.stack(states)
        tgt = np.stack(tgt)
        rat = np.stack(rat)
        mov = np.stack(mov)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy.train()
        mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
        # Train several passes over the collected expert dataset. A single pass
        # (steps // batch_size updates) is far too few for a 3+5-way per-user
        # action head to separate from the majority class.
        n_iters = int(np.clip(steps / batch_size * 4.0, 200, 800))
        for step in range(n_iters):
            idx = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
            obs_b = torch.from_numpy(states[idx]).to(DEVICE)
            tgt_b = torch.from_numpy(tgt[idx]).to(DEVICE)
            rat_b = torch.from_numpy(rat[idx]).to(DEVICE)
            mov_b = torch.from_numpy(mov[idx]).to(DEVICE)
            target_logits, ratio_logits, move_raw, _ = self.policy(obs_b)
            t_logits_flat = target_logits.reshape(-1, MAX_USERS, 3)
            tgt_flat = tgt_b.reshape(-1, MAX_USERS)
            ce_t = F.cross_entropy(t_logits_flat.reshape(-1, 3), tgt_flat.reshape(-1),
                                   reduction="none").view(-1, MAX_USERS)
            r_logits_flat = ratio_logits.reshape(-1, MAX_USERS, 5)
            rat_flat = rat_b.reshape(-1, MAX_USERS)
            ce_r = F.cross_entropy(r_logits_flat.reshape(-1, 5), rat_flat.reshape(-1),
                                   reduction="none").view(-1, MAX_USERS)
            loss = ((ce_t + ce_r) * mask_b).sum() / max(self.U * len(idx), 1)
            loss = loss + 12.0 * F.mse_loss(torch.tanh(move_raw), mov_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.policy.eval()
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(), "U": self.U, "obs_dim": obs_dim(self.config)}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.policy.load_state_dict(data["state_dict"])
        trainer.policy.to(DEVICE)
        return trainer


class PPOPolicy(BasePolicy):
    """Evaluation wrapper: PPO agent as a BasePolicy."""

    name = "ppo"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = PPOTrainer.load(self.model_path, config)
        obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
        targets, ratios, moves = self.trainer.policy.act(obs_t, deterministic=True)
        return {
            "targets": targets[0][: config.users].astype(int),
            "ratios": ratios[0][: config.users],
            "move": moves[0] * config.uav_speed_max,
        }


# ---------------------------------------------------------------------------
# DQN trainer
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    def __init__(self, obs_dim, U, n_actions=15, hidden=256):
        super().__init__()
        self.U = MAX_USERS
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.U * n_actions),
        )

    def forward(self, obs):
        return self.net(obs).view(-1, self.U, self.n_actions)


class DQNTrainer:
    name = "dqn"
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    def __init__(self, config, lr=1.0e-3, gamma=0.99, batch_size=128, buffer_size=100000,
                 target_update=500, eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
                 double_q=False):
        self.config = config
        self.U = config.users
        self.n_actions = 3 * len(self.ratio_grid)
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0
        self.q_net = DQN(obs_dim(config), self.U, self.n_actions).to(DEVICE)
        self.target_net = DQN(obs_dim(config), self.U, self.n_actions).to(DEVICE)
        self.double_q = double_q
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.target_update = target_update
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

    def _action_grid(self):
        grid = []
        for t in range(3):
            for r in self.ratio_grid:
                grid.append((t, float(r)))
        return grid

    def act(self, obs_vec, epsilon=None, rng=None):
        """Per-user epsilon-greedy over the (target, ratio) grid."""
        eps = self.eps if epsilon is None else epsilon
        rng = rng or np.random.default_rng(0)
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.q_net(obs_t).squeeze(0).cpu().numpy()  # (U, 15)
        grid = self._action_grid()
        targets = np.zeros(MAX_USERS, dtype=int)
        ratios = np.zeros(MAX_USERS)
        for u in range(self.U):
            if rng.uniform() < eps:
                t, r = grid[int(rng.integers(0, len(grid)))]
            else:
                t, r = grid[int(np.argmax(q[u]))]
            targets[u] = t
            ratios[u] = r
        return targets[: self.U], ratios[: self.U]

    def _move_heuristic(self, obs, config):
        tea = OptimizedPartialPolicy()
        action = tea.act(obs, np.random.default_rng(0), config)
        return action["move"]

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)
        # pad per-user actions to MAX_USERS so they align with the Q head
        raw_targets = np.stack([b[1][0] for b in batch])
        raw_ratios = np.stack([b[1][1] for b in batch])
        targets = np.zeros((len(batch), MAX_USERS), dtype=np.int64)
        ratios = np.zeros((len(batch), MAX_USERS), dtype=np.float32)
        targets[:, : self.U] = raw_targets
        ratios[:, : self.U] = raw_ratios
        targets = torch.from_numpy(targets).long().to(DEVICE)
        ratios = torch.from_numpy(ratios).float().to(DEVICE)
        action_ids = torch.zeros_like(targets)
        for a in range(len(self.ratio_grid)):
            action_ids = torch.where((targets == 0) & (ratios == self.ratio_grid[a]), a, action_ids)
            action_ids = torch.where((targets == 1) & (ratios == self.ratio_grid[a]), len(self.ratio_grid) + a, action_ids)
            action_ids = torch.where((targets == 2) & (ratios == self.ratio_grid[a]), 2 * len(self.ratio_grid) + a, action_ids)
        # pad to the fixed MAX_USERS action dimension used by the Q-network
        if action_ids.shape[1] < MAX_USERS:
            padded = torch.zeros(action_ids.shape[0], MAX_USERS, dtype=action_ids.dtype, device=action_ids.device)
            padded[:, : action_ids.shape[1]] = action_ids
            action_ids = padded
        q = self.q_net(states)
        mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
        q_values = (q.gather(2, action_ids.unsqueeze(-1)).squeeze(-1) * mask_b).sum(-1)
        with torch.no_grad():
            if self.double_q:
                # Double DQN: online net selects the greedy action, target net
                # evaluates it (reduces the overestimation bias of DQN).
                next_online = self.q_net(next_states)
                next_ids = next_online.argmax(-1).unsqueeze(-1)
                next_q = (self.target_net(next_states).gather(2, next_ids).squeeze(-1) * mask_b).sum(-1)
            else:
                next_q = (self.target_net(next_states).max(-1).values * mask_b).sum(-1)
        target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        return float(loss.item())

    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0):
        config = self.config
        np.random.seed(seed)
        obs = env.reset(seed=config.seed + int(np.random.randint(0, 1000000)))
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        ep_count = 0
        for step in range(1, steps + 1):
            obs_vec = make_obs_vec(config, obs)
            targets, ratios = self.act(obs_vec, rng=rng)
            move = self._move_heuristic(obs, config)
            next_obs, reward, done, _ = env.step({"targets": targets, "ratios": ratios, "move": move})
            self.store(obs_vec, (targets, ratios), reward, make_obs_vec(config, next_obs), done)
            total_reward += reward
            loss = self.train_step()
            obs = next_obs
            if done:
                if log_path is not None and ep_count % 10 == 0:
                    _append_csv(log_path / "train_progress.csv", {"step": step, "train_episode_reward": float(total_reward)})
                obs = env.reset(seed=config.seed + ep_count + 1)
                total_reward = 0.0
                ep_count += 1
            if eval_env is not None and step % eval_every == 0:
                score = self.evaluate(eval_env, episodes=5)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float('-inf')
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.q_net.state_dict(), "U": self.U,
                                    "eval_reward": score}, best_path)
        return self

    def evaluate(self, env, episodes=5, seed=None):
        config = self.config
        if seed is None:
            seed = config.seed
        scores = []
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                targets, ratios = self.act(make_obs_vec(config, obs), epsilon=0.0)
                move = self._move_heuristic(obs, config)
                obs, r, done, _ = env.step({"targets": targets, "ratios": ratios, "move": move})
                total += r
            scores.append(total)
        return float(np.mean(scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.q_net.state_dict(), "U": self.U}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.q_net.load_state_dict(data["state_dict"])
        trainer.target_net.load_state_dict(data["state_dict"])
        return trainer


class DDQNTrainer(DQNTrainer):
    """Double DQN: online network selects the argmax action whose value is then
    evaluated by the target network, mitigating DQN's overestimation bias."""

    name = "ddqn"

    def __init__(self, config, **kwargs):
        super().__init__(config, double_q=True, **kwargs)



class DQNPolicy(BasePolicy):
    """Evaluation wrapper: DQN agent as a BasePolicy."""

    name = "dqn"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = DQNTrainer.load(self.model_path, config)
        targets, ratios = self.trainer.act(make_obs_vec(config, obs), epsilon=0.0)
        tea = OptimizedPartialPolicy()
        action = tea.act(obs, rng, config)
        return {"targets": targets[: config.users].astype(int), "ratios": ratios[: config.users], "move": action["move"]}




class DDQNPolicy(DQNPolicy):
    """Evaluation wrapper: DDQN agent as a BasePolicy."""

    name = "ddqn"

    def __init__(self, model_path=None, trainer=None):
        super().__init__(model_path=model_path, trainer=trainer)

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = DDQNTrainer.load(self.model_path, config)
        targets, ratios = self.trainer.act(make_obs_vec(config, obs), epsilon=0.0)
        tea = OptimizedPartialPolicy()
        action = tea.act(obs, rng, config)
        return {"targets": targets[: config.users].astype(int), "ratios": ratios[: config.users], "move": action["move"]}

# ---------------------------------------------------------------------------
# TD3 (Twin Delayed DDPG) ? continuous trajectory + offloading control
# ---------------------------------------------------------------------------
class TD3Actor(nn.Module):
    def __init__(self, obs_dim, U, hidden=256):
        super().__init__()
        self.U = MAX_USERS
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 4 * self.U + 2)

    def forward(self, obs):
        return torch.tanh(self.head(self.net(obs)))


class TD3Critic(nn.Module):
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


class TD3Trainer:
    """Standard TD3 over the raw (target-score, ratio, move) action vector."""

    name = "td3"

    def __init__(self, config, lr=3.0e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, buffer_size=200000, batch_size=128,
                 exploration_noise=0.1, hidden=256):
        self.config = config
        self.U = config.users
        self.action_dim = 4 * MAX_USERS + 2
        self.actor = TD3Actor(obs_dim(config), self.U, hidden).to(DEVICE)
        self.actor_target = TD3Actor(obs_dim(config), self.U, hidden).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic1_target = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2 = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic2_target = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
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
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0

    def _raw_action(self, obs_vec, noise=0.0, rng=None):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise > 0.0:
            raw = raw + np.random.normal(0.0, noise, size=raw.shape)
        return raw

    def _decode(self, raw):
        U = self.U
        scores = raw[2:2 + 3 * MAX_USERS].reshape(MAX_USERS, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = (raw[2 + 3 * MAX_USERS:2 + 4 * MAX_USERS] + 1.0) / 2.0
        ratios = np.clip(ratios, 0.0, 1.0)
        move = raw[:2]
        return targets, ratios, move

    def act(self, obs_vec, noise=0.0, rng=None):
        raw = self._raw_action(obs_vec, noise=noise)
        targets, ratios, move = self._decode(raw)
        return targets[: self.U].astype(int), ratios[: self.U], move, raw

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
            noise = torch.normal(0.0, self.policy_noise, size=next_actions.shape).to(DEVICE)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            q1_next, q2_next = self.critic1_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * q_next
        q1, q2 = self.critic1(states, actions)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        self.steps += 1
        if self.steps % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states))[0].mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        return float(loss.item())

    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0):
        config = self.config
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=config.seed + int(rng.integers(0, 1000000)))
        total_reward = 0.0
        ep_count = 0
        best_score = float("-inf")
        for step in range(1, steps + 1):
            obs_vec = make_obs_vec(config, obs)
            targets, ratios, move, raw = self.act(obs_vec, noise=self.exploration_noise)
            next_obs, reward, done, _ = env.step({
                "targets": targets, "ratios": ratios,
                "move": move * config.uav_speed_max,
            })
            self.store(obs_vec, raw, reward, make_obs_vec(config, next_obs), done)
            total_reward += reward
            self.train_step()
            obs = next_obs
            if done:
                if log_path is not None and ep_count % 10 == 0:
                    _append_csv(log_path / "train_progress.csv", {"step": step, "train_episode_reward": float(total_reward)})
                obs = env.reset(seed=config.seed + ep_count + 1)
                total_reward = 0.0
                ep_count += 1
            if eval_env is not None and step % eval_every == 0:
                score = self.evaluate(eval_env, episodes=5)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float("-inf")
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.actor.state_dict(), "U": self.U,
                                    "eval_reward": score}, best_path)
        return self

    def evaluate(self, env, episodes=5, seed=None):
        config = self.config
        if seed is None:
            seed = config.seed
        scores = []
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                targets, ratios, move, _ = self.act(make_obs_vec(config, obs), noise=0.0)
                obs, r, done, _ = env.step({"targets": targets, "ratios": ratios,
                                            "move": move * config.uav_speed_max})
                total += r
            scores.append(total)
        return float(np.mean(scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.actor.state_dict(), "U": self.U}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.actor.load_state_dict(data["state_dict"])
        trainer.actor_target.load_state_dict(data["state_dict"])
        return trainer


class TD3Policy(BasePolicy):
    """Evaluation wrapper: TD3 agent as a BasePolicy."""

    name = "td3"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = TD3Trainer.load(self.model_path, config)
        targets, ratios, move, _ = self.trainer.act(make_obs_vec(config, obs), noise=0.0)
        return {"targets": targets[: config.users].astype(int),
                "ratios": ratios[: config.users],
                "move": move * config.uav_speed_max}


# ---------------------------------------------------------------------------
# DDPG (Deep Deterministic Policy Gradient) - standard continuous DRL baseline
# from the MEC/V2X literature. Shares the TD3 action layout: (move[2],
# target-scores[3U], ratio[U]); targets by argmax, ratios by (x+1)/2.
# ---------------------------------------------------------------------------
class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class DDPGTrainer:
    """Standard DDPG over the raw (target-score, ratio, move) action vector."""

    name = "ddpg"

    def __init__(self, config, lr=3.0e-4, gamma=0.99, tau=0.005,
                 buffer_size=200000, batch_size=128, exploration_noise=0.15,
                 hidden=256):
        self.config = config
        self.U = config.users
        self.action_dim = 4 * MAX_USERS + 2
        self.actor = TD3Actor(obs_dim(config), self.U, hidden).to(DEVICE)
        self.actor_target = TD3Actor(obs_dim(config), self.U, hidden).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = DDPGCritic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic_target = DDPGCritic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.steps = 0
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0

    def _raw_action(self, obs_vec, noise=0.0, rng=None):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = self.actor(obs_t).squeeze(0).cpu().numpy()
        if noise > 0.0:
            raw = raw + np.random.normal(0.0, noise, size=raw.shape)
        return raw

    def _decode(self, raw):
        U = self.U
        scores = raw[2:2 + 3 * MAX_USERS].reshape(MAX_USERS, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = (raw[2 + 3 * MAX_USERS:2 + 4 * MAX_USERS] + 1.0) / 2.0
        ratios = np.clip(ratios, 0.0, 1.0)
        move = raw[:2]
        return targets, ratios, move

    def act(self, obs_vec, noise=0.0, rng=None):
        raw = self._raw_action(obs_vec, noise=noise)
        targets, ratios, move = self._decode(raw)
        return targets[: self.U].astype(int), ratios[: self.U], move, raw

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
            q_next = self.critic_target(next_states, next_actions).squeeze(-1)
            target = rewards + self.gamma * (1.0 - dones) * q_next
        q = self.critic(states, actions).squeeze(-1)
        loss = F.mse_loss(q, target)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.steps += 1
        return float(loss.item())

    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0):
        config = self.config
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=config.seed + int(rng.integers(0, 1000000)))
        total_reward = 0.0
        ep_count = 0
        for step in range(1, steps + 1):
            obs_vec = make_obs_vec(config, obs)
            targets, ratios, move, raw = self.act(obs_vec, noise=self.exploration_noise)
            next_obs, reward, done, _ = env.step({
                "targets": targets, "ratios": ratios,
                "move": move * config.uav_speed_max,
            })
            self.store(obs_vec, raw, reward, make_obs_vec(config, next_obs), done)
            total_reward += reward
            self.train_step()
            obs = next_obs
            if done:
                if log_path is not None and ep_count % 10 == 0:
                    _append_csv(log_path / "train_progress.csv", {"step": step, "train_episode_reward": float(total_reward)})
                obs = env.reset(seed=config.seed + ep_count + 1)
                total_reward = 0.0
                ep_count += 1
            if eval_env is not None and step % eval_every == 0:
                score = self.evaluate(eval_env, episodes=5)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float("-inf")
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.actor.state_dict(), "U": self.U,
                                    "eval_reward": score}, best_path)
        return self

    def evaluate(self, env, episodes=5, seed=None):
        config = self.config
        if seed is None:
            seed = config.seed
        scores = []
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                targets, ratios, move, _ = self.act(make_obs_vec(config, obs), noise=0.0)
                obs, r, done, _ = env.step({"targets": targets, "ratios": ratios,
                                            "move": move * config.uav_speed_max})
                total += r
            scores.append(total)
        return float(np.mean(scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.actor.state_dict(), "U": self.U}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.actor.load_state_dict(data["state_dict"])
        trainer.actor_target.load_state_dict(data["state_dict"])
        return trainer


class DDPGPolicy(BasePolicy):
    """Evaluation wrapper: DDPG agent as a BasePolicy."""

    name = "ddpg"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = DDPGTrainer.load(self.model_path, config)
        targets, ratios, move, _ = self.trainer.act(make_obs_vec(config, obs), noise=0.0)
        return {"targets": targets[: config.users].astype(int),
                "ratios": ratios[: config.users],
                "move": move * config.uav_speed_max}


# ---------------------------------------------------------------------------
# SAC (Soft Actor-Critic) - standard continuous DRL baseline from the MEC/V2X
# literature. Shares the TD3 action layout: (move[2], target-scores[3U],
# ratio[U]); targets are decoded by argmax, ratios by (x+1)/2.
# ---------------------------------------------------------------------------
class SACActor(nn.Module):
    def __init__(self, obs_dim, U, hidden=256, log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.U = MAX_USERS
        self.action_dim = 4 * MAX_USERS + 2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, self.action_dim)
        self.log_std_head = nn.Linear(hidden, self.action_dim)

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = (normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1.0e-6)).sum(-1)
        return action, log_prob

    def act_deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SACTrainer:
    """Soft Actor-Critic over the raw (target-score, ratio, move) vector."""

    name = "sac"

    def __init__(self, config, lr=3.0e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 batch_size=128, buffer_size=200000, hidden=256):
        self.config = config
        self.U = config.users
        self.action_dim = 4 * MAX_USERS + 2
        self.actor = SACActor(obs_dim(config), self.U, hidden).to(DEVICE)
        self.critic1 = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic2 = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic1_target = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic2_target = TD3Critic(obs_dim(config), self.action_dim, hidden).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(self.action_dim) / 2.0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = buffer_size
        self.steps = 0
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0

    def _decode(self, raw):
        U = self.U
        scores = raw[2:2 + 3 * MAX_USERS].reshape(MAX_USERS, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = (raw[2 + 3 * MAX_USERS:2 + 4 * MAX_USERS] + 1.0) / 2.0
        ratios = np.clip(ratios, 0.0, 1.0)
        move = raw[:2]
        return targets, ratios, move

    def act(self, obs_vec, deterministic=False, rng=None):
        obs_t = torch.from_numpy(obs_vec).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if deterministic:
                raw = self.actor.act_deterministic(obs_t).squeeze(0).cpu().numpy()
            else:
                raw, _ = self.actor.sample(obs_t)
                raw = raw.squeeze(0).cpu().numpy()
        targets, ratios, move = self._decode(raw)
        return targets[: self.U].astype(int), ratios[: self.U], move, raw

    def store(self, state, raw_action, reward, next_state, done):
        self.buffer.append((state, raw_action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
        batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), self.batch_size, replace=False)]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        actions = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)[0].squeeze(-1)
            q2_next = self.critic2_target(next_states, next_actions)[0].squeeze(-1)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            target = rewards + self.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)[0].squeeze(-1)
        q2 = self.critic2(states, actions)[0].squeeze(-1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)[0].squeeze(-1)
        q2_new = self.critic2(states, new_actions)[0].squeeze(-1)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.log_alpha.exp().detach() * log_probs - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for target, source in [(self.critic1_target, self.critic1), (self.critic2_target, self.critic2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
        self.steps += 1
        return float(critic_loss.item())

    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0):
        config = self.config
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=config.seed + int(rng.integers(0, 1000000)))
        total_reward = 0.0
        ep_count = 0
        for step in range(1, steps + 1):
            obs_vec = make_obs_vec(config, obs)
            targets, ratios, move, raw = self.act(obs_vec, deterministic=False)
            next_obs, reward, done, _ = env.step({
                "targets": targets, "ratios": ratios,
                "move": move * config.uav_speed_max,
            })
            self.store(obs_vec, raw, reward, make_obs_vec(config, next_obs), done)
            total_reward += reward
            self.train_step()
            obs = next_obs
            if done:
                if log_path is not None and ep_count % 10 == 0:
                    _append_csv(log_path / "train_progress.csv", {"step": step, "train_episode_reward": float(total_reward)})
                obs = env.reset(seed=config.seed + ep_count + 1)
                total_reward = 0.0
                ep_count += 1
            if eval_env is not None and step % eval_every == 0:
                score = self.evaluate(eval_env, episodes=5)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float("-inf")
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.actor.state_dict(), "U": self.U,
                                    "eval_reward": score}, best_path)
        return self

    def evaluate(self, env, episodes=5, seed=None):
        config = self.config
        if seed is None:
            seed = config.seed
        scores = []
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            while not done:
                targets, ratios, move, _ = self.act(make_obs_vec(config, obs), deterministic=True)
                obs, r, done, _ = env.step({"targets": targets, "ratios": ratios,
                                            "move": move * config.uav_speed_max})
                total += r
            scores.append(total)
        return float(np.mean(scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.actor.state_dict(), "U": self.U}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.actor.load_state_dict(data["state_dict"])
        return trainer


class SACPolicy(BasePolicy):
    """Evaluation wrapper: SAC agent as a BasePolicy."""

    name = "sac"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = SACTrainer.load(self.model_path, config)
        targets, ratios, move, _ = self.trainer.act(make_obs_vec(config, obs), deterministic=True)
        return {"targets": targets[: config.users].astype(int),
                "ratios": ratios[: config.users],
                "move": move * config.uav_speed_max}


def evaluate_policy(trainer, env, episodes=5, deterministic=True, seed=None):
    """Evaluate a PPO trainer (returns mean episode reward)."""
    config = env.config
    if seed is None:
        seed = config.seed
    scores = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
            targets, ratios, moves = trainer.policy.act(obs_t, deterministic=deterministic)
            obs, r, done, _ = env.step({
                "targets": targets[0][: config.users].astype(int),
                "ratios": ratios[0][: config.users],
                "move": moves[0] * config.uav_speed_max,
            })
            total += r
        scores.append(total)
    return float(np.mean(scores))


def _append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# P-D3QN: Dueling Double DQN with Prioritized Experience Replay
#
# Recent UAV-aided MEC / vehicular edge offloading baseline (2023-2024), e.g.
# "Deep Reinforcement Learning-based Mining Task Offloading Scheme for
# Intelligent Connected Vehicles in UAV-aided MEC" (2024) and "Task Offloading
# via Prioritized Experience-Based Double Dueling DQN in Edge-Assisted IIoT"
# (IEEE IoT-T, 2024). The dueling architecture separates state value from
# per-action advantages and PER replays high-TD-error transitions more often.
# ---------------------------------------------------------------------------
class SumTree:
    """Binary-sum tree for proportional prioritized experience replay."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.data = [None] * self.capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    @property
    def total(self):
        return float(self.tree[0])

    @property
    def max_priority(self):
        if self.n_entries == 0:
            return 1.0
        return float(np.max(self.tree[self.capacity - 1:self.capacity - 1 + self.n_entries]))


class DuelingDQN(nn.Module):
    """Dueling Q-network: shared features, separate value and advantage streams."""

    def __init__(self, obs_dim, U, n_actions=15, hidden=256):
        super().__init__()
        self.U = MAX_USERS
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, self.U * n_actions)

    def forward(self, obs):
        features = self.net(obs)
        value = self.value_head(features).unsqueeze(-1)          # (B, 1, 1)
        advantage = self.advantage_head(features).view(-1, self.U, self.n_actions)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class D3QNTrainer(DQNTrainer):
    """P-D3QN (dueling + double Q + prioritized replay) offloading baseline.

    Follows Chi et al. [1] and the ICV / UAV-aided MEC context of Li et
    al. [2] (full citations: experiments/uav_leo_v2x_paper_final/
    route_a_method.md, Section 6).
    """

    name = "d3qn"
    net_class = DuelingDQN

    def __init__(self, config, lr=1.0e-3, gamma=0.99, batch_size=128, buffer_size=100000,
                 target_update=500, eps_start=1.0, eps_end=0.05, eps_decay=0.9995,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_steps=60000):
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_steps = per_beta_steps
        self.beta = per_beta_start
        self.tree = SumTree(buffer_size)
        super().__init__(config, lr=lr, gamma=gamma, batch_size=batch_size,
                         buffer_size=buffer_size, target_update=target_update,
                         eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                         double_q=True)

    def store(self, state, action, reward, next_state, done):
        self.tree.add(self.tree.max_priority, (state, action, reward, next_state, done))

    def _sample_batch(self):
        n = self.batch_size
        segment = self.tree.total / n
        batch, idxs = [], []
        for i in range(n):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, _, data = self.tree.get(s)
            idxs.append(idx)
            batch.append(data)
        probs = np.array([self.tree.tree[idx] / max(self.tree.total, 1.0) for idx in idxs])
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / max(float(weights.max()), 1.0e-8)
        return batch, idxs, weights.astype(np.float32)

    def train_step(self):
        if self.tree.n_entries < self.batch_size:
            return 0.0
        self.beta = min(1.0, self.per_beta_start
                        + (self.steps + 1) / max(self.per_beta_steps, 1)
                        * (1.0 - self.per_beta_start))
        batch, idxs, weights = self._sample_batch()
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32)).to(DEVICE)
        dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32)).to(DEVICE)
        raw_targets = np.stack([b[1][0] for b in batch])
        raw_ratios = np.stack([b[1][1] for b in batch])
        targets = np.zeros((len(batch), MAX_USERS), dtype=np.int64)
        ratios = np.zeros((len(batch), MAX_USERS), dtype=np.float32)
        targets[:, : self.U] = raw_targets
        ratios[:, : self.U] = raw_ratios
        targets = torch.from_numpy(targets).long().to(DEVICE)
        ratios = torch.from_numpy(ratios).float().to(DEVICE)
        action_ids = torch.zeros_like(targets)
        for a in range(len(self.ratio_grid)):
            action_ids = torch.where((targets == 0) & (ratios == self.ratio_grid[a]), a, action_ids)
            action_ids = torch.where((targets == 1) & (ratios == self.ratio_grid[a]), len(self.ratio_grid) + a, action_ids)
            action_ids = torch.where((targets == 2) & (ratios == self.ratio_grid[a]), 2 * len(self.ratio_grid) + a, action_ids)
        q = self.q_net(states)
        mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
        q_values = (q.gather(2, action_ids.unsqueeze(-1)).squeeze(-1) * mask_b).sum(-1)
        with torch.no_grad():
            next_online = self.q_net(next_states)
            next_ids = next_online.argmax(-1).unsqueeze(-1)
            next_q = (self.target_net(next_states).gather(2, next_ids).squeeze(-1) * mask_b).sum(-1)
            target = rewards + self.gamma * (1.0 - dones) * next_q
        td = q_values - target
        loss = (torch.from_numpy(weights).to(DEVICE) * td.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        priorities = (td.detach().abs().cpu().numpy() + 1.0e-6) ** self.per_alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, float(priority))
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        return float(loss.item())


class D3QNPolicy(DQNPolicy):
    """Evaluation wrapper: P-D3QN agent as a BasePolicy."""

    name = "d3qn"

    def __init__(self, model_path=None, trainer=None):
        super().__init__(model_path=model_path, trainer=trainer)

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = D3QNTrainer.load(self.model_path, config)
        targets, ratios = self.trainer.act(make_obs_vec(config, obs), epsilon=0.0)
        tea = OptimizedPartialPolicy()
        action = tea.act(obs, rng, config)
        return {"targets": targets[: config.users].astype(int),
                "ratios": ratios[: config.users],
                "move": action["move"]}


# ---------------------------------------------------------------------------
# Graph-attention / Transformer PPO
#
# Implements the graph-encoding DRL baselines from recent V2X / MEC offloading
# papers (2024-2025): a GAT encoder following Kim et al. [3] (cooperative
# multiagent DRL for UAV-aided MEC, IEEE IoT-J 2024) and Ullah & Han [4]
# (graph-based double-DQN for vehicular edge computing, J. Supercomputing
# 2024/2025), or a Transformer encoder following Xie et al. [5]
# (Transformer-based DRL offloading, IEEE VTC 2025-Spring), applied over the
# per-user observation block, followed by the same PPO actor-critic heads as
# the standard PPO baseline.
# ---------------------------------------------------------------------------
def split_obs_tensor(config, obs):
    """Slice the flat obs into (global, user_block, valid, tea blocks).

    Must stay in sync with make_obs_vec(): 8 global features (+5 hotspot),
    11 features per user padded to MAX_USERS, the validity mask, the TEA
    action prior (one-hot targets, ratios, move).
    """
    hotspot_extra = 5 if config.hotspot_motion else 0
    g = 8 + hotspot_extra
    user_block = obs[:, g:g + 11 * MAX_USERS].reshape(obs.shape[0], MAX_USERS, 11)
    valid = obs[:, g + 11 * MAX_USERS:g + 12 * MAX_USERS]
    tea_targets = obs[:, g + 12 * MAX_USERS:g + 15 * MAX_USERS].reshape(obs.shape[0], MAX_USERS, 3)
    tea_ratios = obs[:, g + 15 * MAX_USERS:g + 16 * MAX_USERS]
    tea_move = obs[:, g + 16 * MAX_USERS:g + 16 * MAX_USERS + 2]
    return user_block, valid, tea_targets, tea_ratios, tea_move


class GATLayer(nn.Module):
    """Multi-head graph attention over the (fully connected) user graph."""

    def __init__(self, in_dim, out_dim, heads=4, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_dim, heads))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(negative_slope)

    def forward(self, x, mask):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.out_dim)
        adj = mask.unsqueeze(1) * mask.unsqueeze(2)  # (B, N, N) valid-pair mask
        outs = []
        for hidx in range(self.heads):
            a = self.a[:, hidx]
            hi = h[:, :, hidx, :]
            pair = torch.cat([
                hi.unsqueeze(2).expand(B, N, N, self.out_dim),
                hi.unsqueeze(1).expand(B, N, N, self.out_dim),
            ], dim=-1)
            e = (pair * a).sum(-1)
            e = self.leaky(e)
            # finite negative fill instead of -inf so fully-masked (padding)
            # rows still softmax to a finite uniform vector (no NaN)
            e = e.masked_fill(adj == 0, -1e9)
            attn = torch.softmax(e, dim=-1)
            outs.append(torch.einsum("bij,bjo->bio", attn, hi))
        return torch.stack(outs, dim=-1).mean(-1)


class GraphEncoder(nn.Module):
    """GAT or Transformer encoder over the per-user node features."""

    def __init__(self, node_dim=11, hidden=96, heads=4, layers=2, mode="gat", dropout=0.0):
        super().__init__()
        self.mode = mode
        self.hidden = hidden
        self.heads = heads
        self.input_proj = nn.Linear(node_dim, hidden)
        if mode == "gat":
            self.body = nn.ModuleList(
                [GATLayer(hidden, hidden, heads) for _ in range(layers)])
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                dropout=dropout, batch_first=True)
            self.body = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, nodes, valid):
        x = torch.tanh(self.input_proj(nodes))
        if self.mode == "gat":
            for layer in self.body:
                x = layer(x, valid)
        else:
            key_pad = valid < 0.5  # True marks padding for the attention mask
            x = self.body(x, src_key_padding_mask=key_pad)
        mask = valid.unsqueeze(-1)  # (B, N, 1)
        xm = x * mask
        pooled_mean = xm.sum(1) / mask.sum(1).clamp(min=1.0)
        pooled_max = xm.masked_fill(mask == 0, -1e9).max(1).values
        return torch.cat([pooled_mean, pooled_max], dim=-1)


class GraphActorCritic(nn.Module):
    """PPO actor-critic whose policy head consumes graph-pooled user features."""

    def __init__(self, config, U, hidden=256, encoder="gat", enc_hidden=96, heads=4, layers=2):
        super().__init__()
        self.U = MAX_USERS
        self.real_users = U
        self.config = config
        self.hotspot_extra = 5 if config.hotspot_motion else 0
        self.global_dim = 8 + self.hotspot_extra
        self.encoder = GraphEncoder(node_dim=11, hidden=enc_hidden, heads=heads,
                                    layers=layers, mode=encoder)
        tea_dim = 3 * MAX_USERS + MAX_USERS + 2 + MAX_USERS  # targets + ratios + move + valid
        mlp_in = self.global_dim + 2 * enc_hidden + tea_dim
        self.net = nn.Sequential(
            nn.Linear(mlp_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, 3 * self.U + 5 * self.U + 2)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        user_block, valid, tea_targets, tea_ratios, tea_move = split_obs_tensor(self.config, obs)
        pooled = self.encoder(user_block, valid)
        global_feats = obs[:, :self.global_dim]
        tea_flat = torch.cat([tea_targets.reshape(obs.shape[0], -1), tea_ratios,
                              tea_move, valid], dim=-1)
        features = self.net(torch.cat([global_feats, pooled, tea_flat], dim=-1))
        out = self.policy_head(features)
        target_logits = out[:, :3 * self.U].reshape(-1, self.U, 3)
        ratio_logits = out[:, 3 * self.U:8 * self.U].reshape(-1, self.U, 5)
        move_raw = out[:, 8 * self.U:8 * self.U + 2]
        value = self.value_head(features)
        return target_logits, ratio_logits, move_raw, value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        target_logits, ratio_logits, move_raw, _ = self.forward(obs)
        grid_t = torch.from_numpy(RATIO_GRID).to(DEVICE)
        if deterministic:
            targets = torch.argmax(target_logits, dim=-1)
            ratio_bins = torch.argmax(ratio_logits, dim=-1)
            move = torch.tanh(move_raw)
        else:
            t_dist = torch.distributions.Categorical(logits=target_logits)
            r_dist = torch.distributions.Categorical(logits=ratio_logits)
            targets = t_dist.sample()
            ratio_bins = r_dist.sample()
            move_std = torch.ones_like(move_raw) * 0.25
            move = torch.tanh(torch.distributions.Normal(move_raw, move_std).sample())
        ratios = grid_t[ratio_bins]
        return targets.cpu().numpy(), ratios.cpu().numpy(), move.cpu().numpy()


class GraphPPOTrainer(PPOTrainer):
    """PPO with a GAT / Transformer encoder (graph-attention DRL baseline)."""

    def __init__(self, config, encoder="gat", enc_hidden=96, heads=4, layers=2,
                 hidden=256, lr=3.0e-4, **kwargs):
        self.encoder = encoder
        self.enc_hidden = enc_hidden
        self.heads = heads
        self.layers = layers
        self.lr = lr
        super().__init__(config, hidden=hidden, lr=lr, **kwargs)
        self.policy = self._make_policy().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _make_policy(self):
        return GraphActorCritic(self.config, self.U, hidden=self.hidden,
                                encoder=self.encoder, enc_hidden=self.enc_hidden,
                                heads=self.heads, layers=self.layers)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(), "U": self.U,
                    "obs_dim": obs_dim(self.config), "encoder": self.encoder,
                    "enc_hidden": self.enc_hidden, "heads": self.heads,
                    "layers": self.layers}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config, encoder=data.get("encoder", "gat"),
                      enc_hidden=data.get("enc_hidden", 96),
                      heads=data.get("heads", 4), layers=data.get("layers", 2))
        trainer.policy.load_state_dict(data["state_dict"])
        trainer.policy.to(DEVICE)
        return trainer


class GraphPPOPolicy(BasePolicy):
    """Evaluation wrapper: GAT / Transformer-PPO agent as a BasePolicy."""

    def __init__(self, model_path=None, trainer=None, name="gat_ppo"):
        self.name = name
        self.trainer = trainer
        self.model_path = model_path

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = GraphPPOTrainer.load(self.model_path, config)
        obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
        targets, ratios, moves = self.trainer.policy.act(obs_t, deterministic=True)
        return {"targets": targets[0][: config.users].astype(int),
                "ratios": ratios[0][: config.users],
                "move": moves[0] * config.uav_speed_max}
