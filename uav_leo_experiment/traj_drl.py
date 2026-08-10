"""GDRL proposed method: residual trajectory PPO + optimization-based offloading.

Per-slot offloading (target + partial ratio for every user) is solved exactly
by a per-user argmin over the environment's true per-slot cost at the post-move
UAV position -- the position that actually serves the slot.  The DRL head learns only a *residual* on the
trajectory:

    move = clip(expert_move + delta, uav_speed_max)

where ``expert_move`` is the demand-predictive centroid/hotspot tracker
(predict_tea).  At delta = 0 the policy is exactly the expert, so behavior
cloning is trivial; PPO then learns where the myopic expert trajectory is
suboptimal (hotspot anticipation, battery budgeting, queue backpressure).

Reference direction (see README):
  "DRL-based trajectory optimization and computation-aware resource allocation
   for UAV-assisted edge computing networks", IEEE 2025 -- DRL for trajectory +
   optimization for allocation.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import BasePolicy, PredictTeaPolicy
from .physics import simulate_task
from .learning import DEVICE, make_obs_vec, obs_dim

MAX_DELTA_FRAC = 1.0   # residual can add up to one full speed step


def _predict_tea_move(obs, config):
    """Cheap demand-predictive expert trajectory (predict_tea move only)."""
    workload = obs["task_bits"] * obs["cycles_per_bit"]
    if workload.sum() > 1.0e-9:
        centroid = np.average(obs["user_pos"], axis=0, weights=workload)
    else:
        centroid = obs["user_pos"].mean(axis=0)
    if obs.get("hotspot_pos") is not None:
        hotspot_next = obs["hotspot_pos"] + obs.get("hotspot_vel", np.zeros(2)) * config.slot_seconds
        target = 0.5 * centroid + 0.5 * hotspot_next
    else:
        target = centroid
    move = target - np.asarray(obs["uav_pos"], dtype=float)
    norm = float(np.linalg.norm(move))
    if norm > config.uav_speed_max and norm > 1.0e-9:
        move = move / norm * config.uav_speed_max
    return np.asarray(move, dtype=float)


class TrajActorCritic(nn.Module):
    """MLP actor-critic; the action is the 2-D residual on the expert move."""

    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.delta_head = nn.Linear(hidden, 2)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        features = self.net(obs)
        delta_raw = self.delta_head(features)
        value = self.value_head(features).squeeze(-1)
        return delta_raw, value

    def act(self, obs, deterministic=False, move_std=0.25):
        with torch.no_grad():
            delta_raw, _ = self.forward(obs)
            if deterministic:
                delta = torch.tanh(delta_raw)
            else:
                dist = torch.distributions.Normal(
                    delta_raw, torch.ones_like(delta_raw) * move_std)
                delta = torch.tanh(dist.sample())
        return delta.cpu().numpy()



def _exact_offload(obs, config, uav_pos):
    """Exact per-slot offloading: for each user, argmin over (target, ratio) of
    the environment's true per-slot cost (latency + energy_weight*energy +
    drop_penalty*dropped).  With the slot-start backlogs fixed, the per-slot
    cost is additive across users, so this per-user search is the global
    optimum for the slot given the UAV position."""
    U = config.users
    targets = np.zeros(U, dtype=int)
    ratios = np.zeros(U, dtype=float)
    if U == 0:
        return targets, ratios
    d_to_leo = np.linalg.norm(np.asarray(obs["user_pos"], dtype=float)[:, None, :]
                              - np.asarray(obs["leo_pos"], dtype=float)[None, :, :], axis=-1)
    leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
    leo_backlog = float(obs.get("leo_backlog", 0.0))
    uav_backlog = float(obs.get("uav_backlog", 0.0))
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    for u in range(U):
        if float(obs["task_bits"][u]) <= 1.0e-6:
            continue
        best_cost = float("inf")
        best = (0, 0.0)
        for target in (0, 1, 2):
            grid = ratio_grid if target > 0 else np.array([0.0])
            for r in grid:
                res = simulate_task(
                    config, obs["user_pos"][u], uav_pos, leo_for_user[u],
                    obs["task_bits"][u], obs["cycles_per_bit"][u],
                    target, r, uav_backlog, leo_backlog)
                cost = (res["latency"]
                        + config.energy_weight * res["energy"]
                        + config.drop_penalty * int(res["dropped"]))
                if cost < best_cost - 1.0e-9:
                    best_cost = cost
                    best = (target, r)
        targets[u], ratios[u] = best
    return targets, ratios

class TrajPPOTrainer:
    """PPO trainer for the residual trajectory head."""

    def __init__(self, config, hidden=256, lr=3.0e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.003, epochs=4,
                 minibatch=128, rollout_steps=256, move_std=0.25):
        self.config = config
        self.policy = TrajActorCritic(obs_dim(config), hidden=hidden).to(DEVICE)
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

    # -- helpers ------------------------------------------------------------
    def _post_move_pos(self, obs, move):
        """Replicate env.step kinematics so the offload solver sees the position
        that will actually serve this slot."""
        c = self.config
        move = np.asarray(move, dtype=float).copy()
        if c.fixed_uav:
            move = np.zeros(2, dtype=float)
        if c.uav_battery_per_slot > 0.0 and obs.get("uav_battery", 1.0) <= 0.0:
            move = np.zeros(2, dtype=float)
        norm = float(np.linalg.norm(move))
        if norm > c.uav_speed_max and norm > 1.0e-9:
            move = move / norm * c.uav_speed_max
        return np.clip(np.asarray(obs["uav_pos"], dtype=float) + move * c.slot_seconds,
                       0.0, c.area_size)

    def _offload_action(self, obs, uav_pos):
        """Exact per-slot offloading optimization at the given UAV position."""
        return _exact_offload(obs, self.config, uav_pos)
        return np.asarray(targets, dtype=int), np.asarray(ratios, dtype=float)

    def _expert_move(self, obs):
        return _predict_tea_move(obs, self.config)

    def _compose_move(self, obs, delta_norm):
        """expert move + residual, clipped to the speed limit."""
        expert = self._expert_move(obs)
        move = expert + np.asarray(delta_norm, dtype=float) * self.config.uav_speed_max * MAX_DELTA_FRAC
        norm = float(np.linalg.norm(move))
        if norm > self.config.uav_speed_max and norm > 1.0e-9:
            move = move / norm * self.config.uav_speed_max
        return move

    # -- rollout / training -------------------------------------------------
    @torch.no_grad()
    def _rollout(self, env, steps):
        config = self.config
        obs = env.reset(seed=config.seed + int(np.random.randint(0, 1000000)))
        states, deltas, rewards, dones, values, log_probs = [], [], [], [], [], []
        total_reward = 0.0
        for _ in range(steps):
            obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
            delta_raw, value = self.policy(obs_t)
            dist = torch.distributions.Normal(
                delta_raw, torch.ones_like(delta_raw) * self.move_std)
            delta_sample = dist.sample()
            lp = dist.log_prob(delta_sample).sum(-1)
            delta_norm = torch.tanh(delta_sample).squeeze(0).cpu().numpy()
            move = self._compose_move(obs, delta_norm)
            post_pos = self._post_move_pos(obs, move)
            targets, ratios = self._offload_action(obs, post_pos)
            action = {"move": move, "targets": targets, "ratios": ratios}
            next_obs, reward, done, _ = env.step(action)
            states.append(make_obs_vec(config, obs))
            deltas.append(np.concatenate([delta_norm, delta_sample.squeeze(0).cpu().numpy()]))
            rewards.append(reward)
            dones.append(done)
            values.append(value.squeeze(0).cpu().numpy())
            log_probs.append(lp.squeeze(0).cpu().numpy())
            total_reward += reward
            obs = next_obs
            if done:
                obs = env.reset(seed=config.seed + int(np.random.randint(0, 100000)))
        return states, deltas, rewards, dones, values, log_probs, total_reward

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
            anchor_policy = TrajActorCritic(obs_dim(config))
            anchor_policy.load_state_dict(kl_anchor)
            anchor_policy.to(DEVICE)
            anchor_policy.eval()
        self.critic_warmup_updates = critic_warmup_updates
        progress = []
        step = 0
        update = 0
        while step < steps:
            states, deltas, rewards, dones, values, log_probs, ep_reward = self._rollout(env, self.rollout_steps)
            step += self.rollout_steps
            last_obs = torch.from_numpy(make_obs_vec(config, env.observation())).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                last_value = self.policy(last_obs)[1].squeeze(0).cpu().numpy()
            advantages, returns = self._compute_advantages(rewards, dones, values, last_value)
            adv_std = float(np.std(advantages)) + 1.0e-6
            advantages = (advantages - float(np.mean(advantages))) / adv_std

            states_np = np.stack(states)
            delta_norm_np = np.stack([d[:2] for d in deltas]).astype(np.float32)
            delta_raw_np = np.stack([d[2:] for d in deltas]).astype(np.float32)

            for _ in range(self.epochs):
                perm = np.random.permutation(len(states_np))
                for i in range(0, len(states_np), self.minibatch):
                    idx = perm[i:i + self.minibatch]
                    obs_b = torch.from_numpy(states_np[idx]).to(DEVICE)
                    old_lp = torch.from_numpy(np.stack(log_probs)[idx]).to(DEVICE)
                    adv_b = torch.from_numpy(advantages[idx]).to(DEVICE)
                    ret_b = torch.from_numpy(returns[idx]).to(DEVICE)
                    mov_b = torch.from_numpy(delta_raw_np[idx]).to(DEVICE)

                    delta_raw, value = self.policy(obs_b)
                    move_dist = torch.distributions.Normal(
                        delta_raw, torch.ones_like(delta_raw) * self.move_std)
                    logp = move_dist.log_prob(mov_b).sum(-1)
                    ratio = torch.exp(logp - old_lp)
                    ratio = torch.clamp(ratio, 0.0, 10.0)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                        1.0 + self.clip_epsilon) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, ret_b)
                    entropy = move_dist.entropy().mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    if anchor_policy is not None:
                        with torch.no_grad():
                            a_m_raw, _ = anchor_policy(obs_b)
                        kld = ((delta_raw - a_m_raw) ** 2).sum(-1) / (2.0 * self.move_std ** 2)
                        if getattr(self, "critic_warmup_updates", 0) > 0 and update < self.critic_warmup_updates:
                            loss = value_loss + max(kl_coef, 1.0) * kld.mean()
                        else:
                            loss = loss + kl_coef * kld.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            update += 1
            if log_path is not None and update % 5 == 0:
                progress.append({"step": step, "train_episode_reward": float(ep_reward)})
                _append_csv(log_path / "train_progress.csv", progress[-1])
            if eval_env is not None and (step % eval_every < self.rollout_steps):
                score = evaluate_traj_policy(self, eval_env, episodes=5, deterministic=True)
                if log_path is not None:
                    _append_csv(log_path / "eval_progress.csv", {"step": step, "eval_reward": score})
                    best_path = Path(log_path) / "model_best.pt"
                    best = float('-inf')
                    if best_path.exists():
                        best = float(torch.load(best_path, map_location="cpu")["eval_reward"])
                    if score > best:
                        torch.save({"state_dict": self.policy.state_dict(), "eval_reward": score},
                                   best_path)
        return self

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(),
                    "obs_dim": obs_dim(self.config)}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config)
        trainer.policy.load_state_dict(data["state_dict"])
        trainer.policy.to(DEVICE)
        return trainer


def evaluate_traj_policy(trainer, env, episodes=5, deterministic=True, seed=None):
    """Mean episode reward of a TrajPPOTrainer (uses the optimization sub-solver)."""
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
            delta = trainer.policy.act(obs_t, deterministic=deterministic)[0]
            move = trainer._compose_move(obs, delta)
            post_pos = trainer._post_move_pos(obs, move)
            targets, ratios = trainer._offload_action(obs, post_pos)
            obs, r, done, _ = env.step({"move": move, "targets": targets, "ratios": ratios})
            total += r
        scores.append(total)
    return float(np.mean(scores))


class GDRLPolicy(BasePolicy):
    """Proposed GDRL policy: residual learned trajectory + optimization offloading.

    ``ablate_traj`` freezes the learned trajectory residual at zero (the policy
    then follows the demand-predictive expert trajectory while keeping the
    exact offloading layer); ``ablate_offload`` replaces the exact offloading
    layer with the TEA heuristic while keeping the learned trajectory.  The two
    switches isolate each layer's contribution for the ablation study.
    """

    name = "gdrl"

    def __init__(self, model_path=None, trainer=None, ablate_traj=False, ablate_offload=False):
        self.trainer = trainer
        self.model_path = model_path
        self.ablate_traj = bool(ablate_traj)
        self.ablate_offload = bool(ablate_offload)
        self._tea = PredictTeaPolicy() if self.ablate_offload else None
        if self.ablate_traj and self.ablate_offload:
            self.name = "gdrl_expert"
        elif self.ablate_traj:
            self.name = "gdrl_no_traj"
        elif self.ablate_offload:
            self.name = "gdrl_no_opt"

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = TrajPPOTrainer.load(self.model_path, config)
        obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
        if self.ablate_traj:
            delta = np.zeros(2, dtype=float)
        else:
            delta = self.trainer.policy.act(obs_t, deterministic=True)[0]
        move = self.trainer._compose_move(obs, delta)
        post_pos = self.trainer._post_move_pos(obs, move)
        if self.ablate_offload:
            targets, ratios, _ = self._tea._choose_targets(obs, config, post_pos)
        else:
            targets, ratios = self.trainer._offload_action(obs, post_pos)
        return {"move": move, "targets": targets, "ratios": ratios}


def _append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
