"""HADO-PPO: Hybrid-Action Decision-Order-Aware PPO with temporal memory.

Design (paper-facing):
- Hybrid action space: per-user discrete offloading (target x ratio grid)
  plus a continuous 2-D UAV move.  This fixes the encoding mismatch that
  makes naive TD3/SAC (continuous outputs rounded to discrete choices) and
  flat DQN/D3QN (no trajectory learning) fail on this environment.
- Decision-order alignment (post-move): the offloading head is conditioned on
  the *post-move* UAV position (computed in-network from the sampled move),
  i.e. the position that actually serves the slot.  This is the project's C1
  mechanism built into the network, which standard joint trajectory +
  offloading RL papers do not do.
- Temporal memory (QECO-inspired): a GRU over time slots lets the policy
  remember load / hotspot / backlog dynamics instead of only the current
  observation.  Reference: QECO (D3QN+LSTM, IEEE TNSE 2025, arXiv:2311.02525).

Baselines in this repo decide offloading and trajectory from the pre-move
state (or not at all); HADO is the decision-order-aware alternative.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import BasePolicy
from .learning import DEVICE, MAX_USERS, RATIO_GRID, make_obs_vec, obs_dim


class HadoActor(nn.Module):
    """Two-stage decision-order-aware actor-critic.

    Stage 1: move head samples the continuous 2-D UAV move.
    Stage 2: offloading heads are conditioned on the post-move position,
    which is what actually serves the slot.
    """

    def __init__(self, obs_dim, U, hidden=256, use_recurrent=True):
        super().__init__()
        self.U = MAX_USERS
        self.hidden = hidden
        self.use_recurrent = use_recurrent
        self.pre = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden, hidden) if use_recurrent else None
        self.move_head = nn.Linear(hidden, 2)
        self.value_head = nn.Linear(hidden, 1)
        # decision-order: offloading context sees the post-move position
        self.offload_ctx = nn.Sequential(
            nn.Linear(hidden + 2, hidden), nn.Tanh(),
        )
        self.target_head = nn.Linear(hidden, 3 * self.U)
        self.ratio_head = nn.Linear(hidden, 5 * self.U)

    def forward(self, obs, h=None):
        feats = self.pre(obs)
        if self.gru is not None:
            h = self.gru(feats, h)
        else:
            h = feats
        move_raw = self.move_head(h)
        value = self.value_head(h).squeeze(-1)
        return move_raw, value, h

    def offload_forward(self, obs, h, move_raw, config):
        """Stage 2: offloading heads conditioned on the post-move position."""
        area = float(config.area_size)
        move = torch.tanh(move_raw)
        pos_cur = obs[:, :2]
        pos_new = pos_cur + move * (float(config.uav_speed_max)
                                    * float(config.slot_seconds) / area)
        pos_new = torch.clamp(pos_new, 0.0, 1.0)
        ctx = self.offload_ctx(torch.cat([h, pos_new], dim=-1))
        target_logits = self.target_head(ctx).view(-1, self.U, 3)
        ratio_logits = self.ratio_head(ctx).view(-1, self.U, 5)
        return target_logits, ratio_logits

    def act(self, obs_t, h=None, deterministic=False, move_std=0.25, config=None):
        """Sample / argmax a full hybrid action from a single obs tensor."""
        with torch.no_grad():
            return self._act_no_grad(obs_t, h, deterministic, move_std, config)

    def _act_no_grad(self, obs_t, h, deterministic, move_std, config):
        move_raw, _, h = self.forward(obs_t, h)
        if deterministic:
            moves = torch.tanh(move_raw)
            move_sample = move_raw
        else:
            move_dist = torch.distributions.Normal(
                move_raw, torch.ones_like(move_raw) * move_std)
            move_sample = move_dist.sample()
            moves = torch.tanh(move_sample)
        target_logits, ratio_logits = self.offload_forward(obs_t, h, move_sample, config)
        if deterministic:
            targets = torch.argmax(target_logits, dim=-1)
            ratio_bins = torch.argmax(ratio_logits, dim=-1)
        else:
            targets = torch.distributions.Categorical(logits=target_logits).sample()
            ratio_bins = torch.distributions.Categorical(logits=ratio_logits).sample()
        return (targets.cpu().numpy(), ratio_bins.cpu().numpy(),
                moves.cpu().numpy(), move_sample.cpu().numpy(), h)


class HadoPPOTrainer:
    """Recurrent PPO for the two-stage hybrid-action actor."""

    def __init__(self, config, hidden=256, lr=3.0e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4,
                 minibatch=128, rollout_steps=512, move_std=0.25, use_recurrent=True):
        self.config = config
        self.U = config.users
        self.hidden = hidden
        self.move_std = move_std
        self.use_recurrent = use_recurrent
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.minibatch = minibatch
        self.rollout_steps = rollout_steps
        self.policy = HadoActor(obs_dim(config), self.U, hidden=hidden,
                                use_recurrent=use_recurrent).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.user_mask = np.zeros(MAX_USERS, dtype=np.float32)
        self.user_mask[: self.U] = 1.0

    # ------------------------------------------------------------------
    def _zero_hidden(self, batch=1):
        return torch.zeros(batch, self.hidden, device=DEVICE)

    def _rollout(self, env, steps):
        config = self.config
        obs = env.reset(seed=config.seed + int(np.random.randint(0, 1000000)))
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        h_ins = []
        h = self._zero_hidden()
        total_reward = 0.0
        grid_t = torch.from_numpy(RATIO_GRID).to(DEVICE)
        for _ in range(steps):
            with torch.no_grad():
                obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
                move_raw, value, h = self.policy.forward(obs_t, h)
                move_dist = torch.distributions.Normal(
                    move_raw, torch.ones_like(move_raw) * self.move_std)
                move_sample = move_dist.sample()
                target_logits, ratio_logits = self.policy.offload_forward(
                    obs_t, h, move_sample, config)
                t_dist = torch.distributions.Categorical(logits=target_logits)
                r_dist = torch.distributions.Categorical(logits=ratio_logits)
                targets = t_dist.sample()
                ratio_bins = r_dist.sample()
                moves = torch.tanh(move_sample)
                mask_t = torch.from_numpy(self.user_mask).to(DEVICE)
                lp = ((t_dist.log_prob(targets) * mask_t).sum(-1)
                      + (r_dist.log_prob(ratio_bins) * mask_t).sum(-1)
                      + move_dist.log_prob(move_sample).sum(-1))
                targets_np = targets.squeeze(0).cpu().numpy()
                ratio_bins_np = ratio_bins.squeeze(0).cpu().numpy()
                ratios_np = grid_t[ratio_bins.squeeze(0)].cpu().numpy()
                moves_np = moves.squeeze(0).cpu().numpy()
                action = {
                    "targets": targets_np[: self.U].astype(int),
                    "ratios": ratios_np[: self.U],
                    "move": (moves_np * config.uav_speed_max)[: 2],
                }
                next_obs, reward, done, _ = env.step(action)
                h_ins.append(h.squeeze(0).cpu().numpy().copy())
                states.append(make_obs_vec(config, obs))
                actions.append(np.concatenate([
                    targets_np.astype(np.float32),
                    ratio_bins_np.astype(np.float32),
                    move_sample.squeeze(0).cpu().numpy(),
                ]))
                rewards.append(reward)
                dones.append(done)
                values.append(value.squeeze(0).cpu().numpy())
                log_probs.append(lp.squeeze(0).cpu().numpy())
                total_reward += reward
                obs = next_obs
                if done:
                    h = self._zero_hidden()
                    obs = env.reset(seed=config.seed + int(np.random.randint(0, 100000)))
        return (states, actions, rewards, dones, values, log_probs,
                h_ins, total_reward)

    def _recompute_hidden(self, states_np, dones):
        """Recompute the per-step hidden states with the current policy
        (standard recurrent-PPO approximation).

        ``h_ins[t]`` is the hidden state *after* processing state ``t`` --
        exactly the hidden used by ``_rollout`` when it stored the action for
        that step -- so the log-probs of the stored actions match the behavior
        distribution.
        """
        h = self._zero_hidden()
        h_ins = []
        for t in range(len(states_np)):
            obs_t = torch.from_numpy(states_np[t:t + 1]).to(DEVICE)
            _, _, h = self.policy.forward(obs_t, h)
            h_ins.append(h)
            if dones[t]:
                h = self._zero_hidden()
        return torch.cat(h_ins, 0), h

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

    # ------------------------------------------------------------------
    def pretrain_expert(self, env, steps=3000, lr=1.0e-3, batch_size=128, seed=0,
                        expert=None):
        """Behavior-clone an expert (default follow_tea) to bootstrap the actor."""
        from .baselines import (DeadlineAwarePartialPolicy, EnergyGuardedPartialPolicy,
                                FollowCentroidTeaPolicy, FullOffloadTeaPolicy,
                                OptimizedPartialPolicy, PredictTeaPolicy)
        expert_pool = {
            "tea": OptimizedPartialPolicy,
            "deadline_tea": DeadlineAwarePartialPolicy,
            "energy_guarded_tea": EnergyGuardedPartialPolicy,
            "full_offload_tea": FullOffloadTeaPolicy,
            "follow_tea": FollowCentroidTeaPolicy,
            "predict_tea": PredictTeaPolicy,
        }
        if expert is None or isinstance(expert, str):
            expert = expert_pool.get(expert, FollowCentroidTeaPolicy)()
        config = self.config
        rng = np.random.default_rng(seed)
        obs = env.reset(seed=config.seed + seed)
        states, tgt, rat, mov = [], [], [], []
        ep_rewards, ep_done = [], []
        pad_t = np.zeros(MAX_USERS, dtype=np.int64)
        pad_r = np.zeros(MAX_USERS, dtype=np.int64)
        for _ in range(steps):
            action = expert.act(obs, rng, config)
            states.append(make_obs_vec(config, obs))
            t_t = pad_t.copy()
            t_t[: self.U] = np.asarray(action["targets"], dtype=np.int64)
            r_t = pad_r.copy()
            for u in range(self.U):
                r_t[u] = int(np.argmin(np.abs(RATIO_GRID - action["ratios"][u])))
            m_t = np.zeros(2, dtype=np.float32)
            mv = np.clip(np.asarray(action["move"], dtype=np.float32) / max(config.uav_speed_max, 1.0),
                         -0.9999, 0.9999)
            m_t[:] = np.arctanh(mv)
            tgt.append(t_t)
            rat.append(r_t)
            mov.append(m_t)
            obs, rew, done, _ = env.step(action)
            ep_rewards.append(rew)
            ep_done.append(done)
            if done:
                obs = env.reset(seed=config.seed + seed + int(np.random.randint(0, 100000)))
        states = np.stack(states)
        tgt = np.stack(tgt)
        rat = np.stack(rat)
        mov = np.stack(mov)
        # Monte-Carlo returns per episode for value-head pretraining
        rets = np.zeros(len(states), dtype=np.float32)
        acc = 0.0
        for i in range(len(states) - 1, -1, -1):
            acc = acc + float(ep_rewards[i])
            rets[i] = acc
            if ep_done[i]:
                acc = 0.0
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy.train()
        mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
        n_iters = int(np.clip(steps / batch_size * 8.0, 400, 2000))
        for step in range(n_iters):
            idx = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
            obs_b = torch.from_numpy(states[idx]).to(DEVICE)
            tgt_b = torch.from_numpy(tgt[idx]).to(DEVICE)
            rat_b = torch.from_numpy(rat[idx]).to(DEVICE)
            mov_b = torch.from_numpy(mov[idx]).to(DEVICE)
            move_raw, value, h = self.policy.forward(obs_b, self._zero_hidden(batch=len(idx)))
            # condition the offload heads on the *expert's* move (stored label),
            # matching the post-move conditioning used at eval time
            t_logits, r_logits = self.policy.offload_forward(obs_b, h, mov_b, self.config)
            ce_t = F.cross_entropy(t_logits.reshape(-1, 3), tgt_b.reshape(-1),
                                   reduction="none").view(-1, MAX_USERS)
            ce_r = F.cross_entropy(r_logits.reshape(-1, 5), rat_b.reshape(-1),
                                   reduction="none").view(-1, MAX_USERS)
            loss = ((ce_t + ce_r) * mask_b).sum() / max(self.U * len(idx), 1)
            loss = loss + 1.0 * F.mse_loss(move_raw, mov_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # value-head only, on detached features (does not perturb the policy)
            value_d = self.policy.value_head(h.detach()).squeeze(-1)
            loss_v = F.mse_loss(value_d, torch.from_numpy(rets[idx]).to(DEVICE))
            opt.zero_grad()
            loss_v.backward()
            opt.step()
        self.policy.eval()
        return self

    # ------------------------------------------------------------------
    def train(self, env, steps, eval_env=None, eval_every=5000, log_path=None, seed=0,
              kl_anchor=None, kl_coef=0.0, critic_warmup_updates=0):
        config = self.config
        torch.manual_seed(seed)
        np.random.seed(seed)
        anchor_policy = None
        if kl_anchor is not None:
            anchor_policy = HadoActor(obs_dim(config), self.U, hidden=self.hidden,
                                      use_recurrent=self.use_recurrent)
            anchor_policy.load_state_dict(kl_anchor)
            anchor_policy.to(DEVICE)
            anchor_policy.eval()
        warmup_left = max(int(critic_warmup_updates), 0)
        progress = []
        step = 0
        while step < steps:
            (states, actions, rewards, dones, values, log_probs,
             h_ins, total_reward) = self._rollout(env, self.rollout_steps)
            step += self.rollout_steps
            with torch.no_grad():
                last_h = self._recompute_hidden(np.stack(states), dones)[1]
                last_obs = torch.from_numpy(make_obs_vec(config, env.observation())).unsqueeze(0).to(DEVICE)
                last_value = self.policy.forward(last_obs, last_h)[1].squeeze(0).cpu().numpy()
            advantages, returns = self._compute_advantages(rewards, dones, values, last_value)
            adv_std = float(np.std(advantages)) + 1.0e-6
            advantages = (advantages - float(np.mean(advantages))) / adv_std
            returns = returns / max(float(np.std(returns)), 1.0e-6)

            states_np = np.stack(states)
            actions_np = np.stack(actions)
            dones_np = np.stack(dones)
            targets_np = actions_np[:, : MAX_USERS].astype(np.int64)
            ratio_bins_np = actions_np[:, MAX_USERS: 2 * MAX_USERS].astype(np.int64)
            moves_np = actions_np[:, 2 * MAX_USERS: 2 * MAX_USERS + 2]

            with torch.no_grad():
                h_in_all, _ = self._recompute_hidden(states_np, dones_np)

            for _ in range(self.epochs):
                perm = np.random.permutation(len(states_np))
                for i in range(0, len(states_np), self.minibatch):
                    idx = perm[i: i + self.minibatch]
                    obs_b = torch.from_numpy(states_np[idx]).to(DEVICE)
                    h_b = h_in_all[idx].to(DEVICE)
                    old_lp = torch.from_numpy(np.stack(log_probs)[idx]).to(DEVICE)
                    adv_b = torch.from_numpy(advantages[idx]).to(DEVICE)
                    ret_b = torch.from_numpy(returns[idx]).to(DEVICE)
                    tgt_b = torch.from_numpy(targets_np[idx]).to(DEVICE)
                    rat_b = torch.from_numpy(ratio_bins_np[idx]).to(DEVICE)
                    mov_b = torch.from_numpy(moves_np[idx]).to(DEVICE)
                    move_raw, value, _ = self.policy.forward(obs_b, h_b)
                    # Decision-order alignment: the offloading heads are
                    # conditioned on the *executed* (sampled) move, matching
                    # the rollout; move_raw still trains the move head via
                    # the Gaussian log-prob below.
                    t_logits, r_logits = self.policy.offload_forward(obs_b, h_b, mov_b, config)
                    mask_b = torch.from_numpy(self.user_mask).to(DEVICE)
                    t_dist = torch.distributions.Categorical(logits=t_logits)
                    logp_t = (t_dist.log_prob(tgt_b) * mask_b).sum(-1)
                    r_dist = torch.distributions.Categorical(logits=r_logits)
                    logp_r = (r_dist.log_prob(rat_b) * mask_b).sum(-1)
                    move_dist = torch.distributions.Normal(
                        move_raw, torch.ones_like(move_raw) * self.move_std)
                    logp_m = move_dist.log_prob(mov_b).sum(-1)
                    logp = logp_t + logp_r + logp_m
                    ratio = torch.exp(logp - old_lp)
                    ratio = torch.clamp(ratio, 0.0, 10.0)
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                        1.0 + self.clip_epsilon) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, ret_b)
                    entropy = (t_dist.entropy().mean() + r_dist.entropy().mean()
                               + move_dist.entropy().mean())
                    if anchor_policy is not None and kl_coef > 0.0:
                        with torch.no_grad():
                            a_move, _, _ = anchor_policy.forward(obs_b, h_b)
                            a_t, a_r = anchor_policy.offload_forward(obs_b, h_b, mov_b, config)
                        kl = (F.kl_div(F.log_softmax(t_logits, -1), F.softmax(a_t, -1), reduction="batchmean")
                              + F.kl_div(F.log_softmax(r_logits, -1), F.softmax(a_r, -1), reduction="batchmean")
                              + ((move_raw - a_move) ** 2 / (2.0 * self.move_std ** 2)).mean())
                    else:
                        kl = torch.zeros((), device=DEVICE)
                    if warmup_left > 0:
                        # critic-only warmup: learn the value function before
                        # moving the policy (same trick as the PPO baseline)
                        loss = self.value_coef * value_loss
                    else:
                        loss = (policy_loss + self.value_coef * value_loss
                                - self.entropy_coef * entropy + kl_coef * kl)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2.0)
                    self.optimizer.step()
                    if warmup_left > 0:
                        warmup_left -= 1
            progress.append({"step": step, "rollout_reward": total_reward})
            if eval_env is not None and (step % eval_every < self.rollout_steps):
                eval_score = self.evaluate(eval_env, episodes=5)
                print(f"[hado] step={step} rollout={total_reward:.2f} "
                      f"eval={eval_score:.2f}", flush=True)
                progress[-1]["eval_reward"] = eval_score
                if log_path is not None:
                    best_path = Path(log_path) / "model_best.pt"
                    if (not best_path.exists()
                            or eval_score > float(torch.load(best_path, map_location="cpu")["eval_reward"])):
                        torch.save({"state_dict": self.policy.state_dict(),
                                    "U": self.U, "eval_reward": eval_score,
                                    "use_recurrent": self.use_recurrent}, best_path)
            if log_path is not None:
                _append_csv(Path(log_path) / "progress.csv", progress[-1])
        return self

    # ------------------------------------------------------------------
    def evaluate(self, env, episodes=5, deterministic=True, seed=None):
        config = env.config
        if seed is None:
            seed = config.seed
        scores = []
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            done = False
            total = 0.0
            h = self._zero_hidden()
            while not done:
                obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
                targets, ratio_bins, moves, _, h = self.policy.act(
                    obs_t, h, deterministic=deterministic,
                    move_std=self.move_std, config=config)
                ratios = RATIO_GRID[ratio_bins[0]]
                obs, r, done, _ = env.step({
                    "targets": targets[0][: config.users].astype(int),
                    "ratios": ratios[: config.users],
                    "move": moves[0] * config.uav_speed_max,
                })
                total += r
            scores.append(total)
        return float(np.mean(scores))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.policy.state_dict(), "U": self.U,
                    "use_recurrent": self.use_recurrent,
                    "obs_dim": obs_dim(self.config)}, path)

    @classmethod
    def load(cls, path, config):
        data = torch.load(path, map_location=DEVICE)
        trainer = cls(config, use_recurrent=bool(data.get("use_recurrent", True)))
        trainer.policy.load_state_dict(data["state_dict"])
        trainer.policy.to(DEVICE)
        return trainer


class HadoPolicy(BasePolicy):
    """Evaluation wrapper: HADO agent as a BasePolicy (run_experiment.py)."""

    name = "hado"

    def __init__(self, model_path=None, trainer=None):
        self.trainer = trainer
        self.model_path = model_path
        self._h = None

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = HadoPPOTrainer.load(self.model_path, config)
        if self._h is None or float(obs.get("t", 0.0)) <= 0.0:
            self._h = self.trainer._zero_hidden()
        obs_t = torch.from_numpy(make_obs_vec(config, obs)).unsqueeze(0).to(DEVICE)
        targets, ratio_bins, moves, _, self._h = self.trainer.policy.act(
            obs_t, self._h, deterministic=True,
            move_std=self.trainer.move_std, config=config)
        ratios = RATIO_GRID[ratio_bins[0]]
        return {
            "targets": targets[0][: config.users].astype(int),
            "ratios": ratios[: config.users],
            "move": moves[0] * config.uav_speed_max,
        }


def evaluate_hado(trainer, env, episodes=5, deterministic=True, seed=None):
    """Module-level helper mirroring evaluate_policy for the PPO baselines."""
    return trainer.evaluate(env, episodes=episodes, deterministic=deterministic, seed=seed)


def _append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
