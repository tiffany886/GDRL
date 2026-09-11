"""Rollout + training loop helpers for the VA-DAG-HO MAPPO agent."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..env.env import DisasterEnv
from .mappo import Mappo, StepData


def run_episode(env: DisasterEnv, policy, seed: Optional[int] = None,
                collect: bool = True, deterministic: bool = False,
                max_steps: int = 1000, mover=None):
    """Run one episode; optionally record the PPO trajectory.

    ``mover`` (optional callable(env) -> (K,2) m/s) supplies the UAV motion
    for v2 fixed-trajectory rows. When given, the policy's velocity head is
    ignored by the environment (offload-only rows) even though its outputs are
    still recorded in ``StepData`` for backward-compatible training code.
    """
    obs = env.reset(seed)
    vmax = env.cfg.uav_speed_max_mps
    traj: List[StepData] = []
    ep_rew = 0.0
    entropy_sum = 0.0
    comp = {"L": 0.0, "E": 0.0, "F": 0.0,
            "pen_boundary": 0.0, "pen_collision": 0.0, "pen_low_battery": 0.0}
    steps = 0
    done = False
    while not done:
        if deterministic and hasattr(policy, "act_deterministic"):
            out = policy.act_deterministic(obs)
        else:
            out = policy.act(obs)
        if mover is None:
            vel = np.asarray(out["vel"], dtype=float) * vmax
        else:
            vel = np.asarray(
                mover.act(env) if hasattr(mover, "act") else mover(env),
                dtype=float,
            )  # raw m/s from the fixed trajectory module
        dec = out["dec"]
        if dec is not None:
            dec = np.asarray(dec, dtype=int)
        nobs, rew, done, info = env.step(vel, dec)
        if collect:
            next_value = 0.0 if done else policy._critic_value(nobs)
            traj.append(StepData(
                obs=obs,
                next_obs=nobs,
                vel=out["vel"],
                dec=dec,
                rew=float(rew),
                done=done,
                logp_vel=out["logp_vel"],
                logp_cat=out["logp_cat"],
                value=out["value"],
                next_value=next_value,
            ))
        ep_rew += float(rew)
        entropy_sum += out["entropy"]
        comp["L"] += float(info["L"])
        comp["E"] += float(info["e_joules"])
        comp["F"] += float(info["n_failed"])
        comp["pen_boundary"] += float(info["penalties"]["boundary"])
        comp["pen_collision"] += float(info["penalties"]["collision"])
        comp["pen_low_battery"] += float(info["penalties"]["low_battery"])
        obs = nobs
        steps += 1
        if steps > max_steps:
            raise RuntimeError("episode did not terminate")
    m = env.metrics
    done_n = int(m["apps_done"])
    arrived = int(m["apps_arrived"])
    stats = {
        "reward": ep_rew,
        "success_rate": done_n / max(arrived, 1),
        "apps_arrived": arrived,
        "apps_done": done_n,
        "apps_failed": int(m["apps_failed"]),
        "mean_delay_s": m["done_delay_sum_s"] / max(done_n, 1),
        "energy_j": float(m["energy_total_j"]),
        "leo_placements": int(m["leo_placements"]),
        "leo_invalid_attempts": int(m["leo_invalid_attempts"]),
        "dep_violations": int(m["dep_violations"]),
        "frac_unassociated": float(env.frac_unassociated),
        "fail_uncovered": int(m["fail_uncovered"]),
        "steps": steps,
        "mean_entropy": entropy_sum / max(steps, 1),
        "reward_L": comp["L"],
        "reward_E": comp["E"],
        "reward_F": comp["F"],
        "pen_boundary": comp["pen_boundary"],
        "pen_collision": comp["pen_collision"],
        "pen_low_battery": comp["pen_low_battery"],
    }
    return traj, stats


def train(policy: Mappo, env: DisasterEnv, episodes: int, seed: int = 0,
          eval_every: int = 20, eval_seeds: tuple = (0, 1, 2), mover=None):
    """Train and return per-episode + periodic eval rows (dicts)."""
    rows = []
    for ep in range(episodes):
        traj, stats = run_episode(env, policy, seed=(seed + ep * 7) % 1000,
                                  collect=True, mover=mover)
        upd = policy.update(traj)
        row = {"kind": "train", "episode": ep, **stats, **upd}
        rows.append(row)
        if eval_every and (ep + 1) % eval_every == 0:
            for s in eval_seeds:
                _, estats = run_episode(env, policy, seed=int(s), collect=False,
                                        deterministic=True, mover=mover)
                rows.append({"kind": "eval", "episode": ep, "eval_seed": int(s),
                             **estats})
    return rows
