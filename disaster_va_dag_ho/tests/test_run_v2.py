"""v2 W-B plumbing tests: fixed-trajectory mover + offload-only RL heads."""
from __future__ import annotations

from math import ceil

import numpy as np

from disaster_va_dag_ho.agents.mappo import Mappo
from disaster_va_dag_ho.agents.trainer import run_episode
from disaster_va_dag_ho.baselines.sac_flat import SacFlat
from disaster_va_dag_ho.baselines.trajectories import TPatrol
from disaster_va_dag_ho.env.config import load_config
from disaster_va_dag_ho.env.env import DisasterEnv

CONFIG = "disaster_va_dag_ho/configs/main_v2.yaml"


def _env(seed: int = 0) -> DisasterEnv:
    return DisasterEnv(load_config(CONFIG, overrides={"seed": seed}))


def _flat_policy(env: DisasterEnv, seed: int = 0) -> Mappo:
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    return Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                 num_slots=num_slots, mode="flat_no_vel", seed=seed)


def test_flat_no_vel_act_returns_zero_vel_and_cat_dec():
    env = _env(seed=1)
    policy = _flat_policy(env, seed=1)
    obs = env.observation()
    out = policy.act(obs)
    assert out["vel"].shape == (env.cfg.num_uavs, 2)
    assert np.allclose(out["vel"], 0.0)
    assert out["dec"] is not None and out["dec"].shape == (
        env.cfg.num_uavs, ceil(env.cfg.num_terminals / env.cfg.num_uavs))
    det = policy.act_deterministic(obs)
    assert np.allclose(det["vel"], 0.0)


def test_mover_moves_uavs_and_overrides_policy_vel():
    env = _env(seed=2)
    policy = _flat_policy(env, seed=2)
    patrol = TPatrol()
    obs = env.observation()
    start = env.uav_pos.copy()
    out = policy.act(obs)
    # zero-vel from the head is overridden by the patrol mover in the runner
    nobs, rew, done, info = env.step(patrol.act(env), out["dec"])
    moved = np.linalg.norm(env.uav_pos - start, axis=1)
    assert moved.max() > 1e-6  # UAVs actually moved (patrol), not hovered


def test_run_episode_with_mover_and_v2_stats():
    env = _env(seed=3)
    policy = _flat_policy(env, seed=3)
    patrol = TPatrol()
    _, stats = run_episode(env, policy, seed=3, collect=True, mover=patrol)
    assert "frac_unassociated" in stats
    assert "fail_uncovered" in stats
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert stats["apps_arrived"] >= 0


def test_offload_only_update_runs_without_vel_loss():
    env = _env(seed=4)
    policy = _flat_policy(env, seed=4)
    patrol = TPatrol()
    traj, _ = run_episode(env, policy, seed=4, collect=True, mover=patrol)
    upd = policy.update(traj)
    assert np.isfinite(upd["loss"])
    assert upd["entropy"] >= 0.0


def test_b5_sac_offload_only_fit_and_eval():
    env = _env(seed=5)
    mover = TPatrol()
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = SacFlat(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                     num_slots=num_slots, seed=5, warmup=4, batch=4,
                     alpha=0.2, offload_only=True, mover=mover)
    policy.fit(env, episodes=2, eval_every=0, eval_seeds=())
    st = policy._eval_one(env, 5)
    assert "frac_unassociated" in st and "fail_uncovered" in st
    assert 0.0 <= st["success_rate"] <= 1.0
