"""Integration smoke test for twin-driven episode evaluation."""
import numpy as np

from uav_leo_experiment.config import make_config
from uav_leo_experiment.env import UavLeoEnv as Env
from uav_leo_experiment.baselines import ExpertExactOffloadPolicy
from digital_twin_uav_leo.runner import run_dt_episode


def _cfg(seed=1):
    return make_config("v2x_hotspot_hard", ablation="none", seed=seed,
                       episodes=2, uavs=1)


def test_run_dt_episode_smoke():
    cfg = _cfg()
    env = Env(cfg)
    policy = ExpertExactOffloadPolicy(offload_at="post_move")
    out = run_dt_episode(env, policy, cfg, tau=3, predictor="linear",
                         eps=0.0, sync_mode="fixed", seed=1)
    for k in ("reward", "latency", "energy", "completion", "agree", "syncs"):
        assert k in out
    assert 0.0 <= out["completion"] <= 1.0
    assert 0.0 <= out["agree"] <= 1.0
    assert out["syncs"] >= 1


def test_tau1_matches_no_twin():
    cfg = _cfg()
    env = Env(cfg)
    policy = ExpertExactOffloadPolicy(offload_at="post_move")
    out1 = run_dt_episode(env, policy, cfg, tau=1, predictor="linear",
                          eps=0.0, sync_mode="fixed", seed=1)
    # 无孪生：直接用物理 obs 决策
    env2 = Env(cfg)
    obs = env2.reset(seed=1)
    rng = np.random.default_rng(0)
    tot = 0.0
    done = False
    while not done:
        a = policy.act(obs, rng, cfg)
        obs, r, done, _ = env2.step(a)
        tot += r
    assert abs(out1["reward"] - tot) < 1e-6
