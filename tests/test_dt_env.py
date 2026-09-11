"""Unit tests for the digital-twin layer (TwinnedWorld + predictors + adaptive sync)."""
import numpy as np
import pytest

from uav_leo_experiment.config import make_config
from uav_leo_experiment.env import UavLeoEnv as Env
from digital_twin_uav_leo.dt_env import TwinnedWorld
from digital_twin_uav_leo.predictors import predict
from digital_twin_uav_leo.adaptive_sync import estimate_error, should_resync


def _cfg(seed=1):
    return make_config("v2x_hotspot_hard", ablation="none", seed=seed,
                       episodes=2, uavs=1)


def _env(cfg, seed=1):
    env = Env(cfg)
    env.reset(seed=seed)
    return env


def test_sync_mirrors_physical_obs():
    cfg = _cfg()
    env = _env(cfg)
    tw = TwinnedWorld(env, tau=2, predictor="linear", eps=0.0,
                      rng=np.random.default_rng(0))
    obs = env.observation()
    tw.sync(obs)
    tw_obs = tw.get_obs()
    assert np.allclose(tw_obs["uav_pos"], obs["uav_pos"])
    assert np.allclose(tw_obs["user_pos"], obs["user_pos"])
    assert np.allclose(tw_obs["task_bits"], obs["task_bits"])
    assert tw.since_sync == 0
    assert tw.sync_count == 1


def test_no_sync_between_sync_points():
    cfg = _cfg()
    env = _env(cfg)
    tw = TwinnedWorld(env, tau=2, predictor="linear", eps=0.0,
                      rng=np.random.default_rng(0))
    tw.sync(env.observation())
    p0 = tw.get_obs()["user_pos"].copy()
    tw.step_forward(dt=1.0)
    p1 = tw.get_obs()["user_pos"]
    assert p1.shape == p0.shape
    assert tw.since_sync == 1


def test_linear_user_extrapolation():
    cfg = _cfg()
    env = _env(cfg)
    pred = {"user_pos": np.array([[100.0, 100.0]]),
            "user_vel": np.array([[2.0, 0.0]]),
            "hotspot_pos": None, "hotspot_vel": None}
    out = predict("linear", pred, dt=1.0, cfg=cfg, rng=None, eps=0.0)
    assert np.allclose(out["user_pos"], [[102.0, 100.0]])


def test_linear_noise_injects_error():
    cfg = _cfg()
    env = _env(cfg)
    pred = {"user_pos": np.zeros((4, 2)), "user_vel": np.zeros((4, 2)),
            "hotspot_pos": None, "hotspot_vel": None}
    rng = np.random.default_rng(0)
    out1 = predict("linear_noise", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.1)
    out2 = predict("linear_noise", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.1)
    assert not np.allclose(out1["user_pos"], out2["user_pos"])


def test_freeze_keeps_positions():
    cfg = _cfg()
    env = _env(cfg)
    pred = {"user_pos": np.array([[1.0, 2.0]]),
            "user_vel": np.array([[9.0, 9.0]]),
            "hotspot_pos": None, "hotspot_vel": None}
    out = predict("freeze", pred, dt=1.0, cfg=cfg, rng=None, eps=0.0)
    assert np.allclose(out["user_pos"], [[1.0, 2.0]])


def test_probe_error_and_threshold():
    obs_pred = {"user_pos": np.zeros((6, 2)), "hotspot_pos": None}
    obs_true = {"user_pos": np.zeros((6, 2)), "hotspot_pos": None}
    err = estimate_error(obs_pred, obs_true, n_probe=6,
                         rng=np.random.default_rng(0))
    assert err < 1e-9
    assert not should_resync(err, delta=5.0)
    obs_true["user_pos"] = np.full((6, 2), 10.0)
    err = estimate_error(obs_pred, obs_true, n_probe=6,
                         rng=np.random.default_rng(0))
    assert err > 5.0
    assert should_resync(err, delta=5.0)


def test_adaptive_sync_triggers_on_high_error():
    cfg = _cfg()
    env = _env(cfg)
    tw = TwinnedWorld(env, tau=999, predictor="freeze", eps=0.0,
                      rng=np.random.default_rng(0), sync_mode="adaptive",
                      probe_m=1, probe_n=3, delta=1.0)
    tw.sync(env.observation())
    # 物理世界推进 5 个时隙（用户移动），孪生冻结 -> 误差必然超阈值
    for _ in range(5):
        env.step({"move": np.zeros(2), "targets": np.ones(env.config.users, dtype=int),
                  "ratios": np.zeros(env.config.users)})
        obs_phys = env.observation()
        tw._maybe_probe_and_resync(obs_phys)
    assert tw.sync_count > 1


def _burst_cfg(seed=1):
    return make_config("v2x_hotspot_hard", ablation="none", seed=seed,
                       episodes=2, uavs=1, users=12, leos=4, horizon=30,
                       burst_cycle=10, burst_on=2, burst_offset=3,
                       burst_arrival_mult=4.0, burst_size_mult=2.0)


def test_burst_phase_marking():
    cfg = _burst_cfg()
    env = _env(cfg)
    for t in range(30):
        env.t = t
        expect = 3 <= (t % 10) < 5
        assert env._in_burst() == expect, (t, env._in_burst())


def test_burst_raises_task_load():
    cfg = _burst_cfg()
    env = _env(cfg)
    env.reset(seed=7)
    burst_counts, calm_counts = [], []
    for t in range(cfg.horizon):
        env.t = t
        env._sample_tasks()
        n = int((env.task_bits > 0).sum())
        (burst_counts if env._in_burst() else calm_counts).append(n)
    assert np.mean(burst_counts) > np.mean(calm_counts)
    # burst tasks are also bigger (size boost x2)
    assert env.task_bits.max() <= cfg.task_bits_max * cfg.burst_size_mult * 1.001


def test_resample_predictor_burst_mirrors_env():
    cfg = _burst_cfg()
    env = _env(cfg)
    env.reset(seed=7)
    pred = {"user_pos": np.tile(np.array([500.0, 500.0]), (12, 1)),
            "user_vel": np.zeros((12, 2)),
            "hotspot_pos": np.array([[500.0, 500.0]]),
            "hotspot_vel": np.zeros((1, 2))}
    rng = np.random.default_rng(0)
    out_burst = predict("resample", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.0, t=4)
    out_calm = predict("resample", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.0, t=9)
    assert float(out_burst["task_bits"].sum()) > float(out_calm["task_bits"].sum())


def test_twin_tracks_time_for_burst():
    cfg = _burst_cfg()
    env = _env(cfg)
    tw = TwinnedWorld(env, tau=3, predictor="linear", eps=0.0,
                      rng=np.random.default_rng(0))
    tw.sync(env.observation())
    assert tw._t == 0
    tw.step_forward(dt=1.0)
    assert tw._t == 1
    env.t = 10
    tw.sync(env.observation())
    assert tw._t == 10


def test_load_error_separates_burst_from_calm_noise():
    cfg = _burst_cfg()
    rng = np.random.default_rng(0)
    users = 12
    # calm twin vs calm truth: two independent i.i.d. samples of the calm regime
    obs_calm1 = {"user_pos": np.zeros((users, 2)),
                 "hotspot_pos": np.array([[500.0, 500.0]]),
                 "task_bits": np.array([2.0e6, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    obs_calm2 = {"user_pos": np.zeros((users, 2)),
                 "hotspot_pos": np.array([[500.0, 500.0]]),
                 "task_bits": np.array([0.0, 0.0, 1.5e6, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    # calm twin vs burst truth: load regime jumps (~8x)
    obs_burst = {"user_pos": np.zeros((users, 2)),
                 "hotspot_pos": np.array([[500.0, 500.0]]),
                 "task_bits": np.array([4.0e6] * 8 + [0.0] * 4)}
    e_calm = estimate_error(obs_calm1, obs_calm2, 3, rng, cfg=cfg, task_mode="load")
    e_burst = estimate_error(obs_calm1, obs_burst, 3, rng, cfg=cfg, task_mode="load")
    # burst transition must be far above calm noise in the load metric
    assert e_burst > 2 * e_calm, (e_calm, e_burst)


def test_random_burst_offsets_vary_by_seed():
    base = dict(users=12, leos=4, horizon=30, burst_cycle=15, burst_on=3,
                burst_offset=4, burst_arrival_mult=4.0, burst_size_mult=2.0,
                burst_random=True)
    cfg1 = make_config("v2x_hotspot_hard", ablation="none", seed=1,
                       episodes=2, uavs=1, **base)
    cfg2 = make_config("v2x_hotspot_hard", ablation="none", seed=1,
                       episodes=2, uavs=1, **base)
    e1, e2 = Env(cfg1), Env(cfg2)
    e1.reset(seed=7)
    e2.reset(seed=8)
    assert not np.array_equal(e1._burst_offsets, e2._burst_offsets)
    # twin-side predictor treats random burst as unknown (non-burst sampling)
    rng = np.random.default_rng(0)
    pred = {"user_pos": np.tile(np.array([500.0, 500.0]), (12, 1)),
            "user_vel": np.zeros((12, 2)),
            "hotspot_pos": np.array([[500.0, 500.0]]),
            "hotspot_vel": np.zeros((1, 2))}
    out = predict("resample", pred, dt=1.0, cfg=cfg1, rng=rng, eps=0.0, t=11)
    assert not np.allclose(out["task_bits"], np.zeros(12))
