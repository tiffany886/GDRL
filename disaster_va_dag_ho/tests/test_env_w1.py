"""W1 unit tests: config, ephemeris hard-zero windows, projection, SoC,
low-battery gating, association and a full dry-run."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from disaster_va_dag_ho.env.channel import terminal_uav_rate_matrix, uav_leo_capacity_matrix
from disaster_va_dag_ho.env.config import DEFAULT_CONFIG_PATH, load_config
from disaster_va_dag_ho.env.env import DisasterEnv
from disaster_va_dag_ho.env.ephemeris import Ephemeris
from disaster_va_dag_ho.env.tasks import APP_DONE, APP_FAILED, APP_PENDING


def make_env(**overrides) -> DisasterEnv:
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides=overrides)
    return DisasterEnv(cfg)


def test_config_loads_locked_defaults():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.area_size_m == 2000.0
    assert cfg.num_terminals == 30
    assert cfg.num_uavs == 3
    assert cfg.num_leos == 2
    assert cfg.uav_altitude == 100.0
    assert cfg.slot_seconds == 1.0
    assert cfg.horizon == 50
    assert cfg.burst_start_s == 15.0 and cfg.burst_end_s == 35.0
    with pytest.raises(ValueError):
        load_config(DEFAULT_CONFIG_PATH, overrides={"not_a_key": 1})


def test_ephemeris_windows_staggered_and_gapped():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    epi = Ephemeris(cfg)
    assert epi.visible(0, 2.0) and not epi.visible(1, 2.0)  # LEO1 only
    assert epi.visible(1, 12.0) and not epi.visible(0, 12.0)  # LEO2 only
    assert not epi.visible(0, 10.0) and not epi.visible(1, 10.0)  # gap: no star
    assert epi.remaining_s(0, 2.0) == pytest.approx(6.8)
    assert epi.remaining_s(1, 10.0) == 0.0


def test_out_of_window_leo_capacity_is_hard_zero():
    env = make_env()
    hover = np.zeros((env.cfg.num_uavs, 2))
    for _ in range(10):
        _, _, _, _ = env.step(hover)
    assert env.t_s == 10.0
    cap_gap = env.current_leo_capacity()
    assert cap_gap.shape == (env.cfg.num_uavs, env.cfg.num_leos)
    assert np.all(cap_gap == 0.0)  # t=10 lies in an inter-window gap
    env.step(hover)  # t=11: LEO2 window is open
    cap_open = env.current_leo_capacity()
    assert np.all(cap_open[:, 1] > 0.0)


def test_in_window_capacity_respects_cl_cap():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    env = make_env()
    visible = np.ones((cfg.num_uavs, cfg.num_leos), dtype=bool)
    cap = uav_leo_capacity_matrix(cfg, env.uav_pos, env.leo_sub, visible)
    assert np.all(cap > 0.0)
    assert np.all(cap <= cfg.leo_capacity_bps + 1e-6)


def test_terminal_uav_rate_matrix_shape():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    env = make_env()
    rates = terminal_uav_rate_matrix(cfg, env.terminals.pos, env.uav_pos)
    assert rates.shape == (cfg.num_terminals, cfg.num_uavs)
    assert np.all(rates > 0.0)


def test_projection_clamps_and_boundary_penalty_appears():
    env = make_env(seed=1)
    actions = np.zeros((env.cfg.num_uavs, 2))
    env.uav_pos[0] = [5.0, env.cfg.area_size_m / 2]
    actions[0] = [-50.0, 0.0]  # far outside to the west
    obs, reward, done, info = env.step(actions)
    assert env.uav_pos[0, 0] >= 0.0
    assert np.all(env.uav_pos >= 0.0) and np.all(env.uav_pos <= env.cfg.area_size_m)
    assert info["penalties"]["boundary"] > 0.0


def test_soc_monotonic_decrease_with_flight_energy():
    env = make_env(seed=2)
    hover = np.zeros((env.cfg.num_uavs, 2))
    soc_before = env.uav_soc.copy()
    for _ in range(10):
        _, _, _, _ = env.step(hover)
    assert np.all(env.uav_soc < soc_before)  # hover power drains battery
    assert np.all(env.uav_soc >= 0.0)
    assert env.metrics["flight_energy_j"] > 0.0


def test_low_battery_limits_speed_and_penalises():
    env = make_env(seed=3)
    env.uav_soc[0] = 0.05 * env.cfg.soc_init_j  # well below soc_low_fraction=0.2
    env.uav_pos[0] = np.array([env.cfg.area_size_m / 2, env.cfg.area_size_m / 2])
    actions = np.zeros((env.cfg.num_uavs, 2))
    actions[0] = [env.cfg.uav_speed_max_mps, 0.0]  # requests full speed
    obs, reward, done, info = env.step(actions)
    # gated speed factor ~ soc_frac / soc_low_fraction = 0.25
    max_expected = env.cfg.uav_speed_max_mps * (0.05 / 0.2)
    start = np.array([env.cfg.area_size_m / 2, env.cfg.area_size_m / 2])
    moved = np.linalg.norm(env.uav_pos[0] - start)
    assert 0.0 < moved <= max_expected * env.cfg.slot_seconds + 1e-6
    assert info["penalties"]["low_battery"] > 0.0


def test_empty_battery_forces_hover():
    env = make_env(seed=4)
    env.uav_soc[0] = 0.0
    env.uav_pos[0] = np.array([env.cfg.area_size_m / 2, env.cfg.area_size_m / 2])
    actions = np.zeros((env.cfg.num_uavs, 2))
    actions[0] = [env.cfg.uav_speed_max_mps, 0.0]
    before = env.uav_pos.copy()
    obs, reward, done, info = env.step(actions)
    assert np.allclose(env.uav_pos[0], before[0])
    assert info["penalties"]["low_battery"] > 0.0


def test_association_covers_all_terminals_within_cap():
    env = make_env(seed=5)
    env._associate()
    assert set(np.unique(env.assignment)) <= set(range(env.cfg.num_uavs))
    assert np.all(env.assignment >= 0)
    cap = int(np.ceil(env.cfg.num_terminals / env.cfg.num_uavs))
    counts = np.bincount(env.assignment, minlength=env.cfg.num_uavs)
    assert np.all(counts <= cap)
    assert counts.sum() == env.cfg.num_terminals


def test_full_episode_dry_run_and_state_accounting():
    env = make_env(seed=7)
    rng = np.random.default_rng(0)
    done = False
    steps = 0
    while not done:
        actions = rng.uniform(-10, 10, size=(env.cfg.num_uavs, 2))
        obs, reward, done, info = env.step(actions)
        steps += 1
        assert obs.shape == (env.cfg.num_uavs, env.obs_dim)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if steps > 1000:
            pytest.fail("episode did not terminate")
    assert done and steps == env.cfg.horizon
    states = [a.state for a in env.apps]
    n_pending = sum(s == APP_PENDING for s in states)
    n_done = sum(s == APP_DONE for s in states)
    n_failed = sum(s == APP_FAILED for s in states)
    assert n_pending + n_done + n_failed == len(env.apps)
    assert env.metrics["apps_failed"] == sum(s == APP_FAILED for s in states)
    assert env.metrics["apps_done"] == n_done
    assert env.metrics["apps_arrived"] == len(env.apps)
    assert env.metrics["steps"] == env.cfg.horizon
    assert np.isfinite(env.metrics["reward_sum"])


def test_low_rate_then_burst_then_low_rate_arrivals():
    # sanity: arrival intensity differs between phases (W1 bookkeeping only)
    env = make_env(seed=11)
    hover = np.zeros((env.cfg.num_uavs, 2))
    env.step(hover)  # t=1, low phase
    arrivals_low = env.metrics["apps_arrived"]
    for _ in range(19):
        env.step(hover)  # advance to t=20 inside the burst
    arrivals_burst = env.metrics["apps_arrived"]
    assert arrivals_burst > arrivals_low
