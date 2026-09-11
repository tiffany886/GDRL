"""v2 spec-W-A unit tests: hard coverage, rule A, hotspot arrivals, T-patrol.

Mirrors SPEC_V2_DS.md §1.1-§1.5 / §4 W-A acceptance bullets.
"""
from __future__ import annotations

import numpy as np
import pytest

from disaster_va_dag_ho.agents.association import associate_terminals
from disaster_va_dag_ho.baselines.trajectories import TPatrol, make_trajectory
from disaster_va_dag_ho.env.channel import hard_cover_rates, terminal_uav_cover_mask
from disaster_va_dag_ho.env.config import load_config
from disaster_va_dag_ho.env.env import DisasterEnv
from disaster_va_dag_ho.env.tasks import (
    APP_FAILED,
    APP_PENDING,
    AppRequest,
    ArrivalProcess,
)
from disaster_va_dag_ho.scheduler.dag import generate_dag

CONFIG = "disaster_va_dag_ho/configs/main_v2.yaml"


def _cfg(**overrides):
    return load_config(CONFIG, overrides=overrides)


def _quiet_cfg(**overrides):
    """v2 config with arrivals disabled for deterministic rule-A tests."""
    base = dict(
        lambda_low_per_s=0.0,
        lambda_high_per_s=0.0,
        hotspot_arrivals=False,
        horizon=6,
        uav_cover_radius_m=500.0,
    )
    base.update(overrides)
    return load_config(CONFIG, overrides=base)


def _manual_app(env: DisasterEnv, terminal: int, deadline_delta: float,
                template_id: int = 0) -> AppRequest:
    app = AppRequest(
        app_id=env.next_app_id,
        terminal=terminal,
        arrival_s=env.t_s,
        deadline_s=env.t_s + deadline_delta,
        template_id=template_id,
        dag=generate_dag(
            template_id, env.rng,
            cycles_scale=env.cfg.dag_cycles_scale,
            bits_scale=env.cfg.dag_bits_scale,
        ),
    )
    env.apps.append(app)
    env.terminal_apps[terminal].append(app.app_id)
    env.next_app_id += 1
    env.metrics["apps_arrived"] += 1
    return app


# ---------------------------------------------------------------------------
# config / main_v2
# ---------------------------------------------------------------------------
def test_main_v2_config_fields():
    cfg = _cfg(seed=0)
    assert cfg.num_terminals == 40
    assert cfg.num_uavs == 3
    assert cfg.horizon == 50
    assert cfg.uav_cover_radius_m == 800.0
    assert cfg.hotspot_arrivals is True
    assert cfg.hotspot_centers is not None
    assert cfg.trajectory == "patrol"


def test_v1_soft_default_kept():
    # Without main_v2 keys the dataclass defaults keep v1 semantics.
    cfg = load_config("disaster_va_dag_ho/configs/main.yaml", overrides={"seed": 0})
    assert cfg.uav_cover_radius_m is None
    assert cfg.hotspot_arrivals is False


# ---------------------------------------------------------------------------
# hard coverage / association (-1 must really happen)
# ---------------------------------------------------------------------------
def test_cover_mask_and_zeroed_rates():
    cfg = _cfg(seed=0, uav_cover_radius_m=500.0)
    term_pos = np.array([[1000.0, 1200.0], [100.0, 100.0], [1900.0, 1900.0]])
    uav_pos = np.array([[1000.0, 1000.0], [1500.0, 600.0], [400.0, 1700.0]])
    mask = terminal_uav_cover_mask(cfg, term_pos, uav_pos)
    # terminal 0 sits 200 m from UAV 0 and >500 m from the other two
    assert mask[0, 0]
    assert not mask[0, 1]
    assert not mask[0, 2]
    # corner terminal far from every UAV
    assert not mask[2].any()
    rates = np.ones((3, 3))
    masked = hard_cover_rates(cfg, rates, term_pos, uav_pos)
    assert masked[0, 0] == 1.0
    assert np.allclose(masked[0, 1:], 0.0)
    assert masked[2].sum() == 0.0


def test_hard_cover_unassigned_is_real():
    env = DisasterEnv(_cfg(seed=0, uav_cover_radius_m=500.0))
    env.uav_pos[:] = np.tile([1000.0, 1000.0], (env.cfg.num_uavs, 1))
    env.terminals.pos[:] = np.tile([1000.0, 1000.0], (env.cfg.num_terminals, 1))
    env.terminals.pos[1] = [150.0, 150.0]    # far from every UAV
    env.terminals.pos[2] = [1850.0, 1850.0]  # far from every UAV
    env._associate()
    assert env.assignment[0] >= 0
    assert env.assignment[1] == -1
    assert env.assignment[2] == -1
    assert env._rates_nk[1].sum() == 0.0


def test_associate_skips_zero_rate_rows_and_respects_caps():
    rates = np.array([
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],     # uncovered -> -1
        [4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 6.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [2.0, 2.0, 2.0],
    ])
    a = associate_terminals(rates, 3)  # cap = ceil(8/3) = 3
    assert a[1] == -1
    for u in range(3):
        assert int(np.count_nonzero(a == u)) <= 3
    # every row with a positive rate got some UAV
    assert np.all(a[np.flatnonzero(rates.max(axis=1) > 0)] >= 0)


def test_soft_cover_still_full_assignment():
    # R<=0 (v1 / w/o-hard-cover control) keeps full assignment.
    env = DisasterEnv(_quiet_cfg(uav_cover_radius_m=0.0, seed=0))
    env.uav_pos[:] = np.tile([1000.0, 1000.0], (env.cfg.num_uavs, 1))
    env.terminals.pos[:] = np.tile([1000.0, 1000.0], (env.cfg.num_terminals, 1))
    env.terminals.pos[1] = [20.0, 20.0]
    env.terminals.pos[2] = [1980.0, 1980.0]
    env._associate()
    assert np.all(env.assignment >= 0)


# ---------------------------------------------------------------------------
# rule A: uncovered pending apps cannot start / fail labelled uncovered
# ---------------------------------------------------------------------------
def test_uncovered_pending_app_never_registers_and_fails_uncovered():
    env = DisasterEnv(_quiet_cfg(seed=0))
    env.uav_pos[:] = np.tile([1900.0, 1900.0], (env.cfg.num_uavs, 1))
    env.terminals.pos[0] = [100.0, 100.0]
    app = _manual_app(env, terminal=0, deadline_delta=2.0)
    for _ in range(3):
        env.step(np.zeros((3, 2)))
    assert app.state == APP_FAILED
    assert env.metrics["apps_failed"] == 1
    assert env.metrics["fail_uncovered"] == 1
    assert env.scheduler.apps.get(app.app_id) is None  # never started


def test_leave_coverage_fails_uncovered_with_label():
    env = DisasterEnv(_quiet_cfg(seed=0, dag_cycles_scale=60.0))
    env.uav_pos[:] = np.tile([1000.0, 1000.0], (env.cfg.num_uavs, 1))
    env.terminals.pos[0] = [1000.0, 1100.0]  # inside R
    app = _manual_app(env, terminal=0, deadline_delta=1.6)
    env.step(np.zeros((3, 2)))  # covered: app gets registered
    assert app.state == APP_PENDING
    assert app.app_id in env.scheduler.apps
    # terminal walks out of every UAV's coverage before the deadline
    env.uav_pos[:] = np.tile([1900.0, 1900.0], (env.cfg.num_uavs, 1))
    env.terminals.pos[0] = [100.0, 100.0]
    env.step(np.zeros((3, 2)))
    assert app.state == APP_FAILED
    assert env.metrics["apps_failed"] == 1
    assert env.metrics["fail_uncovered"] == 1


def test_frac_unassociated_metric_range():
    env = DisasterEnv(_cfg(seed=0, uav_cover_radius_m=400.0))
    pol = make_trajectory("hover")
    done = False
    while not done:
        _, _, done, _ = env.step(pol.act(env))
    assert 0.0 < env.frac_unassociated < 1.0
    soft = DisasterEnv(_cfg(seed=0, uav_cover_radius_m=0.0))
    soft.step(np.zeros((3, 2)))
    assert soft.frac_unassociated == 0.0


# ---------------------------------------------------------------------------
# hotspot arrivals
# ---------------------------------------------------------------------------
def test_hotspot_spatial_rate_difference():
    cfg = _cfg(seed=0)
    ap = ArrivalProcess(cfg, np.random.default_rng(0))
    centers = ap._hotspot_centers()
    assert centers.shape == (2, 2)
    inside = np.array([centers[0]])
    outside = np.array([[60.0, 60.0]])  # far from both default hotspots
    rates = ap.rate_per_terminal(10.0, np.vstack([inside, outside]))
    assert rates[0] == pytest.approx(cfg.lambda_high_per_s)
    assert rates[1] == pytest.approx(cfg.lambda_low_per_s)
    assert rates[0] > 3.0 * rates[1]


def test_hotspot_sampling_produces_more_inside_hits():
    cfg = _cfg(seed=3)
    ap = ArrivalProcess(cfg, np.random.default_rng(7))
    centers = ap._hotspot_centers()
    pos = np.tile(np.array([centers[0]]), (20, 1))          # rows 0..19 inside
    pos = np.vstack([pos, np.tile(np.array([[60.0, 60.0]]), (20, 1))])  # 20 outside
    hits_in = hits_out = 0
    for _ in range(3000):
        hits = ap.sample_terminals(10.0, np.zeros(40), pos)
        hits_in += sum(1 for n in hits if n < 20)
        hits_out += sum(1 for n in hits if n >= 20)
    assert hits_in > 3.0 * hits_out  # λ ratio is 5; generous margin


def test_v1_uniform_rate_per_s_unchanged():
    cfg = _cfg(seed=0, hotspot_arrivals=False)
    ap = ArrivalProcess(cfg, np.random.default_rng(0))
    assert ap.rate_per_s(10.0) == cfg.lambda_low_per_s
    assert ap.rate_per_s(20.0) == cfg.lambda_high_per_s


# ---------------------------------------------------------------------------
# T-patrol / trajectories module
# ---------------------------------------------------------------------------
def test_tpatrol_stays_in_band_and_bounded():
    env = DisasterEnv(_cfg(seed=0))
    pol = TPatrol()
    vmax = env.cfg.uav_speed_max_mps * env.cfg.slot_seconds
    area = env.cfg.area_size_m
    k = env.cfg.num_uavs
    prev = env.uav_pos.copy()
    for _ in range(40):
        act = pol.act(env)
        assert act.shape == (k, 2)
        env.step(act)
        moved = np.linalg.norm(env.uav_pos - prev, axis=1)
        assert moved.max() <= vmax + 1e-6  # speed gate respected after move
        prev = env.uav_pos.copy()
        assert np.all(env.uav_pos >= 0.0) and np.all(env.uav_pos <= area)
        for u in range(k):
            assert env.uav_pos[u, 0] <= (u + 1) * area / k + 1e-6
            assert env.uav_pos[u, 0] >= u * area / k - 1e-6


def test_make_trajectory_dispatch():
    env = DisasterEnv(_cfg(seed=0))
    prev = env.uav_pos.copy()
    vmax = env.cfg.uav_speed_max_mps * env.cfg.slot_seconds
    for name in ("hover", "patrol", "sweep", "chase", "random"):
        pol = make_trajectory(name, seed=0)
        act = pol.act(env)
        assert act.shape == (3, 2)
        env.step(act)
        moved = np.linalg.norm(env.uav_pos - prev, axis=1)
        assert moved.max() <= vmax + 1e-6
        prev = env.uav_pos.copy()
