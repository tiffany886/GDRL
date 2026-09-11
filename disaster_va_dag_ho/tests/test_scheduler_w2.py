"""W2 hard unit tests (spec §7) + legal-schedule invariants."""
from __future__ import annotations

import numpy as np
import pytest

from disaster_va_dag_ho.env.config import DEFAULT_CONFIG_PATH, load_config
from disaster_va_dag_ho.env.env import DisasterEnv
from disaster_va_dag_ho.env.tasks import APP_DONE, APP_FAILED, APP_PENDING
from disaster_va_dag_ho.scheduler.dag import DAG, DAGNode, generate_dag
from disaster_va_dag_ho.scheduler.schedule_dag import ScheduleDAG, SchedCtx


def make_env(**overrides) -> DisasterEnv:
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides=overrides)
    return DisasterEnv(cfg)


def _chain_dag(cycles, bits=100.0):
    nodes = [
        DAGNode(0, "A", cycles[0], bits, bits, parents=()),
        DAGNode(1, "B", cycles[1], bits, bits, parents=(0,)),
    ]
    nodes[0].children.append(1)
    return DAG(template_id=0, name="manual-chain", nodes=nodes)


def _ctx_at(env: DisasterEnv, t_s: float):
    cfg = env.cfg
    return SchedCtx(
        t_s=t_s,
        t_end=t_s + cfg.slot_seconds,
        terminal_pos=env.terminals.pos,
        uav_pos=env.uav_pos,
        leo_sub=env.leo_sub,
        rates_nk=env.rate_matrix_terminal_uav(),
        rates_kL=None,  # set below
        ephemeris=env.ephemeris,
    )


def _scheduler(env: DisasterEnv) -> ScheduleDAG:
    sch = ScheduleDAG(env.cfg)
    sch.bind_ephemeris(env.ephemeris)
    return sch


def test_dag_templates_are_4_5_nodes_depth_3_4():
    rng = np.random.default_rng(0)
    for tid in range(3):
        dag = generate_dag(tid, rng, cycles_scale=1.0, bits_scale=1.0)
        assert 4 <= len(dag.nodes) <= 5
        order = dag.topological_order()
        assert order == sorted(order, key=lambda v: v)
        assert dag.sink == len(dag.nodes) - 1
        # source input positive; non-source input equals parent outputs sum
        assert dag.nodes[0].in_bits > 0.0
        for i in range(1, len(dag.nodes)):
            assert dag.nodes[i].in_bits == pytest.approx(
                sum(dag.nodes[p].out_bits for p in dag.nodes[i].parents)
            )
        assert all(n.cycles > 0.0 for n in dag.nodes)
        assert all(len(n.children) >= 1 for n in dag.nodes[:-1])


def test_predecessor_must_finish_before_child_starts():
    """前驱未完成不得开工: huge first node forces the gate."""
    env = make_env(seed=1)
    env.uav_pos[:] = env.terminals.pos[: env.cfg.num_uavs]  # high-rate spot
    env._associate()
    sch = _scheduler(env)
    dag = _chain_dag(cycles=[1.0e11, 1.0e7], bits=100.0)  # A ~12.5 s at 8e9 cps
    sch.register(0, dag, terminal=0, uav_k=int(env.assignment[0]),
                 arrival_s=0.0, deadline_s=40.0)
    ctx = _ctx_at(env, 0.0)
    ctx.rates_kL = np.full((env.cfg.num_uavs, env.cfg.num_leos), 1.0e7)
    sch.step(ctx)
    assert sch.apps[0].nodes[0].placed and not sch.apps[0].nodes[0].done
    assert not sch.apps[0].nodes[1].placed  # B must wait for A
    # advance until A finishes; B then starts only after A's finish
    t = 1.0
    while t < 60.0:
        c = _ctx_at(env, t)
        c.rates_kL = np.full((env.cfg.num_uavs, env.cfg.num_leos), 1.0e7)
        sch.step(c)
        if sch.apps[0].nodes[0].done:
            break
        t += 1.0
    assert sch.apps[0].nodes[0].done
    assert sch.apps[0].nodes[1].placed
    assert sch.apps[0].nodes[1].start >= sch.apps[0].nodes[0].finish - 1e-9
    assert sch.dep_violations == 0


def test_cross_window_leo_placement_is_rejected_by_prune():
    """跨窗 LEO 放置被拒绝: single short window [0,1], heavy node ~1.7 s LEO work."""
    env = make_env(seed=2, num_leos=1, num_uavs=1,
                   leo_window_period_s=[10.0], leo_window_duty=[0.1],
                   leo_window_phase_s=[0.0])
    env.uav_pos[:] = env.terminals.pos[:1]
    env._associate()
    sch = _scheduler(env)
    # LEO service duration: 1e11 cycles / 6e10 cps = 1.67 s > window 1 s
    dag = _chain_dag(cycles=[1.0e11, 1.0e7], bits=100.0)
    uav_k = int(env.assignment[0])
    sch.register(0, dag, terminal=0, uav_k=uav_k, arrival_s=0.0, deadline_s=30.0)
    ctx = _ctx_at(env, 0.0)
    ctx.rates_kL = np.full((1, 1), 1.0e7)
    stats = sch.step(ctx)
    # LEO candidate attempted but rejected; node runs on the pinned UAV
    assert stats.leo_attempts >= 1
    assert stats.leo_rejections >= 1
    ns = sch.apps[0].nodes[0]
    assert ns.placed and ns.device == "uav"


def test_out_of_window_or_deadline_gap_leo_is_rejected():
    """窗外 C=0/不可见: LEO window starts beyond the app deadline -> rejected."""
    env = make_env(seed=3, num_leos=1, num_uavs=1,
                   leo_window_period_s=[10.0], leo_window_duty=[0.1],
                   leo_window_phase_s=[3.0])  # first window only at [3,4]
    env.uav_pos[:] = env.terminals.pos[:1]
    env._associate()
    sch = _scheduler(env)
    dag = _chain_dag(cycles=[1.0e7, 1.0e7], bits=100.0)
    uav_k = int(env.assignment[0])
    # deadline at t=2: no LEO window before it -> every LEO attempt is invalid
    sch.register(0, dag, terminal=0, uav_k=uav_k, arrival_s=0.0, deadline_s=2.0)
    ctx = _ctx_at(env, 0.0)
    ctx.rates_kL = np.full((1, 1), 1.0e7)
    stats = sch.step(ctx)
    assert stats.leo_attempts >= 1
    assert stats.leo_rejections >= 1
    assert not any(p.device == "leo" for p in sch.log)


def test_legal_schedule_has_zero_dependency_violations_and_window_safe_leos():
    """随机合法 episode：依赖违反=0；每个 LEO placement 落在单一可见窗内且不超截止期."""
    rng = np.random.default_rng(0)
    for seed in (0, 1, 2):
        env = make_env(seed=seed)
        done = False
        while not done:
            act = rng.uniform(-10, 10, size=(env.cfg.num_uavs, 2))
            obs, r, done, info = env.step(act)
        sch = env.scheduler
        assert sch.dep_violations == 0
        windows = env.ephemeris.windows
        for p in sch.log:
            if p.device != "leo":
                continue
            app = next(a for a in env.apps if a.app_id == p.app_id)
            assert p.finish_s <= app.deadline_s + 1e-9
            cover = [
                (t_in, t_out) for t_in, t_out in windows[p.device_index]
                if t_in - 1e-9 <= p.start_s and p.finish_s <= t_out + 1e-9
            ]
            assert cover, f"LEO placement {p} not inside a single visible window"


def test_deadline_exceeded_apps_fail_and_are_not_done():
    env = make_env(seed=4)
    done = False
    while not done:
        obs, r, done, info = env.step(np.zeros((env.cfg.num_uavs, 2)))
    m = env.metrics
    assert m["apps_done"] + m["apps_failed"] + sum(
        1 for a in env.apps if a.state == APP_PENDING
    ) == m["apps_arrived"]
    # completed apps respect their deadlines exactly (scheduler invariant)
    for p in env.scheduler.log:
        app = env.apps[p.app_id]
        assert p.finish_s <= app.deadline_s + 1e-9
    for a in env.apps:
        if a.state == APP_DONE:
            assert a.completion_s <= a.deadline_s + 1e-9

