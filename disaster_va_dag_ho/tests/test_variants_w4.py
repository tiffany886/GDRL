"""W4 variant unit tests: w/o DAG-order ablation and B6 (no visible-window
pruning) must change the scheduler semantics in exactly the intended way
without breaking legal-schedule invariants."""
from __future__ import annotations

import numpy as np
import pytest

from disaster_va_dag_ho.env.config import DEFAULT_CONFIG_PATH, load_config
from disaster_va_dag_ho.env.env import DisasterEnv
from disaster_va_dag_ho.scheduler.dag import DAG, DAGNode
from disaster_va_dag_ho.scheduler.schedule_dag import ScheduleDAG, SchedCtx


def _chain_dag(cycles, bits=100.0):
    nodes = [
        DAGNode(0, "A", cycles[0], bits, bits, parents=()),
        DAGNode(1, "B", cycles[1], bits, bits, parents=(0,)),
    ]
    nodes[0].children.append(1)
    return DAG(template_id=0, name="manual-chain", nodes=nodes)


def _make_env(**overrides) -> DisasterEnv:
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides=overrides)
    return DisasterEnv(cfg)


def _scheduler(env: DisasterEnv) -> ScheduleDAG:
    sch = ScheduleDAG(env.cfg)
    sch.bind_ephemeris(env.ephemeris)
    return sch


def _ctx_at(env: DisasterEnv, t_s: float) -> SchedCtx:
    cfg = env.cfg
    return SchedCtx(
        t_s=t_s,
        t_end=t_s + cfg.slot_seconds,
        terminal_pos=env.terminals.pos,
        uav_pos=env.uav_pos,
        leo_sub=env.leo_sub,
        rates_nk=env.rate_matrix_terminal_uav(),
        rates_kL=np.full((env.cfg.num_uavs, env.cfg.num_leos), 1.0e7),
        ephemeris=env.ephemeris,
    )


def _park_on_terminal0(env: DisasterEnv):
    env.uav_pos[:] = env.terminals.pos[: env.cfg.num_uavs]
    env._associate()


def test_no_dag_order_lets_child_run_before_parent():
    """w/o DAG-order: each node is an independent package -> B may start
    while A (1e11 cycles ~12.5 s UAV work) is still running; control keeps
    the dependency gate."""
    env = _make_env(seed=5, dag_order=False)
    _park_on_terminal0(env)
    sch = _scheduler(env)
    dag = _chain_dag(cycles=[1.0e11, 1.0e7], bits=100.0)
    sch.register(0, dag, terminal=0, uav_k=int(env.assignment[0]),
                 arrival_s=0.0, deadline_s=40.0)
    stats = sch.step(_ctx_at(env, 0.0))
    assert stats.placements >= 1
    assert sch.apps[0].nodes[0].placed and not sch.apps[0].nodes[0].done
    assert sch.apps[0].nodes[1].placed  # independent: does not wait for A
    # LEO candidates were attempted for B without crashing on parent finish
    # the child is committed before its parent finished -> 1 dep violation
    assert sch.dep_violations == 1
    assert sch.apps[0].violated
    # the child is allowed to finish even though the parent is still open
    assert sch.apps[0].nodes[1].finish is not None

    # control: same scene with dag_order=True keeps the gate closed
    env2 = _make_env(seed=5, dag_order=True)
    _park_on_terminal0(env2)
    sch2 = _scheduler(env2)
    sch2.register(0, dag, terminal=0, uav_k=int(env2.assignment[0]),
                  arrival_s=0.0, deadline_s=40.0)
    sch2.step(_ctx_at(env2, 0.0))
    assert sch2.apps[0].nodes[0].placed and not sch2.apps[0].nodes[0].done
    assert not sch2.apps[0].nodes[1].placed  # B must wait for A here


def test_no_dag_order_counts_dependency_violations_over_episode():
    """随机合法 episode with dag_order=False: dependent nodes start before
    their parents finished and every such commit is counted as a dep
    violation (dependencies are not enforced -> inflated completions)."""
    rng = np.random.default_rng(0)
    env = _make_env(seed=6, dag_order=False)
    done = False
    while not done:
        act = rng.uniform(-10, 10, size=(env.cfg.num_uavs, 2))
        obs, r, done, info = env.step(act)
    assert env.scheduler.dep_violations > 0
    assert env.metrics["dep_violations"] == env.scheduler.dep_violations
    # every node that ever started also finished or was still running at the
    # horizon boundary (no NaN / None finishes on placed nodes)
    for app in env.scheduler.apps.values():
        for ns in app.nodes:
            if ns.placed:
                assert ns.finish is not None


def test_b6_no_vw_prune_cross_window_leo_is_doomed():
    """B6 (vw_prune=False) commits LEO work at always-on nominal times; work
    whose nominal relay+service interval crosses an outage is doomed and never
    completes (spec §1.1 跨窗未完成则失败). The pruned control rejects the same
    cross-window LEO candidate."""
    kw = dict(num_leos=1, num_uavs=1, leo_window_period_s=[10.0],
              leo_window_duty=[0.1], leo_window_phase_s=[0.0])
    env_off = _make_env(seed=7, vw_prune=False, **kw)
    _park_on_terminal0(env_off)
    sch_off = _scheduler(env_off)
    # 5e11 cycles: UAV needs 62.5 s (deadline-infeasible), LEO needs 8.33 s
    # which the 1 s window [0,1] cannot cover -> always-on commit is doomed.
    dag = _chain_dag(cycles=[5.0e11, 1.0e7], bits=100.0)
    sch_off.register(0, dag, terminal=0, uav_k=int(env_off.assignment[0]),
                     arrival_s=0.0, deadline_s=30.0)
    stats_off = sch_off.step(_ctx_at(env_off, 0.0))
    leo_ps = [p for p in sch_off.log if p.device == "leo"]
    assert stats_off.leo_placements >= 1
    assert leo_ps and leo_ps[0].finish_s > 30.0  # doomed: never completes
    ns = sch_off.apps[0].nodes[0]
    assert ns.placed and ns.device == "leo"
    # stepping well beyond the nominal finish never completes the doomed node
    for t in (2.0, 5.0, 12.0, 25.0):
        sch_off.step(_ctx_at(env_off, t))
    assert not ns.done

    # same scene with pruning on: the LEO candidate is rejected instead
    env_on = _make_env(seed=7, vw_prune=True, **kw)
    _park_on_terminal0(env_on)
    sch_on = _scheduler(env_on)
    sch_on.register(0, dag, terminal=0, uav_k=int(env_on.assignment[0]),
                    arrival_s=0.0, deadline_s=30.0)
    stats_on = sch_on.step(_ctx_at(env_on, 0.0))
    assert stats_on.leo_rejections >= 1
    assert not any(p.device == "leo" for p in sch_on.log)


def test_b6_no_vw_prune_within_window_commit_completes():
    """B6 commits still complete when the always-on nominal relay+service
    interval really is covered by one contiguous visible window."""
    env = _make_env(seed=8, vw_prune=False, num_leos=1, num_uavs=1,
                    leo_window_period_s=[10.0], leo_window_duty=[0.9],
                    leo_window_phase_s=[0.0])  # window [0,9], then [10,19]
    _park_on_terminal0(env)
    sch = _scheduler(env)
    dag = _chain_dag(cycles=[1.0e11, 1.0e7], bits=100.0)  # LEO ~1.67 s
    sch.register(0, dag, terminal=0, uav_k=int(env.assignment[0]),
                 arrival_s=0.0, deadline_s=30.0)
    sch.step(_ctx_at(env, 0.0))
    leo_ps = [p for p in sch.log if p.device == "leo"]
    assert leo_ps and leo_ps[0].finish_s <= 9.0  # inside the window
    sch.step(_ctx_at(env, 5.0))
    assert sch.apps[0].nodes[0].done
    assert sch.dep_violations == 0
