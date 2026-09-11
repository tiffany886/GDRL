"""W3 tests: rule association caps, B4 flat obs/decision plumbing, and a
quick forward/update sanity of the MAPPO nets (tiny batch)."""
from __future__ import annotations

import numpy as np

from disaster_va_dag_ho.agents.association import associate_terminals
from disaster_va_dag_ho.agents.mappo import Mappo, StepData
from disaster_va_dag_ho.env.config import DEFAULT_CONFIG_PATH, load_config
from disaster_va_dag_ho.env.env import DisasterEnv
from disaster_va_dag_ho.env.tasks import APP_PENDING
from disaster_va_dag_ho.scheduler.schedule_dag import SchedCtx


def _env(flat: bool, **overrides) -> DisasterEnv:
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides={
        "obs_terminal_features": flat, **overrides,
    })
    return DisasterEnv(cfg)


def test_associate_caps_and_full_coverage():
    rng = np.random.default_rng(0)
    for n_term, k in [(30, 3), (20, 3), (40, 2), (7, 1)]:
        rates = rng.uniform(1.0e4, 1.0e7, size=(n_term, k))
        assign = associate_terminals(rates, k)
        assert np.all(assign >= 0)
        counts = np.bincount(assign, minlength=k)
        assert np.all(counts <= int(np.ceil(n_term / k)))
        assert counts.sum() == n_term


def test_flat_obs_extends_and_is_stable():
    env = _env(flat=False)
    base_dim = env.obs_dim
    env.reset(seed=1)
    obs = env.observation()
    assert obs.shape == (env.cfg.num_uavs, base_dim)
    env2 = _env(flat=True)
    assert env2.obs_dim > base_dim
    obs2 = env2.observation()
    assert obs2.shape == (env2.cfg.num_uavs, env2.obs_dim)
    assert np.all(np.isfinite(obs2))


def test_flat_decision_map_respected_by_scheduler():
    """B4 decision 0 (UAV) forces UAV placement; decision 2 waits."""
    env = _env(flat=False, num_leos=1, num_uavs=1, seed=3,
               leo_window_period_s=[10.0], leo_window_duty=[0.4],
               leo_window_phase_s=[0.0])
    env.uav_pos[:] = env.terminals.pos[:1]
    env._associate()
    # register one tiny app with a manually exposed ctx later
    from disaster_va_dag_ho.scheduler.dag import DAG, DAGNode
    from disaster_va_dag_ho.scheduler.schedule_dag import ScheduleDAG

    nodes = [
        DAGNode(0, "A", 1.0e7, 1.0e3, 1.0e3, parents=()),
        DAGNode(1, "B", 1.0e7, 1.0e3, 1.0e3, parents=(0,)),
    ]
    nodes[0].children.append(1)
    dag = DAG(template_id=0, name="tiny", nodes=nodes)
    sch = ScheduleDAG(env.cfg)
    sch.bind_ephemeris(env.ephemeris)
    uav_k = int(env.assignment[0])
    sch.register(0, dag, 0, uav_k, 0.0, 30.0)

    def ctx(decisions):
        return SchedCtx(
            t_s=0.0, t_end=1.0,
            terminal_pos=env.terminals.pos, uav_pos=env.uav_pos,
            leo_sub=env.leo_sub,
            rates_nk=env.rate_matrix_terminal_uav(),
            rates_kL=np.full((1, 1), 1.0e7),
            ephemeris=env.ephemeris,
            decisions=decisions,
        )

    stats = sch.step(ctx({0: 0}))  # force UAV
    assert sch.apps[0].nodes[0].device == "uav"
    assert stats.placements >= 1

    # new app whose decision is "wait" (2): nothing placed this slot
    sch.register(1, dag, 0, uav_k, 0.0, 30.0)
    stats2 = sch.step(ctx({1: 2}))
    assert not sch.apps[1].nodes[0].placed
    assert stats2.placements == 0


def test_mappo_tiny_update_roundtrip():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    num_slots = 2
    for mode in ("embed", "flat"):
        pol = Mappo(obs_dim=9, num_uavs=2, num_slots=num_slots, mode=mode,
                    seed=0, device="cpu", hidden=16)
        rng = np.random.default_rng(0)
        traj = []
        for _ in range(3):
            obs = rng.uniform(-1, 1, size=(2, 9))
            nobs = rng.uniform(-1, 1, size=(2, 9))
            out = pol.act(obs)
            dec = out["dec"]
            traj.append(StepData(
                obs=obs, next_obs=nobs, vel=out["vel"], dec=dec,
                rew=-1.0, done=False,
                logp_vel=out["logp_vel"], logp_cat=out["logp_cat"],
                value=out["value"], next_value=0.0,
            ))
        info = pol.update(traj)
        assert np.isfinite(info["loss"])
