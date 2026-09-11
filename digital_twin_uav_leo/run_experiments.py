"""RQ1-4 experiment driver for the digital-twin UAV-LEO line.

RQ1  sync-period vs performance tradeoff (tau scan)
RQ2  predictor / error-robustness scan
RQ3  decision-policy robustness to twin staleness
RQ4  adaptive (error-threshold) sync vs fixed-period sync

Usage examples:
  python -m digital_twin_uav_leo.run_experiments --rq rq1 --scenarios hard \
      --methods postmove_exact --taus 1,3 --episodes 1
  python -m digital_twin_uav_leo.run_experiments --rq rq1 --k3 --methods pmeo_m_eco,mpc_m_h3
  python -m digital_twin_uav_leo.run_experiments --rq rq4 --scenarios hard
"""
import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uav_leo_experiment.config import make_config
from uav_leo_experiment.env import UavLeoEnv
from uav_leo_experiment.baselines import ExpertExactOffloadPolicy
from uav_leo_experiment.mpc_traj import MPCTrajPolicy
from uav_leo_experiment.multi_uav import CurrentExactMPolicy, MpcMPolicy, PmeoMEcoPolicy
from uav_leo_experiment.run_multi_uav import build_config
from uav_leo_experiment.scenario_spec import MY_SCENARIOS

from .runner import run_dt_episode

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SINGLE_UAV_METHODS = {
    "postmove_exact": lambda: ExpertExactOffloadPolicy(offload_at="post_move"),
    "current_exact": lambda: ExpertExactOffloadPolicy(offload_at="current"),
    "mpc_traj_h3": lambda: MPCTrajPolicy(horizon=3, offload="exact"),
}
MULTI_UAV_METHODS = {
    "pmeo_m_eco": lambda: PmeoMEcoPolicy(),
    "current_exact_m": lambda: CurrentExactMPolicy(),
    "mpc_m_h3": lambda: MpcMPolicy(horizon=3),
}
K3_SEEDS = [1, 7, 42, 73, 2024]

BURST_OVERRIDES = {
    "burst_cycle": 15,
    "burst_on": 3,
    "burst_offset": 4,
    "burst_arrival_mult": 4.0,
    "burst_size_mult": 2.0,
}

SCENARIO_KEYS = {"hard": "v2x_hotspot_hard", "stress": "v2x_hotspot_stress"}


def make_single_cfg(scenario, seed, episodes, burst=False, burst_random=False):
    key = SCENARIO_KEYS.get(scenario, scenario)
    spec = MY_SCENARIOS[key]
    overrides = dict(users=spec["users"], leos=spec["leos"],
                     horizon=spec["horizon"])
    if burst:
        overrides.update(BURST_OVERRIDES)
        if burst_random:
            overrides["burst_random"] = True
    return make_config(key, ablation="none", seed=seed,
                       episodes=episodes, uavs=1, **overrides)


def make_k3_cfg(seed, episodes, burst=False, burst_random=False):
    cfg = build_config("k3", seed, episodes)
    if burst:
        cfg = replace(cfg, **BURST_OVERRIDES)
        if burst_random:
            cfg = replace(cfg, burst_random=True)
    return cfg


def csv_name(base, burst, burst_random=False):
    if burst_random:
        return f"{base}_burst_random.csv"
    return f"{base}_burst.csv" if burst else f"{base}.csv"


def scenario_label(scenario, burst, burst_random):
    if burst_random:
        return "burst_random"
    return "burst" if burst else scenario


def aggregate(env_cfg, policy, cfg, tau, predictor, eps, sync_mode, delta,
              episodes, seed0, probe_m=2, probe_n=3, task_mode="per_user"):
    agg = {"reward": 0.0, "latency": 0.0, "energy": 0.0,
           "completion": 0.0, "agree": 0.0, "syncs": 0.0}
    env = UavLeoEnv(env_cfg)
    for i in range(episodes):
        out = run_dt_episode(env, policy, env_cfg, tau=tau,
                             predictor=predictor, eps=eps, sync_mode=sync_mode,
                             seed=seed0 + i, delta=delta, probe_m=probe_m,
                             probe_n=probe_n, task_mode=task_mode)
        for k in agg:
            agg[k] += out[k]
    n = max(episodes, 1)
    return {k: v / n for k, v in agg.items()}


def write_rows(rows, name):
    path = RESULTS / name
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}  ({len(rows)} rows)")


def run_rq1(args):
    rows = []
    t0 = time.time()
    for method, make in (MULTI_UAV_METHODS if args.k3 else SINGLE_UAV_METHODS).items():
        if args.methods and method not in args.methods:
            continue
        policy = make()
        if args.k3:
            for tau in args.taus:
                for seed in K3_SEEDS:
                    cfg = make_k3_cfg(seed, args.episodes, args.burst,
                                      args.burst_random)
                    out = aggregate(cfg, policy, cfg, tau=tau,
                                    predictor="linear", eps=0.0, sync_mode="fixed",
                                    delta=10.0, episodes=args.episodes, seed0=seed)
                    out.update(scenario=("k3_burst_random" if args.burst_random
                                         else "k3_burst" if args.burst
                                         else "k3"),
                               method=method, tau=tau,
                               sync_mode="fixed", predictor="linear", eps=0.0)
                    rows.append(out)
                    print(f"[k3 {method} tau={tau}] {out['reward']:.1f}  "
                          f"({time.time()-t0:.0f}s)")
            continue
        for scenario in args.scenarios:
            for tau in args.taus:
                cfg = make_single_cfg(scenario, args.seed0, args.episodes,
                                      burst=args.burst,
                                      burst_random=args.burst_random)
                out = aggregate(cfg, policy, cfg, tau=tau, predictor="linear",
                                eps=0.0, sync_mode="fixed", delta=10.0,
                                episodes=args.episodes, seed0=args.seed0)
                out.update(scenario=scenario_label(scenario, args.burst,
                                                   args.burst_random),
                           method=method, tau=tau,
                           sync_mode="fixed", predictor="linear", eps=0.0)
                rows.append(out)
                print(f"[{scenario} {method} tau={tau}] {out['reward']:.1f}  "
                      f"({time.time()-t0:.0f}s)")
    write_rows(rows, csv_name("rq1_k3" if args.k3 else "rq1_single",
                              args.burst, args.burst_random))


def run_rq2(args):
    rows = []
    t0 = time.time()
    cfg = make_single_cfg(args.scenarios[0], args.seed0, args.episodes,
                          burst=args.burst, burst_random=args.burst_random)
    policy = SINGLE_UAV_METHODS["postmove_exact"]()
    configs = [("freeze", 0.0), ("linear", 0.0), ("resample", 0.0)]
    for eps in args.eps:
        configs.append(("linear_noise", eps))
    for predictor, eps in configs:
        out = aggregate(cfg, policy, cfg, tau=args.taus[0], predictor=predictor,
                        eps=eps, sync_mode="fixed", delta=10.0,
                        episodes=args.episodes, seed0=args.seed0)
        out.update(scenario=args.scenarios[0], method="postmove_exact",
                   tau=args.taus[0], sync_mode="fixed", predictor=predictor,
                   eps=eps)
        rows.append(out)
        print(f"[{predictor} eps={eps}] {out['reward']:.1f} ({time.time()-t0:.0f}s)")
    write_rows(rows, csv_name("rq2_single", args.burst, args.burst_random))


def run_rq3(args):
    rows = []
    t0 = time.time()
    for method, make in SINGLE_UAV_METHODS.items():
        if args.methods and method not in args.methods:
            continue
        policy = make()
        for tau in args.taus:
            cfg = make_single_cfg(args.scenarios[0], args.seed0, args.episodes)
            out = aggregate(cfg, policy, cfg, tau=tau, predictor="linear",
                            eps=0.0, sync_mode="fixed", delta=10.0,
                            episodes=args.episodes, seed0=args.seed0)
            out.update(scenario=args.scenarios[0], method=method, tau=tau,
                       sync_mode="fixed", predictor="linear", eps=0.0)
            rows.append(out)
            print(f"[{method} tau={tau}] {out['reward']:.1f} ({time.time()-t0:.0f}s)")
    write_rows(rows, "rq3_single.csv")


def run_rq4(args):
    rows = []
    t0 = time.time()
    if args.k3:
        for seed in K3_SEEDS:
            cfg = make_k3_cfg(seed, args.episodes, args.burst,
                              args.burst_random)
            policy = MULTI_UAV_METHODS["pmeo_m_eco"]()
            for tau in args.taus:
                out = aggregate(cfg, policy, cfg, tau=tau, predictor="linear",
                                eps=0.0, sync_mode="fixed", delta=10.0,
                                episodes=args.episodes, seed0=seed)
                out.update(scenario=("k3_burst_random" if args.burst_random
                                     else "k3_burst" if args.burst else "k3"),
                           method="pmeo_m_eco", tau=tau,
                           sync_mode="fixed", predictor="linear", eps=0.0, delta="")
                rows.append(out)
                print(f"[k3 fixed tau={tau}] {out['reward']:.1f} ({time.time()-t0:.0f}s)")
        for delta in args.deltas:
            for seed in K3_SEEDS:
                cfg = make_k3_cfg(seed, args.episodes, args.burst,
                                  args.burst_random)
                policy = MULTI_UAV_METHODS["pmeo_m_eco"]()
                out = aggregate(cfg, policy, cfg, tau=9999,
                                predictor="linear", eps=0.0,
                                sync_mode="adaptive", delta=delta,
                                episodes=args.episodes, seed0=seed,
                                probe_m=args.probe_m,
                                probe_n=args.probe_n,
                                task_mode=args.task_mode)
                out.update(scenario=("k3_burst_random"
                                     if args.burst_random else
                                     "k3_burst" if args.burst else "k3"),
                           method="pmeo_m_eco", tau=9999,
                           sync_mode="adaptive", predictor="linear",
                           eps=0.0, delta=delta,
                           task_mode=args.task_mode)
                rows.append(out)
                print(f"[k3 adaptive delta={delta}] {out['reward']:.1f} "
                      f"syncs={out['syncs']:.1f} "
                      f"({time.time()-t0:.0f}s)")
        write_rows(rows, csv_name("rq4_k3", args.burst, args.burst_random))
        return
    cfg = make_single_cfg(args.scenarios[0], args.seed0, args.episodes,
                          burst=args.burst, burst_random=args.burst_random)
    policy = SINGLE_UAV_METHODS["postmove_exact"]()
    for tau in args.taus:
        out = aggregate(cfg, policy, cfg, tau=tau, predictor="linear", eps=0.0,
                        sync_mode="fixed", delta=10.0, episodes=args.episodes,
                        seed0=args.seed0)
        out.update(scenario=scenario_label(args.scenarios[0], args.burst,
                                           args.burst_random),
                   method="postmove_exact", tau=tau,
                   sync_mode="fixed", predictor="linear", eps=0.0, delta="")
        rows.append(out)
        print(f"[fixed tau={tau}] {out['reward']:.1f} ({time.time()-t0:.0f}s)")
    for delta in args.deltas:
        out = aggregate(cfg, policy, cfg, tau=9999, predictor="linear", eps=0.0,
                        sync_mode="adaptive", delta=delta, episodes=args.episodes,
                        seed0=args.seed0, probe_m=args.probe_m,
                        probe_n=args.probe_n, task_mode=args.task_mode)
        out.update(scenario=scenario_label(args.scenarios[0], args.burst,
                                           args.burst_random),
                   method="postmove_exact", tau=9999,
                   sync_mode="adaptive", predictor="linear", eps=0.0,
                   delta=delta, task_mode=args.task_mode)
        rows.append(out)
        print(f"[adaptive delta={delta}] {out['reward']:.1f} "
              f"syncs={out['syncs']:.1f} ({time.time()-t0:.0f}s)")
    write_rows(rows, csv_name("rq4_single", args.burst, args.burst_random))


def parse_list(s):
    return [float(x.strip()) for x in s.split(",")] if s else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq", required=True, choices=["rq1", "rq2", "rq3", "rq4"])
    ap.add_argument("--scenarios", default="hard", type=str)
    ap.add_argument("--methods", default="", type=str)
    ap.add_argument("--taus", default="1,2,3,5,10", type=str)
    ap.add_argument("--eps", default="0.05,0.1,0.2", type=str)
    ap.add_argument("--deltas", default="5,10,20", type=str)
    ap.add_argument("--episodes", default=40, type=int)
    ap.add_argument("--seed0", default=73, type=int)
    ap.add_argument("--k3", action="store_true")
    ap.add_argument("--burst", action="store_true",
                    help="bursty traffic variant (periodic burst windows)")
    ap.add_argument("--burst-random", action="store_true",
                    help="bursty traffic with random burst offset per cycle "
                         "(twin does not know the phase)")
    ap.add_argument("--probe-m", default=2, type=int,
                    help="probe interval (slots) for adaptive sync")
    ap.add_argument("--probe-n", default=3, type=int,
                    help="users sampled per probe (per_user mode)")
    ap.add_argument("--task-mode", default="per_user",
                    choices=["per_user", "load"],
                    help="task-drift error mode for adaptive sync")
    args = ap.parse_args()
    args.scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.taus = [int(t) for t in parse_list(args.taus)]
    args.eps = parse_list(args.eps)
    args.deltas = parse_list(args.deltas)
    if args.rq == "rq1":
        run_rq1(args)
    elif args.rq == "rq2":
        run_rq2(args)
    elif args.rq == "rq3":
        run_rq3(args)
    elif args.rq == "rq4":
        run_rq4(args)


if __name__ == "__main__":
    main()
