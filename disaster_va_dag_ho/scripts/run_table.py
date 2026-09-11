"""Main table + ablation runner.

v1 archive scopes (base = configs/main.yaml, MAPPO learns the trajectory):
  results/main_table_raw.csv / main_table_summary.csv  (B1-B6, >=5 seeds)
  results/ablation_raw.csv   / ablation_summary.csv    (4 ablations)

v2 scopes (base = configs/main_v2.yaml, no trajectory learning; the fixed
trajectory module moves the UAVs, B4/B5 learn offload only, Ours = T-patrol):
  results/main_table_v2_raw.csv / main_table_v2_summary.csv
  results/ablation_v2_raw.csv   / ablation_v2_summary.csv

Every row carries wall-clock seconds for the training methods. No-RL rows
simply run one episode per scene seed. RL rows train on seeds 100..100+S-1
and are evaluated deterministically on eval seeds 0..4.
"""
from __future__ import annotations

import argparse
import csv
import time
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..agents.mappo import Mappo
from ..agents.trainer import run_episode, train
from ..baselines.sac_flat import SacFlat
from ..baselines.trajectories import make_trajectory
from ..env.config import load_config_variant
from ..env.env import DisasterEnv

RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE_V2 = Path(__file__).resolve().parent.parent / "configs" / "main_v2.yaml"


def _stats(env: DisasterEnv) -> dict:
    m = env.metrics
    done_n = int(m["apps_done"])
    arrived = int(m["apps_arrived"])
    return {
        "success_rate": done_n / max(arrived, 1),
        "mean_delay_s": m["done_delay_sum_s"] / max(done_n, 1),
        "energy_j": float(m["energy_total_j"]),
        "leo_invalid_attempts": int(m["leo_invalid_attempts"]),
        "leo_placements": int(m["leo_placements"]),
        "dep_violations": int(m["dep_violations"]),
        "reward": float(m["reward_sum"]),
        "apps_arrived": arrived,
        "apps_done": done_n,
        "apps_failed": int(m["apps_failed"]),
        "frac_unassociated": float(env.frac_unassociated),
        "fail_uncovered": int(m["fail_uncovered"]),
    }


def _average(stats_list):
    out = {}
    keys = stats_list[0].keys()
    for k in keys:
        out[k] = float(np.mean([s[k] for s in stats_list]))
    return out


def _eval_mappo(env: DisasterEnv, policy, eval_seeds, mover=None) -> dict:
    evals = []
    for s in eval_seeds:
        _, st = run_episode(env, policy, seed=int(s), collect=False,
                            deterministic=True, mover=mover)
        evals.append({k: st[k] for k in (
            "success_rate", "mean_delay_s", "energy_j",
            "leo_invalid_attempts", "leo_placements", "dep_violations",
            "reward", "apps_arrived", "apps_done", "apps_failed",
            "frac_unassociated", "fail_uncovered",
        )})
    return _average(evals)


def _make_env(delta: Optional[str], seed: int, extra=None,
              base: Optional[str | Path] = None) -> DisasterEnv:
    overrides = {"seed": seed}
    if extra:
        overrides.update(extra)
    cfg = load_config_variant(delta, overrides=overrides, base_path=base)
    return DisasterEnv(cfg)


def run_norl_row(method: str, seed: int, base: Optional[str | Path] = None,
                 delta: Optional[str] = None,
                 method_label: Optional[str] = None,
                 extra: Optional[dict] = None) -> tuple:
    """One episode per (method, scene seed): mean +/- std over scene seeds."""
    env = _make_env(delta, seed, extra=extra, base=base)
    policy = make_trajectory(method, seed=seed)
    env.reset(seed=seed)
    done = False
    while not done:
        _, r, done, info = env.step(policy.act(env))
    return {"method": method_label or method, "wall_s": 0.0}, _stats(env)


def run_rl_row(mode: str, delta: Optional[str], train_seed: int, episodes: int,
               eval_seeds, device: str, extra=None,
               base: Optional[str | Path] = None,
               mover_name: Optional[str] = None,
               method_label: Optional[str] = None) -> tuple:
    t0 = time.perf_counter()
    env = _make_env(delta, train_seed, extra=extra, base=base)
    mover = (make_trajectory(mover_name, seed=train_seed)
             if mover_name else None)
    if mover is not None and mode == "embed":
        raise ValueError("v2: embed MAPPO rows must not use an external mover")
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                   num_slots=num_slots, mode=mode, seed=train_seed,
                   device=device)
    train(policy, env, episodes=episodes, seed=train_seed, eval_every=0,
          eval_seeds=(), mover=mover)
    meta = {"method": method_label or mode, "delta": delta or "-", "wall_s": round(
        time.perf_counter() - t0, 2)}
    return meta, _eval_mappo(env, policy, eval_seeds, mover=mover)


def run_b5_row(delta: Optional[str], train_seed: int, episodes: int,
               eval_seeds, device: str, extra=None,
               base: Optional[str | Path] = None,
               mover_name: Optional[str] = None,
               method_label: Optional[str] = None) -> tuple:
    t0 = time.perf_counter()
    env = _make_env(delta, train_seed, extra=extra, base=base)
    mover = (make_trajectory(mover_name, seed=train_seed)
             if mover_name else None)
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    # B5: flat off-policy MARL. MATD3-flat (deterministic) collapses to a
    # full-speed corner run (value overestimation); the stochastic SAC-flat
    # member of the same off-policy MADDPG/MATD3 family is used instead
    # (Gumbel reparameterised discrete offload head; see baselines/sac_flat.py).
    # v2: when a mover is given, offload_only=True makes the velocity head
    # inert and the fixed trajectory (T-patrol) moves the UAVs.
    policy = SacFlat(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                     num_slots=num_slots, seed=train_seed, device=device,
                     warmup=400, batch=256, alpha=0.2,
                     offload_only=mover is not None, mover=mover)
    policy.fit(env, episodes=episodes, eval_every=0, eval_seeds=())
    evals = [policy._eval_one(_make_env(delta, int(s), extra=extra, base=base),
                              int(s)) for s in eval_seeds]
    meta = {"method": method_label or "sacflat", "delta": delta or "-",
            "wall_s": round(time.perf_counter() - t0, 2)}
    return meta, _average(evals)


def write(out_stem: str, rows):
    raw_path = RESULTS / f"{out_stem}_raw.csv"
    summary_path = RESULTS / f"{out_stem}_summary.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_fields = sorted({k for r, _ in rows for k in r}) + list(
        rows[0][1].keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        w.writeheader()
        for meta, stats in rows:
            w.writerow({**meta, **stats})
    # summary: mean +/- std over seeds per method
    methods = {}
    for meta, stats in rows:
        methods.setdefault(meta["method"], []).append(stats)
    val_cols = [k for k in rows[0][1].keys() if k != "reward"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "seeds"] + [c for c in val_cols])
        for method, stats_list in methods.items():
            w.writerow([method, len(stats_list)] + [
                f"{np.mean([s[c] for s in stats_list]):.4f}"
                f"±{np.std([s[c] for s in stats_list]):.4f}" for c in val_cols
            ])
    print(f"wrote {raw_path} ({len(rows)} raw rows) + {summary_path}")


def _run_v1_main(seeds, eval_seeds, device, args) -> list:
    rows = []
    norl = ["hover", "random", "sweep", "chase"]
    rl = [
        ("embed", None, "VA-DAG-HO-MAPPO"),
        ("flat", "b4_flat.yaml", "B4-flat"),
        ("embed", "b6_no_vw_prune.yaml", "B6-noVWprune"),
    ]
    for method in norl:
        for seed in seeds:
            rows.append(run_norl_row(method, seed))
    for mode, delta, label in rl:
        for seed in seeds:
            ep = (args.episodes_flat if mode == "flat" else args.episodes)
            meta, stats = run_rl_row(
                mode, delta, train_seed=100 + seed, episodes=ep,
                eval_seeds=eval_seeds, device=device,
            )
            meta["method"] = label
            rows.append((meta, stats))
    for seed in seeds:
        meta, stats = run_b5_row(
            "b5_flat.yaml", train_seed=100 + seed,
            episodes=args.episodes_b5, eval_seeds=eval_seeds,
            device=device,
        )
        meta["method"] = "B5-SACflat"
        rows.append((meta, stats))
    return rows


def _run_v1_ablation(seeds, eval_seeds, device, args) -> list:
    ablations = [
        ("embed", "ablation_no_dag_order.yaml", "ablate-no-DAG-order"),
        ("embed", "ablation_no_energy_state.yaml", "ablate-no-energy-state"),
        ("embed", "b6_no_vw_prune.yaml", "ablate-no-VWprune"),
        ("flat", "b4_flat.yaml", "ablate-no-embed(=B4)"),
    ]
    rows = []
    for mode, delta, label in ablations:
        episodes = (args.episodes_flat if mode == "flat"
                    else args.episodes_ablation)
        for seed in seeds:
            meta, stats = run_rl_row(
                mode, delta, train_seed=100 + seed, episodes=episodes,
                eval_seeds=eval_seeds, device=device,
            )
            meta["method"] = label
            rows.append((meta, stats))
    return rows


def _run_v2_main(seeds, eval_seeds, device, args) -> list:
    """v2 main table (SPEC_V2_DS.md §3.1): all motion = fixed trajectories."""
    rows = []
    norl = [
        ("random", "B1-random"),
        ("chase", "B2-chase"),
        ("hover", "B3-hover"),
        ("patrol", "Ours(T-patrol)"),
    ]
    for method, label in norl:
        for seed in seeds:
            rows.append(run_norl_row(method, seed, base=BASE_V2,
                                     method_label=label))
    for seed in seeds:
        rows.append(run_norl_row("patrol", seed, base=BASE_V2,
                                 delta="b6_no_vw_prune.yaml",
                                 method_label="B6-noVWprune"))
    for seed in seeds:
        meta, stats = run_rl_row(
            "flat_no_vel", "b4_flat.yaml", train_seed=100 + seed,
            episodes=args.episodes_flat, eval_seeds=eval_seeds, device=device,
            base=BASE_V2, mover_name="patrol", method_label="B4-flat",
        )
        rows.append((meta, stats))
    for seed in seeds:
        meta, stats = run_b5_row(
            "b5_flat.yaml", train_seed=100 + seed,
            episodes=args.episodes_b5, eval_seeds=eval_seeds, device=device,
            base=BASE_V2, mover_name="patrol", method_label="B5-SACflat",
        )
        rows.append((meta, stats))
    return rows


def _run_v2_ablation(seeds, eval_seeds, device, args) -> list:
    ablations = [
        ("patrol", "b6_no_vw_prune.yaml", "ablate-no-VWprune"),
        ("patrol", "ablation_no_dag_order.yaml", "ablate-no-DAG-order"),
        ("patrol", "ablation_no_hard_cover.yaml", "ablate-no-hard-cover"),
    ]
    rows = []
    for method, delta, label in ablations:
        for seed in seeds:
            rows.append(run_norl_row(method, seed, base=BASE_V2, delta=delta,
                                     method_label=label))
    for seed in seeds:
        meta, stats = run_rl_row(
            "flat_no_vel", "b4_flat.yaml", train_seed=100 + seed,
            episodes=args.episodes_flat, eval_seeds=eval_seeds, device=device,
            base=BASE_V2, mover_name="patrol",
            method_label="ablate-no-embed(=B4)",
        )
        rows.append((meta, stats))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="main",
                    choices=["main", "ablation", "smoke",
                             "v2", "ablation-v2", "smoke-v2"])
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--episodes-flat", type=int, default=250,
                    help="episodes for B4 flat MAPPO rows")
    ap.add_argument("--episodes-b5", type=int, default=400,
                    help="episodes for the slower-converging B5 SAC-flat row")
    ap.add_argument("--episodes-ablation", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--eval-seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",") if s.strip())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.scope in ("v2", "smoke-v2"):
        write("main_table_v2",
              _run_v2_main(seeds, eval_seeds, device, args))
    if args.scope in ("ablation-v2", "smoke-v2"):
        write("ablation_v2",
              _run_v2_ablation(seeds, eval_seeds, device, args))

    if args.scope in ("main", "smoke"):
        write("main_table", _run_v1_main(seeds, eval_seeds, device, args))
    if args.scope in ("ablation", "smoke"):
        write("ablation", _run_v1_ablation(seeds, eval_seeds, device, args))


if __name__ == "__main__":
    main()
