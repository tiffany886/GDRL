"""Appendix scale sweep (W5.1): N in {20,30,40}, K in {2,3}, burst x1 / x1.5.

Runs the embedded main method (MAPPO, short fixed budget) and the no-RL hover
reference for each cell and writes results/sweep_summary.csv (3 seeds each,
mean +/- std). Light appendix-level evidence: used to show trend consistency,
not the headline table.
"""
from __future__ import annotations

import argparse
import csv
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch

from ..agents.mappo import Mappo
from ..agents.trainer import train
from ..env.config import load_config_variant
from ..env.env import DisasterEnv

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _metrics(env: DisasterEnv) -> dict:
    m = env.metrics
    done_n = int(m["apps_done"])
    arrived = int(m["apps_arrived"])
    return {
        "success_rate": done_n / max(arrived, 1),
        "mean_delay_s": m["done_delay_sum_s"] / max(done_n, 1),
        "energy_j": float(m["energy_total_j"]),
        "leo_invalid_attempts": int(m["leo_invalid_attempts"]),
    }


def run_hover(cfg_over: dict, seed: int) -> dict:
    env = DisasterEnv(load_config_variant(None, overrides={**cfg_over, "seed": seed}))
    from ..baselines.no_rl import Hover
    env.reset(seed=seed)
    pol = Hover()
    done = False
    while not done:
        obs, r, done, info = env.step(pol.act(env))
    return _metrics(env)


def run_embed(cfg_over: dict, seed: int, episodes: int, device: str) -> dict:
    env = DisasterEnv(load_config_variant(None, overrides={**cfg_over,
                                                           "seed": seed}))
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                   num_slots=num_slots, mode="embed", seed=100 + seed,
                   device=device)
    train(policy, env, episodes=episodes, seed=100 + seed, eval_every=0,
          eval_seeds=())
    env.reset(seed=seed)
    from ..agents.trainer import run_episode
    _, st = run_episode(env, policy, seed=seed, collect=False,
                        deterministic=True)
    return _metrics(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cells = []
    for n in (20, 30, 40):
        for k in (2, 3):
            for burst in (1.0, 1.5):
                cells.append({
                    "num_terminals": n, "num_uavs": k,
                    "lambda_low_per_s": 0.02 * burst,
                    "lambda_high_per_s": 0.1 * burst,
                })
    rows = []
    t_start = time.perf_counter()
    for cell in cells:
        n, k, b = cell["num_terminals"], cell["num_uavs"], cell["lambda_high_per_s"] / 0.1
        for seed in seeds:
            t0 = time.perf_counter()
            h = run_hover(cell, seed)
            rows.append({"method": "hover", "N": n, "K": k, "burst": b,
                         "seed": seed, **h, "wall_s": round(time.perf_counter() - t0, 2)})
            t0 = time.perf_counter()
            e = run_embed(cell, seed, args.episodes, device)
            rows.append({"method": "embed", "N": n, "K": k, "burst": b,
                         "seed": seed, **e, "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"N={n} K={k} burst={b} done at {time.perf_counter()-t_start:.0f}s", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = ["method", "N", "K", "burst", "seed", "success_rate",
              "mean_delay_s", "energy_j", "leo_invalid_attempts", "wall_s"]
    with (RESULTS / "sweep_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # summary per cell: mean +/- std across seeds
    groups = {}
    for r in rows:
        key = (r["method"], r["N"], r["K"], r["burst"])
        groups.setdefault(key, []).append(r)
    val_cols = ["success_rate", "mean_delay_s", "energy_j",
                "leo_invalid_attempts"]
    with (RESULTS / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "N", "K", "burst", "seeds"] + val_cols)
        for key in sorted(groups):
            rs = groups[key]
            w.writerow(list(key) + [len(rs)] + [
                f"{np.mean([r[c] for r in rs]):.4f}±{np.std([r[c] for r in rs]):.4f}"
                for c in val_cols
            ])
    print(f"wrote {RESULTS/'sweep_raw.csv'} and sweep_summary.csv "
          f"in {time.perf_counter()-t_start:.0f}s")


if __name__ == "__main__":
    main()
