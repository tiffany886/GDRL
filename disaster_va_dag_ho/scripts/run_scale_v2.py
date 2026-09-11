"""v2 scale sweep (P3 gate, frozen main_v2.yaml): N x K.

Cells: N in {30,40,60} with K=3, plus K in {2,3} with N=40 (union = 4 cells).
Methods: Ours(T-patrol)/B3-hover/B2-chase/B6-noVWprune (deterministic per
scene seed) + B4-flat (RL, trained on train_seed=100+s, evaluated on scene
seed s, paired). 5 scene seeds; everything else frozen (R=800, hotspot
layout, lambda). Writes results/v2_scale_raw.csv + v2_scale_summary.csv.

Run: python -m disaster_va_dag_ho.scripts.run_scale_v2
"""
from __future__ import annotations

import csv
import time
from math import ceil
from pathlib import Path

from ..agents.mappo import Mappo
from ..agents.trainer import train, run_episode
from ..baselines.trajectories import make_trajectory
from .run_table import BASE_V2, RESULTS, _make_env, _stats

CELLS = [(30, 3), (40, 3), (60, 3), (40, 2)]
SEEDS = [0, 1, 2, 3, 4]
DET = [("patrol", "Ours(T-patrol)"), ("hover", "B3-hover"),
       ("chase", "B2-chase"), ("b6", "B6-noVWprune")]
KEYS = ("success_rate", "mean_delay_s", "energy_j",
        "leo_invalid_attempts", "leo_placements", "dep_violations",
        "apps_arrived", "apps_done", "apps_failed",
        "frac_unassociated", "fail_uncovered")


def _over(n: int, k: int) -> dict:
    return {"num_terminals": n, "num_uavs": k}


def _run_det(extra, seed, mover, label):
    delta = "b6_no_vw_prune.yaml" if mover == "b6" else None
    env = _make_env(delta, seed, extra=extra, base=BASE_V2)
    pol = make_trajectory("patrol" if mover == "b6" else mover, seed=seed)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(pol.act(env))
    return {"method": label}, {k: _stats(env)[k] for k in KEYS}


def _run_b4(extra, s, device):
    t0 = time.perf_counter()
    env = _make_env(None, 100 + s, extra=extra, base=BASE_V2)
    mover = make_trajectory("patrol", seed=100 + s)
    slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                   num_slots=slots, mode="flat_no_vel",
                   seed=100 + s, device=device)
    train(policy, env, episodes=250, seed=100 + s, eval_every=0,
          eval_seeds=(), mover=mover)
    env_e = _make_env(None, s, extra=extra, base=BASE_V2)
    _, st = run_episode(env_e, policy, seed=int(s), collect=False,
                        deterministic=True, mover=mover)
    meta = {"method": "B4-flat", "wall_s": round(time.perf_counter() - t0, 2)}
    return meta, {k: st[k] for k in KEYS}


def _write(rows):
    raw_path = RESULTS / "v2_scale_raw.csv"
    summary_path = RESULTS / "v2_scale_summary.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "N", "K", "method", "seed"]
                           + list(KEYS), extrasaction="ignore")
        w.writeheader()
        for (n, k), seed, method, stats in rows:
            w.writerow({"cell": f"N{n}-K{k}", "N": n, "K": k,
                        "method": method, "seed": seed, **stats})
    groups = {}
    for (n, k), seed, method, stats in rows:
        groups.setdefault((n, k, method), []).append(stats)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "N", "K", "method", "seeds"] + list(KEYS))
        for (n, k, method), ss in sorted(groups.items()):
            line = [f"N{n}-K{k}", n, k, method, len(ss)]
            for c in KEYS:
                mean = sum(x[c] for x in ss) / len(ss)
                sd = (sum((x[c] - mean) ** 2 for x in ss) / len(ss)) ** 0.5
                line.append(f"{mean:.4f}±{sd:.4f}")
            w.writerow(line)
    print(f"wrote {raw_path} ({len(rows)} rows) + {summary_path}")


def main():
    t_all = time.perf_counter()
    rows = []
    for n, k in CELLS:
        extra = _over(n, k)
        for mover, label in DET:
            for s in SEEDS:
                meta, stats = _run_det(extra, s, mover, label)
                rows.append(((n, k), s, meta["method"], stats))
        for s in SEEDS:
            meta, stats = _run_b4(extra, s, "cpu")
            rows.append(((n, k), s, meta["method"], stats))
        print(f"cell N{n}-K{k} done ({time.perf_counter()-t_all:.0f}s)")
    _write(rows)
    print(f"[run_scale_v2] total {time.perf_counter()-t_all:.0f}s")


if __name__ == "__main__":
    main()
