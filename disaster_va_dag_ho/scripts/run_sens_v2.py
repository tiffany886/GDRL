"""P1-lite sensitivity runner for the v2 main table (hard-cover + T-patrol).

Grid cells: R in {600,800,900} x lambda_high in {0.1,0.2}, everything else
locked to configs/main_v2.yaml (frozen; this file never modifies it).

Rows:
- deterministic trajectories (patrol=Ours / hover / chase / b6-noVWprune):
  one episode per (cell, scene_seed).
- RL flat (B4-flat): policy trained on train_seed=100+s (same cell overrides)
  with the fixed mover, then evaluated deterministically on scene_seed=s, so
  every method row is paired by scene seed for the Wilcoxon tests.

Outputs: results/v2_p1_sens_<tag>_raw.csv + _summary.csv (tag = scope).
The summary groups by (cell, method). Reward is kept in raw, dropped from
the mean/std summary exactly like scripts/run_table.py.write().
"""
from __future__ import annotations

import argparse
import csv
import time
from math import ceil
from pathlib import Path
from typing import Optional

import torch

from ..agents.mappo import Mappo
from ..agents.trainer import run_episode, train
from ..baselines.trajectories import make_trajectory
from ..env.config import load_config_variant
from ..env.env import DisasterEnv
from .run_table import BASE_V2, RESULTS, _make_env, _stats, run_norl_row

LAMBDA_LOW = 0.03  # locked alongside main_v2.yaml

DET_METHODS = {
    "patrol": "Ours(T-patrol)",
    "hover": "B3-hover",
    "chase": "B2-chase",
    "b6": "B6-noVWprune",
}

CELLS = [(600, 0.1), (800, 0.1), (900, 0.1),
         (600, 0.2), (800, 0.2), (900, 0.2)]
CENTER = (800, 0.2)

SCOPES = {
    # scope -> (cells, seeds, methods, episodes_flat, tag)
    "center-smoke": ([CENTER], [0], ["patrol", "hover", "b4"], 250, "smoke"),
    "grid-lite": (CELLS, [0, 1, 2, 3, 4],
                  ["patrol", "hover", "b4"], 250, "grid"),
    "center-10seed": ([CENTER], list(range(10)),
                      ["patrol", "hover", "chase", "b4"], 250, "c10"),
    "center-b4-500": ([CENTER], [0, 1, 2, 3, 4],
                      ["patrol", "hover", "b4"], 500, "b4x500"),
    # deterministic-only 10-seed rows at the center cell (B6 delay pairing,
    # chase narrative); fast, no RL training.
    "det-10": ([CENTER], list(range(10)),
               ["patrol", "hover", "chase", "b6"], 0, "det10"),
    # chase-mover B4-flat at the center cell (embed-vs-flat under chase):
    # pair with frozen B2-chase (= embed + chase) rows in main_table_v2.
    "chase-b4": ([CENTER], [0, 1, 2, 3, 4], ["b4"], 250, "chaseb4"),
}

STAT_KEYS = ("success_rate", "mean_delay_s", "energy_j",
             "leo_invalid_attempts", "leo_placements", "dep_violations",
             "apps_arrived", "apps_done", "apps_failed",
             "frac_unassociated", "fail_uncovered")


def _overrides(R: float, lambda_high: float) -> dict:
    return {"uav_cover_radius_m": float(R),
            "lambda_high_per_s": float(lambda_high),
            "lambda_low_per_s": LAMBDA_LOW}


def _label(method: str) -> str:
    return DET_METHODS.get(method, method)


def run_det(method: str, R: float, lh: float, seed: int,
            mover_name: Optional[str] = None,
            method_label: Optional[str] = None) -> tuple:
    delta = "b6_no_vw_prune.yaml" if method == "b6" else None
    mover = "patrol" if method == "b6" else (mover_name or method)
    meta, stats = run_norl_row(
        mover, seed, base=BASE_V2, delta=delta,
        extra=_overrides(R, lh),
        method_label=method_label or _label(method))
    meta.update(cell=f"R{R}-l{lh}", R=float(R), lambda_high=float(lh),
                scene_seed=seed)
    return meta, stats


def run_b4_pair(R: float, lh: float, train_seed: int, scene_seed: int,
                episodes: int, device: str,
                mover_name: str = "patrol",
                method_label: str = "B4-flat") -> tuple:
    """Train flat-no-vel MAPPO on train_seed, evaluate on scene_seed=s."""
    over = _overrides(R, lh)
    t0 = time.perf_counter()
    env = _make_env(None, train_seed, extra=over, base=BASE_V2)
    mover = make_trajectory(mover_name, seed=train_seed)
    num_slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                   num_slots=num_slots, mode="flat_no_vel",
                   seed=train_seed, device=device)
    train(policy, env, episodes=episodes, seed=train_seed, eval_every=0,
          eval_seeds=(), mover=mover)
    env_e = _make_env(None, scene_seed, extra=over, base=BASE_V2)
    _, st = run_episode(env_e, policy, seed=int(scene_seed), collect=False,
                        deterministic=True, mover=mover)
    meta = {"method": method_label, "wall_s": round(time.perf_counter() - t0, 2),
            "delta": "-", "episodes": episodes, "mover": mover_name,
            "cell": f"R{R}-l{lh}", "R": float(R), "lambda_high": float(lh),
            "scene_seed": scene_seed, "train_seed": train_seed}
    stats = {k: st[k] for k in STAT_KEYS}
    return meta, stats


def write(tag: str, rows):
    raw_path = RESULTS / f"v2_p1_sens_{tag}_raw.csv"
    summary_path = RESULTS / f"v2_p1_sens_{tag}_summary.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_fields = sorted({k for r, _ in rows for k in r}) + list(rows[0][1].keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        w.writeheader()
        for meta, stats in rows:
            w.writerow({**meta, **stats})
    groups = {}
    for meta, stats in rows:
        groups.setdefault((meta["cell"], meta["method"]), []).append(stats)
    val_cols = [k for k in rows[0][1].keys() if k != "reward"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "R", "lambda_high", "method", "seeds"] + val_cols)
        for (cell, method), stats_list in sorted(groups.items()):
            n = len(stats_list)
            r = float(cell.split("-")[0][1:])
            lh = float(cell.split("-")[1][1:])
            line = [cell, r, lh, method, n]
            for c in val_cols:
                mean = sum(x[c] for x in stats_list) / n
                sd = (sum((x[c] - mean) ** 2 for x in stats_list) / n) ** 0.5
                line.append(f"{mean:.4f}±{sd:.4f}")
            w.writerow(line)
    print(f"wrote {raw_path} ({len(rows)} raw rows) + {summary_path}")
    return raw_path, summary_path


def run_scope(scope: str, device: str, episodes_flat: int,
              seeds_override: Optional[str], mover_name: str) -> tuple:
    cells, seeds, methods, ep_default, tag = SCOPES[scope]
    ep = episodes_flat or ep_default
    if seeds_override is not None:
        seeds = [int(x) for x in seeds_override.split(",") if x.strip()]
    rows = []
    for (R, lh) in cells:
        for m in methods:
            for s in seeds:
                if m == "b4":
                    rows.append(run_b4_pair(R, lh, 100 + s, s, ep, device,
                                            mover_name=mover_name))
                else:
                    rows.append(run_det(m, R, lh, s))
    raw, summary = write(tag, rows)
    return raw, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="grid-lite",
                    choices=list(SCOPES))
    ap.add_argument("--seeds", default=None,
                    help="override seeds (comma list); default per scope")
    ap.add_argument("--episodes-flat", type=int, default=0,
                    help="override B4 training episodes")
    ap.add_argument("--mover-name", default="patrol",
                    help="B4 mover: patrol (default) or chase (narrative cell)")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    raw, summary = run_scope(a.scope, device, a.episodes_flat, a.seeds,
                             a.mover_name)
    print(f"[run_sens_v2 scope={a.scope}] raw={raw} summary={summary}")


if __name__ == "__main__":
    main()
