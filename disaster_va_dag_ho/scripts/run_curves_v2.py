"""B4/B5 training-convergence curves on the FROZEN v2 center config.

Both RL baselines use the same fixed T-patrol mover (offload-only). One train
seed (100) is reported; eval every 25 (B4) / 50 (B5) episodes on scene seeds
0-4. Outputs results/train_curves_v2/*.csv and
results/figures_v2/fig_v2_train_curves.png (success vs episodes, with the
deterministic Ours level and the B4@500ep budget-check point annotated).

Run: python -m disaster_va_dag_ho.scripts.run_curves_v2
"""
from __future__ import annotations

import csv
import re
from math import ceil
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..agents.mappo import Mappo
from ..agents.trainer import train
from ..baselines.sac_flat import SacFlat
from ..baselines.trajectories import make_trajectory
from ..env.env import DisasterEnv
from .run_table import BASE_V2, RESULTS, _make_env

OUT = RESULTS / "train_curves_v2"
FIG = RESULTS / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
EVAL_SEEDS = tuple(range(5))
TRAIN_SEED = 100


def _dump(name: str, rows):
    path = OUT / f"{name}_seed{TRAIN_SEED}.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _env():
    return _make_env(None, TRAIN_SEED, base=BASE_V2)


def run_b4():
    env = _env()
    mover = make_trajectory("patrol", seed=TRAIN_SEED)
    slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                   num_slots=slots, mode="flat_no_vel", seed=TRAIN_SEED,
                   device="cpu")
    rows = train(policy, env, episodes=250, seed=TRAIN_SEED, eval_every=25,
                 eval_seeds=EVAL_SEEDS, mover=mover)
    _dump("b4_flat", rows)


def run_b5():
    env = _env()
    mover = make_trajectory("patrol", seed=TRAIN_SEED)
    slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = SacFlat(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                     num_slots=slots, seed=TRAIN_SEED, device="cpu",
                     warmup=400, batch=256, alpha=0.2,
                     offload_only=True, mover=mover)
    rows = policy.fit(env, episodes=400, eval_every=50,
                      eval_seeds=EVAL_SEEDS)
    _dump("b5_sac", rows)


def _eval_means(name):
    path = OUT / f"{name}_seed{TRAIN_SEED}.csv"
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("kind") == "eval":
                rows.append((int(r["episode"]), float(r["success_rate"])))
    agg = {}
    for ep, v in rows:
        agg.setdefault(ep, []).append(v)
    return {ep: (float(np.mean(vs)), float(np.std(vs))) for ep, vs in agg.items()}


def _parse_cell(s):
    m = re.match(r"([-+0-9.eE]+)±([-+0-9.eE]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


def plot():
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, label, color, marker in [
            ("b4_flat", "B4-flat (MAPPO, offload-only)", "#2E86AB", "o"),
            ("b5_sac", "B5-SAC-flat (off-policy)", "#E9A319", "s")]:
        agg = _eval_means(name)
        eps = sorted(agg)
        ys = [agg[e][0] for e in eps]
        errs = [agg[e][1] for e in eps]
        ax.errorbar(eps, ys, yerr=errs, marker=marker, capsize=3,
                    color=color, label=label, lw=1.6)
    ours = 0.7311  # frozen main table Ours (deterministic patrol + embed)
    ax.axhline(ours, color="#C44E52", ls="--", lw=1.4,
               label=f"Ours (T-patrol + ScheduleDAG), no training = {ours:.3f}")
    # B4 budget check endpoint (center-b4-500 runs)
    try:
        with (RESULTS / "v2_p1_sens_b4x500_summary.csv").open() as f:
            for row in csv.DictReader(f):
                if row["method"] == "B4-flat":
                    v, _ = _parse_cell(row["success_rate"])
                    ax.plot([250, 500], [0.5972, v], marker="x", ms=8,
                            color="#2E86AB", ls=":", lw=1.2)
                    ax.text(500, v + 0.015, f"B4 @500ep {v:.3f}",
                            fontsize=8, color="#2E86AB")
    except FileNotFoundError:
        pass
    ax.set_xlabel("training episodes")
    ax.set_ylabel("eval success rate (mean over 5 scene seeds)")
    ax.set_ylim(0.3, 0.85)
    ax.set_title("Flat RL baselines plateau; doubling B4's budget does not "
                 "close the gap to the embedding")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_train_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_train_curves.png")


if __name__ == "__main__":
    run_b4()
    run_b5()
    plot()
