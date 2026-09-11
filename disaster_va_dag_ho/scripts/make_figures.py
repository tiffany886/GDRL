"""Paper figures (W5.2). English labels (server has no CJK fonts).

Reads results/main_table_summary.csv, ablation_summary.csv and
results/train_curves/*.csv when present, writing PNGs into results/figures/:
  fig_main_metrics.png    B1-B6 four-metric comparison (bar + mean+/-std)
  fig_ablations.png       ablation comparison vs main row
  fig_train_curves.png    reward / eval success curves (本文 vs B4/B5)
  fig_leo_timeline.png    visibility timeline with/without VW pruning
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIG = RESULTS / "figures"

METRIC_TITLES = {
    "success_rate": "App success rate",
    "mean_delay_s": "Mean completion delay (s)",
    "energy_j": "Total UAV energy (J)",
    "leo_invalid_attempts": "Invalid LEO attempts",
}
METHOD_ORDER = ["hover", "random", "sweep", "chase", "VA-DAG-HO-MAPPO",
                "B4-flat", "B5-SACflat", "B6-noVWprune"]
SHORT = {m: m for m in METHOD_ORDER}


def _parse_cell(s: str) -> tuple[float, float]:
    m = re.match(r"([-+0-9.eE]+)±([-+0-9.eE]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


def _read_summary(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["method"]
            out[name] = {c: _parse_cell(row[c])
                         for c in row if c not in ("method", "seeds")}
    return out


def fig_main_metrics():
    data = _read_summary(RESULTS / "main_table_summary.csv")
    if not data:
        print("skip fig_main_metrics: no summary")
        return
    methods = [m for m in METHOD_ORDER if m in data]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_TITLES):
        vals = np.array([data[m][metric][0] for m in methods])
        errs = np.array([data[m][metric][1] for m in methods])
        ax.bar(range(len(methods)), vals, yerr=errs, capsize=4,
               color="#4C72B0", alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(METRIC_TITLES[metric])
    fig.suptitle("B1-B6 main comparison (5 seeds, mean +/- std)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_main_metrics.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_main_metrics.png")


def fig_ablations():
    data = _read_summary(RESULTS / "ablation_summary.csv")
    if not data:
        print("skip fig_ablations: no ablation summary")
        return
    base = _read_summary(RESULTS / "main_table_summary.csv")
    main = base.get("VA-DAG-HO-MAPPO", None)
    order = [k for k in data if k.startswith("ablate")]
    metrics = ["success_rate", "mean_delay_s", "energy_j",
               "leo_invalid_attempts"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, metrics):
        labels = ["VA-DAG-HO-MAPPO"] + order
        vals, errs = [], []
        if main:
            vals.append(main[metric][0]); errs.append(main[metric][1])
        else:
            labels = labels[1:]
        for k in order:
            vals.append(data[k][metric][0]); errs.append(data[k][metric][1])
        ax.bar(range(len(vals)), vals, yerr=errs, capsize=4, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(METRIC_TITLES[metric])
    fig.suptitle("Ablations (w/o VW-prune / DAG-order / embed=B4 / energy-in-state)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_ablations.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_ablations.png")


def _read_curve_csv(path: Path):
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tr = [r for r in rows if r.get("kind", "train") == "train"]
    ev = [r for r in rows if r.get("kind") == "eval"]
    return tr, ev


def fig_train_curves():
    curves = sorted((RESULTS / "train_curves").glob("*.csv")) if (
        RESULTS / "train_curves").exists() else []
    if not curves:
        print("skip fig_train_curves: no results/train_curves/*.csv yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for path in curves:
        tr, ev = _read_curve_csv(path)
        label = path.stem.replace("_seed", " seed").replace("_", " ")
        if not tr:
            continue
        ep = np.array([float(r["episode"]) for r in tr])
        # smoothed rolling mean
        rew = np.array([float(r["reward"]) for r in tr])
        win = max(5, len(rew) // 20)
        kern = np.ones(win) / win
        axes[0].plot(ep, np.convolve(rew, kern, mode="same"), label=label, lw=1.5)
        if ev:
            eep = np.array([float(r["episode"]) for r in ev])
            ess = np.array([float(r["success_rate"]) for r in ev])
            axes[1].plot(eep, ess, label=label, lw=1.5, marker="o", ms=3)
    axes[0].set_title("Training reward (rolling mean)")
    axes[0].set_xlabel("episode")
    axes[1].set_title("Eval success rate")
    axes[1].set_xlabel("episode")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_train_curves.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_train_curves.png")


if __name__ == "__main__":
    fig_main_metrics()
    fig_ablations()
    fig_train_curves()
