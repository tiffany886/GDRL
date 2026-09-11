"""Publication figures for the VA-DAG-HO v2 paper (frozen main_v2.yaml data).

Reads the CSV artifacts produced by run_table / run_sens_v2 / p1_stats and
writes English-labelled PNGs into results/figures_v2/:
  fig_v2_main.png        main-table metrics, B1-B6/B4/B5 vs Ours
  fig_v2_ablation.png    4 ablations (+Ours reference)
  fig_v2_sensitivity.png success-rate gaps over the (R, lambda) grid
  fig_v2_budget.png      B4 250ep vs 500ep vs Ours at the center cell
  fig_v2_depviol.png     dependency-violation counts per ablation (W4 claim)
Run: python -m disaster_va_dag_ho.scripts.make_figures_v2
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
FIG = RESULTS / "figures_v2"
FIG.mkdir(parents=True, exist_ok=True)

ORDER = ["B1-random", "B2-chase", "B3-hover", "Ours(T-patrol)",
         "B6-noVWprune", "B4-flat", "B5-SACflat"]
ABL_ORDER = ["Ours(T-patrol)", "ablate-no-VWprune", "ablate-no-DAG-order",
             "ablate-no-hard-cover", "ablate-no-embed(=B4)"]
HILITE = {"Ours(T-patrol)"}
BAR_COLOR = "#4C72B0"
HILITE_COLOR = "#C44E52"


def _parse_cell(s: str):
    m = re.match(r"([-+0-9.eE]+)±([-+0-9.eE]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


def _read_summary(path: Path):
    out = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["method"]
            out[name] = {c: _parse_cell(row[c])
                         for c in row if c not in ("method", "seeds",
                                                   "cell", "R",
                                                   "lambda_high")}
    return out


def _bar_group(fig, axes, data, methods, metrics):
    for ax, metric in zip(axes, metrics):
        vals = np.array([data[m][metric][0] for m in methods])
        errs = np.array([data[m][metric][1] for m in methods])
        colors = [HILITE_COLOR if m in HILITE else BAR_COLOR for m in methods]
        ax.bar(range(len(methods)), vals, yerr=errs, capsize=4,
               color=colors, alpha=0.9)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] * 0.05 + np.ptp(vals) * 0.01,
                    f"{v:.3f}", ha="center", fontsize=7)


def fig_main():
    data = _read_summary(RESULTS / "main_table_v2_summary.csv")
    methods = [m for m in ORDER if m in data]
    if len(methods) < 4:
        print("skip main: no main_table_v2_summary rows")
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _bar_group(fig, axes.ravel(), data, methods,
               ["success_rate", "mean_delay_s", "energy_j",
                "leo_invalid_attempts"])
    fig.suptitle("VA-DAG-HO v2 main table (5 seeds, mean +/- std)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig_v2_main.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_main.png")


def fig_ablation():
    data = _read_summary(RESULTS / "ablation_v2_summary.csv")
    main = _read_summary(RESULTS / "main_table_v2_summary.csv")
    if not data:
        print("skip ablation: no ablation_v2_summary")
        return
    rows = {}
    if "Ours(T-patrol)" in main:
        rows["Ours(T-patrol)"] = main["Ours(T-patrol)"]
    rows.update(data)
    methods = [m for m in ABL_ORDER if m in rows]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    _bar_group(fig, axes.ravel(), rows, methods,
               ["success_rate", "mean_delay_s", "energy_j",
                "leo_invalid_attempts"])
    fig.suptitle("Ablations on frozen main_v2 (5 seeds): w/o VW-prune / "
                 "DAG-order / hard-cover / embed(=B4)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "fig_v2_ablation.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_ablation.png")


def fig_depviol():
    data = _read_summary(RESULTS / "ablation_v2_summary.csv")
    if not data:
        print("skip depviol: no ablation_v2_summary")
        return
    methods = [m for m in data if m == "ablate-no-DAG-order"] or list(data)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = []
    vals, errs = [], []
    for m in methods:
        labels.append(m)
        vals.append(data[m]["dep_violations"][0])
        errs.append(data[m]["dep_violations"][1])
    ax.bar(range(len(labels)), vals, yerr=errs, capsize=4,
           color=[HILITE_COLOR if "DAG" in l else BAR_COLOR for l in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("Dependency violations (child committed before its parents\n"
                 "finished) per episode - legal schedules stay at zero")
    ax.set_ylabel("dep_violations / episode")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_depviol.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_depviol.png")


def fig_sensitivity():
    # success-rate gap Ours-hover and Ours-B4 over the (R, lambda) grid
    path = RESULTS / "v2_p1_sens_grid_summary.csv"
    if not path.exists():
        print("skip sensitivity: no grid summary")
        return
    cells = {}  # (R, lh) -> {method: (succ, std)}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (float(row["R"]), float(row["lambda_high"]))
            cells.setdefault(key, {})[row["method"]] = _parse_cell(
                row["success_rate"])
    if not cells:
        return
    Rs = sorted({k[0] for k in cells})
    lambdas = sorted({k[1] for k in cells})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panels = [("Ours(T-patrol)", "B3-hover", "vs hover"),
              ("Ours(T-patrol)", "B4-flat", "vs B4-flat RL")]
    colors = {0.1: "#2E86AB", 0.2: "#C44E52"}
    for ax, (a, b, label) in zip(axes, panels):
        for lh in lambdas:
            gap, gerr = [], []
            for R in Rs:
                ca, cb = cells[(R, lh)].get(a), cells[(R, lh)].get(b)
                if ca and cb:
                    gap.append(ca[0] - cb[0])
                    gerr.append(np.hypot(ca[1], cb[1]))
            ax.errorbar(Rs, gap, yerr=gerr, marker="o", capsize=4,
                        color=colors[lh], label=f"$\\lambda_{{high}}$={lh}")
        ax.axhline(0.05, ls="--", lw=0.8, color="gray")
        ax.set_title(f"Ours {label}")
        ax.set_xlabel("UAV cover radius R (m)")
        ax.legend()
    axes[0].set_ylabel("success-rate gap")
    fig.suptitle("Sensitivity: conclusion region R in {600..900} m, "
                 "lambda_high in {0.1, 0.2} (5 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig_v2_sensitivity.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_sensitivity.png")


def fig_budget():
    c10 = _read_summary(RESULTS / "v2_p1_sens_c10_summary.csv")
    b4x = _read_summary(RESULTS / "v2_p1_sens_b4x500_summary.csv")
    if not c10:
        print("skip budget: no c10 summary")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels, vals, errs, cols = [], [], [], []
    for m in ["Ours(T-patrol)", "B3-hover", "B2-chase", "B4-flat"]:
        if m in c10:
            labels.append(m + " (10 seeds)")
            vals.append(c10[m]["success_rate"][0])
            errs.append(c10[m]["success_rate"][1])
            cols.append(HILITE_COLOR if "Ours" in m else BAR_COLOR)
    if b4x.get("B4-flat"):
        labels.append("B4-flat, 500 ep")
        vals.append(b4x["B4-flat"]["success_rate"][0])
        errs.append(b4x["B4-flat"]["success_rate"][1])
        cols.append(BAR_COLOR)
    ax.bar(range(len(labels)), vals, yerr=errs, capsize=4, color=cols)
    ax.axhline(vals[0], ls="--", lw=0.8, color=HILITE_COLOR)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.3, 0.95)
    ax.set_title("Center cell (R=800, lambda_high=0.2): more B4 training "
                 "does not close the gap")
    ax.set_ylabel("success rate")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_budget.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "fig_v2_budget.png")


if __name__ == "__main__":
    fig_main()
    fig_ablation()
    fig_depviol()
    fig_sensitivity()
    fig_budget()
