
# -*- coding: utf-8 -*-
"""Paper-quality 2x2 comparison figure at k3 (10 seeds x 10 episodes)."""
import csv, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"experiments/uav_leo_v2x_paper_final"
FIG = BASE + "/figures"

COLORS = {
    "pmeo_m_eco": "#c1272d",      # ours: deep red
    "mpc_m_h3":   "#0072b2",      # blue
    "pmeo_m":     "#6a737b",      # grey
    "current_exact_m": "#e69f00", # amber
    "follow_tea_m": "#9a4d96",    # purple
    "random_m":   "#b0b0b0",      # light grey
}
ORDER = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]
LABELS = ["PMEO-M-Eco\n(ours)", "MPC-M-H3", "PMEO-M", "Current-pos", "Follow-TEA", "Random"]

SRC = {
    "pmeo_m_eco":      BASE + "/multi_uav_eco/multi_uav_summary.csv",
    "mpc_m_h3":        BASE + "/multi_uav_final/multi_uav_summary.csv",
    "pmeo_m":          BASE + "/multi_uav_final/multi_uav_summary.csv",
    "current_exact_m": BASE + "/multi_uav_final/multi_uav_summary.csv",
    "follow_tea_m":    BASE + "/multi_uav_final/multi_uav_summary.csv",
    "random_m":        BASE + "/multi_uav_final/multi_uav_summary.csv",
}
KEYS = ["reward_mean", "latency_mean", "total_energy_mean", "success_rate"]

def load():
    acc = {}
    for m, path in SRC.items():
        rows = []
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["preset"] == "k3" and r["method"] == m:
                rows.append({k: float(r[k]) for k in KEYS})
        acc[m] = rows
    return acc

acc = load()

def mean_std(m, key):
    v = [r[key] for r in acc[m]]
    return float(np.mean(v)), float(np.std(v, ddof=1))

def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "0.35",
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.fontsize": 8,
})
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.09, wspace=0.30, hspace=0.42)

def panel(ax, key, ylabel, higher_better):
    means, errs = [], []
    for m in ORDER:
        mu, sd = mean_std(m, key)
        means.append(mu); errs.append(sd)
    x = np.arange(len(ORDER))
    colors = [COLORS[m] for m in ORDER]
    bars = ax.bar(x, means, yerr=errs, capsize=2.5, color=colors,
                  edgecolor="black", linewidth=0.5,
                  error_kw=dict(lw=0.8), zorder=3)
    # significance vs ours
    ours = [r["reward_mean"] if key == "reward_mean" else r[key] for r in acc["pmeo_m_eco"]]
    for i, m in enumerate(ORDER[1:], start=1):
        other = [r["reward_mean"] if key == "reward_mean" else r[key] for r in acc[m]]
        if key in ("success_rate", "reward_mean"):
            _, p = stats.ttest_rel(ours, other)
        else:
            _, p = stats.ttest_rel(ours, other)
        p = float(p)
        ytop = means[i] + errs[i]
        span = max(means) - min(means)
        ax.text(x[i], ytop + 0.03 * (span + 1e-9), stars(p), ha="center", fontsize=8, color="0.2")
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=0, fontsize=7.6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax

ax = panel(axes[0, 0], "reward_mean", "Reward (higher is better)", True)
ax.set_ylim(-9800, -2300)
ax.set_title("(a) Reward  [$*p$<0.05, $**p$<0.01, $***p$<0.001]")
ax = panel(axes[0, 1], "success_rate", "Success rate", True)
ax.set_ylim(0.55, 1.0)
ax.set_title("(b) Task success rate")
ax = panel(axes[1, 0], "latency_mean", "Avg latency (s)", False)
ax.set_title("(c) Service latency")
ax = panel(axes[1, 1], "total_energy_mean", "Total energy (J/slot)", False)
ax.set_title("(d) Total energy")

fig.suptitle("k3: 3 UAVs / 51 users / 3 hotspots / 2 km  (10 seeds x 10 episodes)",
             fontsize=11, fontweight="bold", y=0.985)
out = f"{FIG}/multi_uav_k3_comparison.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)
