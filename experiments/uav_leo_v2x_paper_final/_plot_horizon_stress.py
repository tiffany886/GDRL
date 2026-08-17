
# -*- coding: utf-8 -*-
"""Horizon enhancement + deadline stress comparison (k3, 10 seeds x 10 episodes)."""
import csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"experiments/uav_leo_v2x_paper_final"
rows = list(csv.DictReader(open(f"{BASE}/multi_uav_boost2/multi_uav_summary.csv", encoding="utf-8")))

acc = collections.defaultdict(list)
for r in rows:
    acc[(float(r["success_deadline_s"]), r["method"])].append(r)

METHODS = ["pmeo_m_eco", "pmeo_m_eco_f5", "mpc_m_h3", "pmeo_m"]
LABELS = ["PMEO-M-Eco\n(H=3)", "PMEO-M-Eco\n(H=5+vel)", "MPC-M-H3", "PMEO-M"]
COLORS = {"pmeo_m_eco": "#c1272d", "pmeo_m_eco_f5": "#d9534f", "mpc_m_h3": "#0072b2", "pmeo_m": "#6a737b"}
DLS = [0.65, 0.5]
DL_LABELS = {"0.65": "Default deadline 0.65s", "0.5": "Tight deadline 0.5s"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 300,
                     "savefig.dpi": 300, "axes.linewidth": 0.7, "axes.edgecolor": "0.35"})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

for ax, key, ylab in [(axes[0], "reward_mean", "Reward (higher is better)"),
                      (axes[1], "success_rate", "Task success rate")]:
    x = np.arange(len(DLS))
    w = 0.19
    for i, m in enumerate(METHODS):
        means, errs = [], []
        for dl in DLS:
            v = [float(r[key]) for r in acc[(dl, m)]]
            means.append(float(np.mean(v)))
            errs.append(float(np.std(v, ddof=1)))
        off = (i - 1.5) * w
        ax.bar(x + off, means, w, yerr=errs, capsize=2.2, color=COLORS[m],
               label=LABELS[i], edgecolor="black", linewidth=0.4,
               error_kw=dict(lw=0.7), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([DL_LABELS[str(d)] for d in DLS], fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=9)
    ax.grid(axis="y", color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_title("(a) Reward under both deadlines")
axes[1].set_title("(b) Success rate under both deadlines")
axes[0].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
fig.suptitle("k3: horizon enhancement (H=5 + velocity lookahead) and deadline stress",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = f"{BASE}/figures/multi_uav_horizon_stress.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)
