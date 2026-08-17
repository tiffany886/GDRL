
# -*- coding: utf-8 -*-
"""Drop-penalty sensitivity: eco advantage widens when failures are costly."""
import csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"experiments/uav_leo_v2x_paper_final"

# per-method per-dp reward means (10 seeds)
def load_dp6():
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{BASE}/multi_uav_eco/multi_uav_summary.csv", encoding="utf-8")):
        if r["preset"] == "k3": out[r["method"]].append(float(r["reward_mean"]))
    for r in csv.DictReader(open(f"{BASE}/multi_uav_final/multi_uav_summary.csv", encoding="utf-8")):
        if r["preset"] == "k3" and r["method"] != "pmeo_m_eco":
            out[r["method"]].append(float(r["reward_mean"]))
    return {m: float(np.mean(v)) for m, v in out.items()}

dp6 = load_dp6()
by_dp = {6.0: dp6}
acc = collections.defaultdict(list)
for r in csv.DictReader(open(f"{BASE}/multi_uav_dropsweep/multi_uav_summary.csv", encoding="utf-8")):
    acc[(float(r["drop_penalty"]), r["method"])].append(float(r["reward_mean"]))
for dp in (30.0, 60.0):
    by_dp[dp] = {m: float(np.mean(acc[(dp, m)])) for m in
                 ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]}

methods = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]
labels = {"pmeo_m_eco": "PMEO-M-Eco (ours)", "mpc_m_h3": "MPC-M-H3", "pmeo_m": "PMEO-M",
          "current_exact_m": "Current-pos", "follow_tea_m": "Follow-TEA", "random_m": "Random"}
colors = {"pmeo_m_eco": "#c1272d", "mpc_m_h3": "#0072b2", "pmeo_m": "#6a737b",
          "current_exact_m": "#e69f00", "follow_tea_m": "#9a4d96", "random_m": "#b0b0b0"}
styles = {"pmeo_m_eco": ("-", "o"), "mpc_m_h3": ("--", "s"), "pmeo_m": (":", "^"),
          "current_exact_m": ("-.", "D"), "follow_tea_m": (":", "v"), "random_m": ("--", "x")}

dps = [6.0, 30.0, 60.0]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 300,
                     "savefig.dpi": 300, "axes.linewidth": 0.7, "axes.edgecolor": "0.35"})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for m in methods:
    ys = [by_dp[dp][m] for dp in dps]
    ls, mk = styles[m]
    lw = 2.0 if m == "pmeo_m_eco" else 1.4
    ax.plot(dps, ys, ls, marker=mk, ms=4.5, lw=lw, color=colors[m], label=labels[m], zorder=3)
ax.set_xscale("log")
ax.set_xticks(dps); ax.set_xticklabels(["6", "30", "60"])
ax.set_xlabel("Drop penalty  $\\lambda_{drop}$ (log scale)")
ax.set_ylabel("Reward (higher is better)")
ax.grid(color="0.88", lw=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
ax.set_title("k3 sensitivity to task-failure cost", fontsize=10, fontweight="bold")
fig.tight_layout()
out = f"{BASE}/figures/multi_uav_drop_penalty.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)
