# -*- coding: utf-8 -*-
"""Restyle fig6 (drop penalty) and fig7 (horizon/deadline stress)."""
import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"experiments/uav_leo_v2x_paper_final"
FIG = os.path.join(BASE, "figures")
C_OURS, C_H5, C_MPC, C_PMEO = "#C1272D", "#D9534F", "#0072B2", "#6A737B"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False,
})

def style_ax(ax):
    ax.grid(axis="y", color="#E3E3E3", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
        ax.spines[s].set_linewidth(0.8)

def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

# ---------------- fig 6: drop penalty ----------------
rows = load(os.path.join(BASE, "multi_uav_dropsweep", "multi_uav_summary.csv"))
if rows:
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        dp = float(r["drop_penalty"])
        m = r["method"]
        for k in ("reward_mean", "success_rate", "total_energy_mean"):
            acc[(dp, m)][k].append(float(r[k]))
    dps = sorted({float(r["drop_penalty"]) for r in rows})
    methods = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]
    styles = {"pmeo_m_eco": (C_OURS, "PMEO-M-Eco (ours)"),
              "mpc_m_h3": (C_MPC, "MPC-M-H3"),
              "pmeo_m": (C_PMEO, "PMEO-M"),
              "current_exact_m": ("#E69F00", "Current-pos"),
              "follow_tea_m": ("#9A4D96", "Follow-TEA"),
              "random_m": ("#A6A6A6", "Random")}
    methods = [m for m in methods if (dps[0], m) in acc]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    for ax, key, ylab in [(axes[0], "reward_mean", "Reward (higher better)"),
                          (axes[1], "success_rate", "Task success rate")]:
        x = np.arange(len(dps))
        w = 0.13
        for i, m in enumerate(methods):
            ys = [float(np.mean(acc[(dp, m)][key])) for dp in dps]
            es = [float(np.std(acc[(dp, m)][key], ddof=1)) for dp in dps]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, ys, w, yerr=es,
                   capsize=2.0, color=styles[m][0], label=styles[m][1],
                   edgecolor="black", linewidth=0.35, error_kw=dict(lw=0.6),
                   zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([rf"$\lambda_{{drop}}$={dp:g}" for dp in dps], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=7.2, ncol=2, loc="lower right")
        style_ax(ax)
    axes[0].set_title("(a) Reward vs drop penalty")
    axes[1].set_title("(b) Success rate vs drop penalty")
    fig.suptitle("k3: PMEO-M-Eco advantage grows when failures are costly "
                 "(10 seeds x 10 episodes)", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "fig6_drop_penalty.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig6_drop_penalty.png")
else:
    print("drop sweep data missing")

# ---------------- fig 7: horizon / deadline stress ----------------
rows = load(os.path.join(BASE, "multi_uav_boost2", "multi_uav_summary.csv"))
if rows:
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        dl = float(r["success_deadline_s"])
        acc[(dl, r["method"])]["reward_mean"].append(float(r["reward_mean"]))
        acc[(dl, r["method"])]["success_rate"].append(float(r["success_rate"]))
    dls = [0.65, 0.5]
    methods = ["pmeo_m_eco", "pmeo_m_eco_f5", "mpc_m_h3", "pmeo_m"]
    styles = {"pmeo_m_eco": (C_OURS, "PMEO-M-Eco (H=3)"),
              "pmeo_m_eco_f5": (C_H5, "PMEO-M-Eco (H=5+vel)"),
              "mpc_m_h3": (C_MPC, "MPC-M-H3"),
              "pmeo_m": (C_PMEO, "PMEO-M")}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, key, ylab in [(axes[0], "reward_mean", "Reward (higher better)"),
                          (axes[1], "success_rate", "Task success rate")]:
        x = np.arange(len(dls))
        w = 0.18
        for i, m in enumerate(methods):
            ys = [float(np.mean(acc[(dl, m)][key])) for dl in dls]
            es = [float(np.std(acc[(dl, m)][key], ddof=1)) for dl in dls]
            ax.bar(x + (i - 1.5) * w, ys, w, yerr=es, capsize=2.2, color=styles[m][0],
                   label=styles[m][1], edgecolor="black", linewidth=0.4,
                   error_kw=dict(lw=0.7), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"deadline {dl:.2f}s" for dl in dls], fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=7.4)
        style_ax(ax)
    axes[0].set_title("(a) Reward under deadlines")
    axes[1].set_title("(b) Success rate under deadlines")
    fig.suptitle("k3: horizon enhancement (H=5 + velocity lookahead) and deadline stress",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "fig7_horizon_stress.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig7_horizon_stress.png")
else:
    print("boost2 data missing")
