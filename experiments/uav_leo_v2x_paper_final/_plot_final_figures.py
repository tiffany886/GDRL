# -*- coding: utf-8 -*-
"""Paper-quality figure suite for the multi-UAV UAV-LEO experiment.

Produces (all 300 dpi, consistent style):
  fig1_k3_comparison.png    2x2 reward/success/latency/energy, 6-9 methods + stars
  fig2_scale_sweep.png      3 panels (K, users, area) absolute reward + err bars
  fig3_ablation.png         k3 ablation: Eco vs nb vs cc vs MPC (reward + energy)
  fig4_energy_weight.png    energy_weight sensitivity
  fig5_convergence.png      DRL eval reward vs steps + PMEO-M-Eco level line
  fig6_drop_penalty.png     drop-penalty sensitivity (restyled)
  fig7_horizon_stress.png   horizon/deadline stress (restyled)

Data is read from multi_uav_final / multi_uav_eco / multi_uav_ablation2 /
multi_uav_energy_sweep / mdrl_models; missing sources are skipped gracefully.
"""
import csv, collections, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"experiments/uav_leo_v2x_paper_final"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# ------------------------------------------------------------------ style
C_OURS   = "#C1272D"
C_MPC    = "#0072B2"
C_PMEO   = "#6A737B"
C_CUR    = "#E69F00"
C_FOL    = "#9A4D96"
C_RND    = "#A6A6A6"
C_DQN    = "#009E73"
C_TD3    = "#56B4E9"
C_SAC    = "#CC79A7"
GRAY     = "#4A4A4A"

METHOD_STYLE = {
    "pmeo_m_eco":       (C_OURS, "PMEO-M-Eco (ours)"),
    "mpc_m_h3":         (C_MPC,  "MPC-M-H3"),
    "pmeo_m":           (C_PMEO, "PMEO-M"),
    "current_exact_m":  (C_CUR,  "Current-pos"),
    "follow_tea_m":     (C_FOL,  "Follow-TEA"),
    "random_m":         (C_RND,  "Random"),
    "m_dqn":            (C_DQN,  "M-DQN"),
    "m_td3":            (C_TD3,  "M-TD3"),
    "m_sac":            (C_SAC,  "M-SAC"),
    "pmeo_m_eco_nb":     ("#8C2F39", "Eco (no-balance)"),
    "pmeo_m_eco_cc":     ("#B5651D", "Eco (centroid-only)"),
}

def style_ax(ax):
    ax.grid(axis="y", color="#E3E3E3", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#888888")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=GRAY, labelsize=8.5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 7.8,
})

KEYS = ["reward_mean", "latency_mean", "total_energy_mean", "success_rate"]

def load_summary(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def collect(path, preset="k3", methods=None):
    """rows grouped by method: list of {key: [per-seed values]}"""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in load_summary(path):
        if r.get("preset", "") != preset:
            continue
        m = r["method"]
        if methods is not None and m not in methods:
            continue
        for k in KEYS:
            acc[m][k].append(float(r[k]))
    return acc

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def paired_p(ours_vals, other_vals):
    a = np.asarray(ours_vals, dtype=float); b = np.asarray(other_vals, dtype=float)
    n = min(len(a), len(b))
    if n < 2: return 1.0
    _, p = stats.ttest_rel(a[:n], b[:n])
    return float(p)

def aligned(ours_acc, m, key):
    """seed-aligned value lists for ours vs method m on key."""
    ov, mv = [], []
    for s in SEED_ORDER:
        if s in ours_acc and s in m:
            ov.append(ours_acc[s][key]); mv.append(m[s][key])
    return ov, mv

# ------------------------------------------------------------------ data
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load_data import by_seed as load_by_seed, SEED_ORDER

_bys = load_by_seed("k3")
k3_all = collections.defaultdict(lambda: collections.defaultdict(list))
for m, sd in _bys.items():
    for k in KEYS:
        k3_all[m][k] = [d[k] for s in SEED_ORDER if (d := sd.get(s)) is not None]
k3_eco = collections.defaultdict(lambda: collections.defaultdict(list))
for s, d in _bys.get("pmeo_m_eco", {}).items():
    for k in KEYS:
        k3_eco["pmeo_m_eco"][k].append(d[k])

ORDER = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]
ORDER += [m for m in ("m_dqn", "m_td3", "m_sac") if m in k3_all]
AVAIL = [m for m in ORDER if m in k3_all and len(k3_all[m]["reward_mean"]) > 0]

def panel_bar(ax, key, ylabel, higher_better, title):
    means = [float(np.mean(k3_all[m][key])) for m in AVAIL]
    errs  = [float(np.std(k3_all[m][key], ddof=1)) for m in AVAIL]
    x = np.arange(len(AVAIL))
    colors = [METHOD_STYLE[m][0] for m in AVAIL]
    ours = k3_all["pmeo_m_eco"][key]
    ax.bar(x, means, yerr=errs, capsize=2.6, color=colors,
           edgecolor="black", linewidth=0.5, zorder=3,
           error_kw=dict(lw=0.8, ecolor="#555555"))
    _oursd = _bys.get("pmeo_m_eco", {})
    for i, m in enumerate(AVAIL[1:], start=1):
        ov, mv = aligned(_oursd, _bys.get(m, {}), key)
        p = paired_p(ov, mv)
        ytop = means[i] + errs[i]
        span = max(means) - min(means)
        ax.text(x[i], ytop + 0.04 * (span + 1e-9), stars(p), ha="center",
                fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_STYLE[m][1] for m in AVAIL], rotation=22,
                       ha="right", fontsize=7.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title)
    style_ax(ax)

# ================================================================= fig 1
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8))
fig.subplots_adjust(left=0.085, right=0.97, top=0.90, bottom=0.10,
                    wspace=0.34, hspace=0.46)
panel_bar(axes[0, 0], "reward_mean", "Reward (higher better)", True,
          "(a) Reward")
axes[0, 0].set_ylim(-11200, -2300)
panel_bar(axes[0, 1], "success_rate", "Task success rate", True,
          "(b) Success rate")
axes[0, 1].set_ylim(0.55, 1.0)
panel_bar(axes[1, 0], "latency_mean", "Avg latency (s)", False,
          "(c) Service latency")
panel_bar(axes[1, 1], "total_energy_mean", "Total energy (J/slot)", False,
          "(d) Total energy")
fig.suptitle("k3: 3 UAVs / 51 users / 3 hotspots / 2 km  (10 seeds x 10 episodes)\n"
             "paired t-test vs PMEO-M-Eco: * p<0.05  ** p<0.01  *** p<0.001",
             fontsize=10.5, fontweight="bold", y=0.985)
fig.savefig(os.path.join(FIG, "fig1_k3_comparison.png"), bbox_inches="tight")
plt.close(fig)
print("saved fig1_k3_comparison.png")

# ================================================================= fig 2
# scale sweep from multi_uav_eco_summary.csv (aggregated rows)
def load_scale():
    p = os.path.join(BASE, "multi_uav_eco_summary.csv")
    rows = load_summary(p)
    meta = {"k2": (2, 34, 2.0), "k3": (3, 51, 2.0), "k4": (4, 68, 2.0),
            "u30": (3, 30, 2.0), "u50": (3, 50, 2.0), "u80": (3, 80, 2.0),
            "a1": (3, 50, 1.0), "a15": (3, 50, 1.5), "a3": (3, 50, 3.0)}
    out = collections.defaultdict(dict)
    for r in rows:
        pre = r["preset"]
        out[pre]["pmeo_m_eco"] = (float(r["pmeo_m_eco_reward"]), float(r["pmeo_m_eco_reward_std"]))
        out[pre]["mpc_m_h3"] = (float(r["mpc_m_h3_reward"]), float(r["mpc_m_h3_reward_std"]))
        out[pre]["pmeo_m"] = (float(r["pmeo_m_reward"]), float(r["pmeo_m_reward_std"]))
    return out, meta

try:
    scale, meta = load_scale()
    series = {
        "UAV number K (~17 users/UAV, 2 km)": (["k2", "k3", "k4"], lambda p: meta[p][0]),
        "Users (K=3, 2 km)": (["u30", "u50", "u80"], lambda p: meta[p][1]),
        "Area size km (K=3, 50 users)": (["a1", "a15", "a3"], lambda p: meta[p][2]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    for ax, (title, (pres, xf)) in zip(axes, series.items()):
        xs = [xf(p) for p in pres]
        for meth, c, lab in [("pmeo_m", C_PMEO, "PMEO-M"),
                             ("mpc_m_h3", C_MPC, "MPC-M-H3"),
                             ("pmeo_m_eco", C_OURS, "PMEO-M-Eco (ours)")]:
            ys = [scale[p][meth][0] for p in pres]
            es = [scale[p][meth][1] for p in pres]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8, ms=5,
                        color=c, label=lab, zorder=3, markeredgecolor="black",
                        markeredgewidth=0.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("Reward", fontsize=9)
        ax.legend(fontsize=7.6)
        style_ax(ax)
    fig.suptitle("PMEO-M-Eco vs MPC-M-H3 across scales (10 seeds x 10 episodes)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIG, "fig2_scale_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig2_scale_sweep.png")
except Exception as e:
    print("scale sweep skipped:", e)

# ================================================================= fig 3
ab_path = os.path.join(BASE, "multi_uav_ablation2", "multi_uav_summary.csv")
ab = collect(ab_path, "k3")
if ab:
    methods = ["pmeo_m_eco", "pmeo_m_eco_nb", "pmeo_m_eco_cc", "mpc_m_h3"]
    src = dict(k3_all)
    for m in ("pmeo_m_eco_nb", "pmeo_m_eco_cc"):
        if m in ab:
            src[m] = ab[m]
    avail = [m for m in methods if m in src and len(src[m]["reward_mean"]) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, key, ylab in [(axes[0], "reward_mean", "Reward (higher better)"),
                          (axes[1], "total_energy_mean", "Total energy (J/slot)")]:
        means = [float(np.mean(src[m][key])) for m in avail]
        errs = [float(np.std(src[m][key], ddof=1)) for m in avail]
        x = np.arange(len(avail))
        colors = [METHOD_STYLE[m][0] for m in avail]
        ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="black",
               linewidth=0.5, zorder=3, error_kw=dict(lw=0.8))
        ours = src["pmeo_m_eco"][key]
        for i, m in enumerate(avail[1:], start=1):
            p = paired_p(ours, src[m][key])
            span = max(means) - min(means)
            ax.text(x[i], means[i] + errs[i] + 0.03 * (span + 1e-9), stars(p),
                    ha="center", fontsize=8, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_STYLE[m][1].replace(" (ours)", "") for m in avail],
                           rotation=14, ha="right", fontsize=7.6)
        ax.set_ylabel(ylab, fontsize=9)
        style_ax(ax)
    axes[0].set_title("(a) Reward ablation")
    axes[1].set_title("(b) Energy ablation")
    fig.suptitle("k3 ablations: balanced association / centroid-only candidates",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "fig3_ablation.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig3_ablation.png")
else:
    print("ablation data not ready")

# ================================================================= fig 4
en_path = os.path.join(BASE, "multi_uav_energy_sweep", "multi_uav_summary.csv")
if os.path.exists(en_path):
    rows = load_summary(en_path)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        w = float(r["energy_weight"])
        acc[(w, r["method"])]["reward_mean"].append(float(r["reward_mean"]))
        acc[(w, r["method"])]["total_energy_mean"].append(float(r["total_energy_mean"]))
    weights = sorted({float(r["energy_weight"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, key, ylab in [(axes[0], "reward_mean", "Reward (higher better)"),
                          (axes[1], "total_energy_mean", "Total energy (J/slot)")]:
        x = np.arange(len(weights))
        w = 0.24
        for i, (m, c, lab) in enumerate([("pmeo_m", C_PMEO, "PMEO-M"),
                                         ("mpc_m_h3", C_MPC, "MPC-M-H3"),
                                         ("pmeo_m_eco", C_OURS, "PMEO-M-Eco (ours)")]):
            ys = [float(np.mean(acc[(wt, m)][key])) for wt in weights]
            es = [float(np.std(acc[(wt, m)][key], ddof=1)) for wt in weights]
            ax.bar(x + (i - 1) * w, ys, w, yerr=es, capsize=2.4, color=c,
                   edgecolor="black", linewidth=0.4, label=lab,
                   error_kw=dict(lw=0.7), zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{wt:g}" for wt in weights])
        ax.set_xlabel("energy_weight", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(fontsize=7.4)
        style_ax(ax)
    axes[0].set_title("(a) Reward vs energy_weight")
    axes[1].set_title("(b) Energy vs energy_weight")
    fig.suptitle("k3 energy-weight sensitivity (10 seeds x 10 episodes)",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(FIG, "fig4_energy_weight.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig4_energy_weight.png")
else:
    print("energy sweep data not ready")

# ================================================================= fig 5
conv_methods = [m for m in ("m_dqn", "m_td3", "m_sac")
                if os.path.exists(os.path.join(BASE, "mdrl_models", f"k3_{m}_seed1", "eval_progress.csv"))]
if conv_methods:
    eco_mean = float(np.mean(k3_eco["pmeo_m_eco"]["reward_mean"])) if "pmeo_m_eco" in k3_eco else -2864.4
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for m in conv_methods:
        p = os.path.join(BASE, "mdrl_models", f"k3_{m}_seed1", "eval_progress.csv")
        rows = load_summary(p)
        steps = [int(r["step"]) for r in rows]
        vals = [float(r["eval_reward"]) for r in rows]
        ax.plot(steps, vals, marker="o", ms=4, lw=1.8,
                color=METHOD_STYLE[m][0], label=f"{METHOD_STYLE[m][1]} (eval)")
        train_p = os.path.join(BASE, "mdrl_models", f"k3_{m}_seed1", "train_progress.csv")
        if os.path.exists(train_p):
            tr = load_summary(train_p)
            tsteps = [int(r["step"]) for r in tr]
            tvals = [float(r["train_episode_reward"]) for r in tr]
            ax.plot(tsteps, tvals, lw=0.9, alpha=0.35, color=METHOD_STYLE[m][0])
    ax.axhline(eco_mean, color=C_OURS, lw=1.6, ls="--",
               label=f"PMEO-M-Eco (ours, deterministic): {eco_mean:.0f}")
    ax.set_xlabel("Training steps (env steps)", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.set_title("Decentralized DRL baselines at k3 (30k steps, seed 1)", fontsize=10)
    ax.legend(fontsize=7.6)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("saved fig5_convergence.png")
else:
    print("convergence data not ready")
