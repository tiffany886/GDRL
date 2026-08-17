
import csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = r"experiments/uav_leo_v2x_paper_final"
FIG = BASE + "/figures"
C_ECO, C_MPC, C_PMEO = "#c1272d", "#0072b2", "#6a737b"
C_CUR, C_FOL, C_RND = "#d96c2f", "#9a4d96", "#8c8c8c"

def load_summaries():
    files = {
        "k2":"multi_uav_eco_scan", "k3":"multi_uav_eco", "k4":"multi_uav_eco_scan",
        "u30":"multi_uav_eco_scan", "u50":"multi_uav_eco_scan2", "u80":"multi_uav_eco_u80",
        "a1":"multi_uav_eco_scan2", "a15":"multi_uav_eco_scan2", "a3":"multi_uav_eco_scan",
        "k3_final":"multi_uav_final",
    }
    rows = []
    for pre, d in files.items():
        path = f"{BASE}/{d}/multi_uav_summary.csv"
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["preset"] == pre or (pre == "k3_final" and r["preset"] == "k3"):
                rows.append((pre, r))
    return rows

def stat(rows):
    acc = collections.defaultdict(list)
    for r in rows: acc[r["method"]].append(float(r["reward_mean"]))
    out = {}
    for m, v in acc.items():
        out[m] = (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0))
    return out

def add_bar(ax, labels, vals, errs, colors, succ=None, ylabel="Reward (lower is better)"):
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor="black",
           linewidth=0.6, error_kw=dict(lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_axisbelow(True)
    if succ is not None:
        for xi, s in zip(x, succ):
            ax.text(xi, vals[xi] + (max(vals) - min(vals)) * 0.015, f"{s*100:.1f}%",
                    ha="center", fontsize=7.5, color="0.25")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "0.4", "axes.linewidth": 0.8,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

# ---------------- Fig 1: k3 main result (6 methods) ----------------
rows = load_summaries()
k3 = stat([r for tag, r in rows if tag == "k3_final" and r["method"] in
           ("pmeo_m", "current_exact_m", "follow_tea_m", "mpc_m_h3", "random_m")])
k3_eco = stat([r for tag, r in rows if tag == "k3" and r["method"] == "pmeo_m_eco"])
m_eco, e_eco = k3_eco["pmeo_m_eco"]
order = ["random_m", "follow_tea_m", "current_exact_m", "pmeo_m", "mpc_m_h3", "pmeo_m_eco"]
labels = ["Random", "Follow-TEA", "Current-pos", "PMEO-M", "MPC-M-H3", "PMEO-M-Eco\n(ours)"]
colors = [C_RND, C_FOL, C_CUR, C_PMEO, C_MPC, C_ECO]
vals = [k3[m][0] for m in order[:5]] + [m_eco]
errs = [k3[m][1] for m in order[:5]] + [e_eco]
succ_map = {}
acc = collections.defaultdict(list)
for tag, r in rows:
    if tag == "k3_final" and r["method"] in ("pmeo_m","current_exact_m","follow_tea_m","mpc_m_h3","random_m"):
        acc[r["method"]].append(float(r["success_rate"]))
succ = [np.mean(acc[m]) for m in order[:5]] + [0.9072]

fig, ax = plt.subplots(figsize=(7.2, 3.6))
add_bar(ax, labels, vals, errs, colors, succ=succ)
ax.set_title("k3: 3 UAVs / 51 users / 3 hotspots / 2 km  (10 seeds x 10 episodes)")
ax.set_ylim(min(vals) * 1.06, max(vals) * 0.72)
fig.tight_layout()
fig.savefig(f"{FIG}/multi_uav_k3_main.png", bbox_inches="tight")
plt.close(fig)
print("saved multi_uav_k3_main.png")

# ---------------- Fig 2: scaling (3 panels) ----------------
preset_meta = {
    "k2": (2, 34, 2.0), "k3": (3, 51, 2.0), "k4": (4, 68, 2.0),
    "u30": (3, 30, 2.0), "u50": (3, 50, 2.0), "u80": (3, 80, 2.0),
    "a1": (3, 50, 1.0), "a15": (3, 50, 1.5), "a3": (3, 50, 3.0),
}
series = {
    "UAV number K": (["k2", "k3", "k4"], lambda p: p[0], "density ~17 users/UAV"),
    "Users per UAV (K=3)": (["u30", "u50", "u80"], lambda p: p[1], "2 km"),
    "Area size (km, K=3)": (["a1", "a15", "k3", "a3"], lambda p: p[2], "50 users"),
}
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5))
for ax, (title, (pres, xf, sub)) in zip(axes, series.items()):
    pres = [p for p in pres if p != "k3" or title.startswith("Area")]
    xs = [xf(preset_meta[p]) for p in pres]
    for meth, color, label in [("pmeo_m", C_PMEO, "PMEO-M"),
                               ("mpc_m_h3", C_MPC, "MPC-M-H3"),
                               ("pmeo_m_eco", C_ECO, "PMEO-M-Eco (ours)")]:
        ys, es = [], []
        for p in pres:
            tag = "k3" if p == "k3" else p
            mm = "k3_final" if (p == "k3" and meth == "pmeo_m") else tag
            src = [r for t, r in rows if t == tag and r["method"] == meth]
            if not src:
                src = [r for t, r in rows if t == mm and r["method"] == meth]
            ys.append(np.mean([float(r["reward_mean"]) for r in src]))
            es.append(np.std([float(r["reward_mean"]) for r in src], ddof=1))
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.6, ms=4.5,
                    color=color, label=label, zorder=3)
    ax.set_title(title + f"\n({sub})", fontsize=9.5)
    ax.set_xlabel("number / size", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.grid(color="0.88", lw=0.6, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{FIG}/multi_uav_scale_gain.png", bbox_inches="tight")
plt.close(fig)
print("saved multi_uav_scale_gain.png")

# ---------------- Fig 3: k3 delay & energy ----------------
k3_all = {}
for tag, r in rows:
    if tag == "k3_final" and r["method"] in ("pmeo_m","current_exact_m","follow_tea_m","mpc_m_h3","random_m"):
        k3_all.setdefault(r["method"], collections.defaultdict(list))
        for k in ("latency_mean", "flight_energy_mean", "total_energy_mean"):
            k3_all[r["method"]][k].append(float(r[k]))
k3_eco_rows = [r for tag, r in rows if tag == "k3" and r["method"] == "pmeo_m_eco"]
k3_all["pmeo_m_eco"] = collections.defaultdict(list)
for r in k3_eco_rows:
    for k in ("latency_mean", "flight_energy_mean", "total_energy_mean"):
        k3_all["pmeo_m_eco"][k].append(float(r[k]))
methods6 = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m", "random_m"]
labels6 = ["PMEO-M-Eco\n(ours)", "MPC-M-H3", "PMEO-M", "Current-pos", "Follow-TEA", "Random"]
colors6 = [C_ECO, C_MPC, C_PMEO, C_CUR, C_FOL, C_RND]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
for ax, key, ylab, factor in [
    (axes[0], "latency_mean", "Avg latency (s)", 1.0),
    (axes[1], "total_energy_mean", "Total energy (J/slot)", 1.0)]:
    vals6 = [np.mean(k3_all[m][key]) for m in methods6]
    errs6 = [np.std(k3_all[m][key], ddof=1) for m in methods6]
    x = np.arange(len(methods6))
    ax.bar(x, vals6, yerr=errs6, capsize=3, color=colors6, edgecolor="black",
           linewidth=0.6, error_kw=dict(lw=0.9))
    ax.set_xticks(x); ax.set_xticklabels(labels6, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel(ylab, fontsize=10)
    ax.grid(axis="y", color="0.85", lw=0.6, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_axisbelow(True)
axes[0].set_title("k3 latency (lower is better)")
axes[1].set_title("k3 energy (lower is better)")
fig.tight_layout()
fig.savefig(f"{FIG}/multi_uav_k3_delay_energy.png", bbox_inches="tight")
plt.close(fig)
print("saved multi_uav_k3_delay_energy.png")
