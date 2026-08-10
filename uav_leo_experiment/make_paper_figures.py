# -*- coding: utf-8 -*-
"""Generate publication-quality figures for the UAV-LEO paper.

Reads the authoritative 40-episode sweep under
``experiments/uav_leo_v2x_paper_final/`` (sweep_summary.csv,
sweep_episodes.csv, sweep_steps.csv) and the DRL training logs under
``experiments/uav_leo_v2x/drl_models_v2/`` and produces a consistent set of
paper figures:

- training convergence (eval reward vs steps) for GDRL / PPO / TD3 / SAC
- mean reward, latency, energy (task + flight stacked), success rate bars
- per-episode reward box-plots and latency CDFs
- per-slot latency curves over the horizon
- UAV trajectories (episode 1) for the leading methods
- offloading target distribution (local / UAV / LEO stacked bars)
- latency-energy trade-off scatter

Usage:
    python -m uav_leo_experiment.make_paper_figures [--sweep_dir DIR]
                                                     [--output_dir DIR]
                                                     [--models_dir DIR]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#9e9e9e",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DIFFICULTIES = ["v2x_hotspot_hard", "v2x_hotspot_stress"]
DIFF_TITLES = {
    "v2x_hotspot_hard": "Hotspot Hard (12 users)",
    "v2x_hotspot_stress": "Hotspot Stress (16 users)",
}
DIFF_SHORT = {"v2x_hotspot_hard": "Hard", "v2x_hotspot_stress": "Stress"}

# Full comparison set used in bar charts (best first ordering is done later).
METHOD_ORDER = [
    "gdrl", "mpc_traj_h3", "predict_tea", "follow_tea", "deadline_tea", "ppo",
    "tea_partial", "energy_guarded_tea", "full_offload_tea", "greedy_partial",
    "lyapunov", "dqn", "sac", "td3", "random",
]

METHOD_LABELS = {
    "gdrl": "GDRL (ours)", "mpc_traj_h3": "MPC-H3",
    "predict_tea": "Predict-TEA", "follow_tea": "Follow-TEA",
    "deadline_tea": "Deadline-TEA", "ppo": "PPO (BC+KL)",
    "tea_partial": "TEA", "energy_guarded_tea": "E-Guard TEA",
    "full_offload_tea": "Full-offload", "greedy_partial": "Greedy",
    "lyapunov": "Lyapunov", "dqn": "DQN", "sac": "SAC", "td3": "TD3",
    "random": "Random", "gdrl_no_traj": "GDRL - traj",
    "gdrl_no_opt": "GDRL - exact offload", "postmove_exact": "Post-move exact",
    "current_exact": "Current-pos exact",
}

COLORS = {
    "gdrl": "#C62828", "mpc_traj_h3": "#1565C0", "predict_tea": "#2E7D32",
    "follow_tea": "#6A1B9A", "deadline_tea": "#EF6C00", "ppo": "#455A64",
    "tea_partial": "#00838F", "energy_guarded_tea": "#795548",
    "full_offload_tea": "#9E9D24", "greedy_partial": "#D81B60",
    "lyapunov": "#5D4037", "dqn": "#78909C", "sac": "#90A4AE",
    "td3": "#B0BEC5", "random": "#CFD8DC",
    "gdrl_no_traj": "#E57373", "gdrl_no_opt": "#EF9A9A",
}

# Methods shown on line/box/cdf charts (readability).
TOP_METHODS = ["gdrl", "mpc_traj_h3", "predict_tea", "follow_tea", "deadline_tea",
               "ppo", "tea_partial", "energy_guarded_tea"]
TRAJ_METHODS = ["gdrl", "mpc_traj_h3", "predict_tea", "follow_tea"]

CONV_METHODS = [("gdrl", "GDRL (ours)"), ("ppo", "PPO (BC+KL)"),
                ("td3", "TD3"), ("sac", "SAC")]
CONV_SUBDIR = {"gdrl": "gdrl/gdrl", "ppo": "ppo", "td3": "td3", "sac": "sac"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def fmt_label(method):
    return METHOD_LABELS.get(method, method)


def color_of(method):
    return COLORS.get(method, "#607D8B")


def _despine(ax, which="both"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}", flush=True)


# ---------------------------------------------------------------------------
# Figure 1: training convergence
# ---------------------------------------------------------------------------
def plot_convergence(models_dir, output_dir, difficulty):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    plotted = False
    for method, label in CONV_METHODS:
        prog = Path(models_dir) / CONV_SUBDIR[method] / difficulty / "eval_progress.csv"
        if not prog.exists():
            continue
        rows = read_csv(prog)
        steps = np.array([float(r["step"]) for r in rows])
        evals = np.array([float(r["eval_reward"]) for r in rows])
        ax.plot(steps / 1000.0, evals, marker="o", markersize=3.5, linewidth=1.7,
                label=label, color=color_of(method), zorder=3)
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("Training steps ($\\times 10^3$)")
    ax.set_ylabel("Evaluation reward (5 episodes)")
    ax.set_title(f"Training Convergence — {DIFF_TITLES[difficulty]}")
    ax.legend(loc="lower right", ncol=1)
    ax.grid(axis="y", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"convergence_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# Shared bar-chart helpers
# ---------------------------------------------------------------------------
def _bars_metric(summary_rows, difficulty, metric, best_largest=True):
    """Return (methods, values) sorted so the best method is on top.

    ``best_largest=True``  -> larger metric is better (sort descending).
    ``best_largest=False`` -> smaller metric is better (sort ascending).
    """
    rows = [r for r in summary_rows if r["difficulty"] == difficulty]
    items = []
    for r in rows:
        method = r["method"]
        if method not in METHOD_ORDER:
            continue
        items.append((method, float(r[metric])))
    items.sort(key=lambda kv: kv[1], reverse=best_largest)
    return [kv[0] for kv in items], [kv[1] for kv in items]


def plot_hbar_metric(summary_rows, difficulty, metric, ylabel, fname, output_dir,
                     best_largest=True, value_fmt="{:.1f}", xlim=None):
    methods, values = _bars_metric(summary_rows, difficulty, metric,
                                   best_largest=best_largest)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    y = np.arange(len(methods))
    colors = [color_of(m) for m in methods]
    bars = ax.barh(y, values, height=0.62, color=colors, edgecolor="white",
                   zorder=3)
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value, bar.get_y() + bar.get_height() / 2, value_fmt.format(value),
                va="center", ha="left" if value >= 0 else "right",
                fontsize=8.2, color="#333333", clip_on=False)
    ax.set_yticks(y)
    ax.set_yticklabels([fmt_label(m) for m in methods], fontsize=9.5)
    ax.set_xlabel(ylabel)
    ax.set_title(f"{ylabel} — {DIFF_TITLES[difficulty]}")
    ax.grid(axis="x", alpha=0.5)
    _despine(ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    save(fig, output_dir / f"{fname}_{DIFF_SHORT[difficulty].lower()}.png")


def plot_hbar_reward_std(episode_rows, summary_rows, difficulty, output_dir):
    """Mean episode cost (= -reward) with per-episode std error bars.

    Positive bars keep the chart standard for papers (lower cost = better);
    the best method is on top with the shortest bar.
    """
    summary = {r["method"]: r for r in summary_rows if r["difficulty"] == difficulty}
    methods, costs = _bars_metric(summary_rows, difficulty, "reward_mean",
                                  best_largest=True)
    costs = [-c for c in costs]
    stds = [float(summary[m]["reward_std"]) for m in methods]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    y = np.arange(len(methods))
    colors = [color_of(m) for m in methods]
    ax.barh(y, costs, xerr=stds, height=0.62, color=colors, edgecolor="white",
            zorder=3, error_kw=dict(ecolor="#444444", lw=0.9, capsize=2.5))
    for i, value in enumerate(costs):
        ax.text(value, y[i], f"{value:.1f}", va="center", ha="left",
                fontsize=8.2, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([fmt_label(m) for m in methods], fontsize=9.5)
    ax.set_xlabel(r"Mean episode cost (= $-\mathrm{reward}$) $\pm$ std")
    ax.set_title("Mean Episode Cost — {0}".format(DIFF_TITLES[difficulty]))
    ax.grid(axis="x", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"reward_{DIFF_SHORT[difficulty].lower()}.png")
def plot_stacked_energy(summary_rows, difficulty, output_dir):
    """Task energy + flight energy stacked horizontal bars."""
    methods = [m for m in METHOD_ORDER]
    rows = {r["method"]: r for r in summary_rows if r["difficulty"] == difficulty}
    methods = [m for m in methods if m in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(methods))
    task = np.array([float(rows[m]["task_energy_mean"]) for m in methods])
    flight = np.array([float(rows[m]["flight_energy_mean"]) for m in methods])
    colors = [color_of(m) for m in methods]
    ax.barh(y, flight, height=0.62, color=colors, edgecolor="white", zorder=3,
            label="Flight energy")
    ax.barh(y, task, left=flight, height=0.62, color="#BDBDBD",
            edgecolor="white", zorder=3, label="Task energy")
    for i, (t, fl) in enumerate(zip(task, flight)):
        ax.text(t + fl, y[i], f"{t + fl:.1f}", va="center", ha="left",
                fontsize=8.2, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([fmt_label(m) for m in methods], fontsize=9.5)
    ax.set_xlabel("Mean total energy (J)")
    ax.set_title(f"Energy Consumption — {DIFF_TITLES[difficulty]}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"energy_{DIFF_SHORT[difficulty].lower()}.png")


def plot_offload_stack(summary_rows, difficulty, output_dir):
    methods = [m for m in METHOD_ORDER]
    rows = {r["method"]: r for r in summary_rows if r["difficulty"] == difficulty}
    methods = [m for m in methods if m in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(methods))
    local = np.array([float(rows[m]["target_local_rate"]) for m in methods])
    uav = np.array([float(rows[m]["target_uav_rate"]) for m in methods])
    leo = np.array([float(rows[m]["target_leo_rate"]) for m in methods])
    colors = [color_of(m) for m in methods]
    ax.barh(y, local, height=0.62, color=colors, edgecolor="white", zorder=3,
            label="Local", alpha=0.95)
    ax.barh(y, uav, left=local, height=0.62, color="#1976D2",
            edgecolor="white", zorder=3, label="UAV")
    ax.barh(y, leo, left=local + uav, height=0.62, color="#FFB300",
            edgecolor="white", zorder=3, label="LEO")
    ax.set_yticks(y)
    ax.set_yticklabels([fmt_label(m) for m in methods], fontsize=9.5)
    ax.set_xlabel("Offloading target selection ratio")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"Offloading Decision Distribution — {DIFF_TITLES[difficulty]}")
    ax.legend(loc="lower right", ncol=3)
    ax.grid(axis="x", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"offload_stack_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# Box plot of per-episode reward
# ---------------------------------------------------------------------------
def plot_reward_boxplot(episode_rows, summary_rows, difficulty, output_dir):
    methods = [m for m in METHOD_ORDER]
    ep = {}
    for r in episode_rows:
        if r["difficulty"] == difficulty:
            ep.setdefault(r["method"], []).append(float(r["reward"]))
    methods = [m for m in methods if m in ep and len(ep[m]) >= 5]
    data = [ep[m] for m in methods]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2.4, alpha=0.35,
                                    markeredgecolor="none"),
                    medianprops=dict(color="#111111", lw=1.2),
                    whiskerprops=dict(color="#555555", lw=0.9),
                    capprops=dict(color="#555555", lw=0.9))
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(color_of(method))
        patch.set_alpha(0.85)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    ax.set_yticks(np.arange(1, len(methods) + 1))
    ax.set_yticklabels([fmt_label(m) for m in methods], fontsize=9.5)
    ax.set_xlabel("Per-episode reward (40 episodes)")
    ax.set_title(f"Reward Distribution — {DIFF_TITLES[difficulty]}")
    ax.grid(axis="x", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"reward_boxplot_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# CDF of per-episode mean latency
# ---------------------------------------------------------------------------
def plot_cdf_latency(episode_rows, difficulty, output_dir):
    ep = {}
    for r in episode_rows:
        if r["difficulty"] == difficulty:
            ep.setdefault(r["method"], []).append(float(r["latency_mean"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for method in TOP_METHODS:
        if method not in ep or len(ep[method]) < 5:
            continue
        vals = np.sort(np.array(ep[method]))
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, where="post", linewidth=1.8, color=color_of(method),
                label=fmt_label(method), zorder=3)
    ax.set_xlabel("Per-episode mean service latency (s/user)")
    ax.set_ylabel("CDF")
    ax.set_title(f"Latency CDF — {DIFF_TITLES[difficulty]}")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"cdf_latency_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# Per-slot latency curves
# ---------------------------------------------------------------------------
def plot_slot_latency(steps_rows, difficulty, output_dir):
    series = {}
    for r in steps_rows:
        if r["difficulty"] != difficulty or r["method"] not in TOP_METHODS:
            continue
        slot = int(r["step"]) - 1
        series.setdefault(r["method"], []).append(
            (slot, float(r["latency_mean"]), float(r["success_rate"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in TOP_METHODS:
        if method not in series:
            continue
        pts = sorted(series[method])
        slots = np.array([p[0] for p in pts])
        lat = np.array([p[1] for p in pts])
        # average over episodes by slot
        uniq = np.unique(slots)
        avg = np.array([lat[slots == u].mean() for u in uniq])
        ax.plot(uniq + 1, avg, marker="o", markersize=3.0, linewidth=1.6,
                color=color_of(method), label=fmt_label(method), zorder=3)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Mean service latency (s/user)")
    ax.set_title(f"Per-Slot Latency — {DIFF_TITLES[difficulty]}")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"slot_latency_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# UAV trajectory (episode 1)
# ---------------------------------------------------------------------------
def plot_trajectory(steps_rows, difficulty, output_dir):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for method in TRAJ_METHODS:
        pts = [r for r in steps_rows
               if r["difficulty"] == difficulty and r["method"] == method
               and int(r["episode"]) == 1]
        if not pts:
            continue
        pts = sorted(pts, key=lambda r: int(r["step"]))
        xs = np.array([float(r["uav_pos_x"]) for r in pts])
        ys = np.array([float(r["uav_pos_y"]) for r in pts])
        ax.plot(xs, ys, marker="o", markersize=2.2, linewidth=1.5,
                color=color_of(method), label=fmt_label(method), zorder=3)
        ax.plot(xs[0], ys[0], marker="s", markersize=6, color=color_of(method),
                zorder=4)
        ax.annotate("start", (xs[0], ys[0]), fontsize=8, color=color_of(method))
    ax.set_xlabel("UAV x position (m)")
    ax.set_ylabel("UAV y position (m)")
    ax.set_title(f"UAV Trajectory (episode 1) — {DIFF_TITLES[difficulty]}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(alpha=0.5)
    _despine(ax)
    save(fig, output_dir / f"trajectory_{DIFF_SHORT[difficulty].lower()}.png")


# ---------------------------------------------------------------------------
# Latency-energy trade-off scatter
# ---------------------------------------------------------------------------
def plot_tradeoff(summary_rows, output_dir):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    markers = {"v2x_hotspot_hard": "o", "v2x_hotspot_stress": "s"}
    for difficulty in DIFFICULTIES:
        rows = {r["method"]: r for r in summary_rows
                if r["difficulty"] == difficulty}
        for method in METHOD_ORDER:
            if method not in rows:
                continue
            lat = float(rows[method]["latency_mean"])
            energy = float(rows[method]["total_energy_mean"])
            ax.scatter(lat, energy, s=85, color=color_of(method),
                       marker=markers[difficulty], alpha=0.9,
                       edgecolor="white", linewidth=0.6, zorder=3,
                       label=f"{fmt_label(method)}" if difficulty == DIFFICULTIES[0] else None)
    # highlight GDRL
    for difficulty in DIFFICULTIES:
        rows = {r["method"]: r for r in summary_rows
                if r["difficulty"] == difficulty}
        if "gdrl" in rows:
            ax.scatter(float(rows["gdrl"]["latency_mean"]),
                       float(rows["gdrl"]["total_energy_mean"]), s=190,
                       facecolors="none", edgecolors=color_of("gdrl"),
                       linewidths=2.0, marker=markers[difficulty], zorder=4)
    ax.set_xlabel("Mean service latency (s/user)")
    ax.set_ylabel("Mean total energy (J)")
    ax.set_title("Latency–Energy Trade-off")
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate labels
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(*zip(*uniq) if uniq else ([], []), loc="center left",
              bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=8.5)
    ax.grid(alpha=0.5)
    _despine(ax)
    save(fig, output_dir / "tradeoff.png")


# ---------------------------------------------------------------------------
def plot_ablation(ablation_root, output_dir):
    """Grouped cost bars for the GDRL contribution ablation."""
    variants = [("gdrl", "Full GDRL"), ("gdrl_no_traj", "- trajectory residual"),
                ("gdrl_no_opt", "- exact offload")]
    cfg_dir = {"v2x_hotspot_hard": "U12_L4_40ep_T30",
               "v2x_hotspot_stress": "U16_L4_40ep_T30"}
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(DIFFICULTIES))
    width = 0.26
    for idx, (method, label) in enumerate(variants):
        costs, stds = [], []
        for difficulty in DIFFICULTIES:
            base = Path(ablation_root) / difficulty / "none" / cfg_dir[difficulty]
            rows = read_csv(base / "uav_leo_summary.csv")
            row = next(r for r in rows if r["method"] == method)
            costs.append(-float(row["reward_mean"]))
            stds.append(float(row["reward_std"]))
        bars = ax.bar(x + (idx - 1) * width, costs, width, yerr=stds,
                      label=label, color=[color_of(method)] * 2,
                      edgecolor="white", zorder=3,
                      error_kw=dict(ecolor="#444444", lw=0.9, capsize=2.5))
        for bar, value in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2,
                    f"{value:.1f}", ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels([DIFF_TITLES[d] for d in DIFFICULTIES], fontsize=9.5)
    ax.set_ylabel("Mean episode cost (= -reward)")
    ax.set_title("GDRL Contribution Ablation")
    ax.legend(loc="upper right", fontsize=8.8)
    ax.grid(axis="y", alpha=0.5)
    _despine(ax)
    save(fig, output_dir / "ablation_gdrl.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default="experiments/uav_leo_v2x_paper_final")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--models_dir", default="experiments/uav_leo_v2x/drl_models_v2")
    args = parser.parse_args()

    root = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)

    summary = read_csv(root / "sweep_summary.csv")
    episodes = read_csv(root / "sweep_episodes.csv")
    steps = read_csv(root / "sweep_steps.csv")

    for difficulty in DIFFICULTIES:
        short = DIFF_SHORT[difficulty].lower()
        plot_convergence(models_dir, output_dir, difficulty)
        plot_hbar_reward_std(episodes, summary, difficulty, output_dir)
        plot_hbar_metric(summary, difficulty, "latency_mean",
                         "Mean service latency (s/user)",
                         "latency", output_dir, best_largest=False,
                         value_fmt="{:.4f}")
        plot_stacked_energy(summary, difficulty, output_dir)
        plot_hbar_metric(summary, difficulty, "success_rate",
                         "Task success rate",
                         "success", output_dir, best_largest=True,
                         value_fmt="{:.3f}", xlim=(0.7, 1.0))
        plot_reward_boxplot(episodes, summary, difficulty, output_dir)
        plot_cdf_latency(episodes, difficulty, output_dir)
        plot_slot_latency(steps, difficulty, output_dir)
        plot_trajectory(steps, difficulty, output_dir)
        plot_offload_stack(summary, difficulty, output_dir)
    plot_tradeoff(summary, output_dir)
    plot_ablation(root / "ablation", output_dir)
    print(f"All figures saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
