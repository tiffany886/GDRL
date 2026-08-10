import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIFFICULTY_ORDER = ["easy", "medium", "hard", "stress",
                   "v2x_easy", "v2x_medium", "v2x_hard", "v2x_stress",
                   "v2x_hotspot_hard", "v2x_hotspot_stress"]
ABLATION_ORDER = ["none", "fixed_uav", "no_flight_energy", "full_offload_only"]
METHOD_LABELS = {
    "random": "Random",
    "local_only": "Local",
    "uav_only": "UAV-only",
    "leo_only": "LEO-only",
    "greedy_partial": "Greedy",
    "rate_aware": "Rate-aware",
    "full_offload_tea": "Full-offload TEA",
    "energy_guarded_tea": "Energy-guarded TEA",
    "deadline_tea": "Deadline TEA",
    "tea_partial": "TEA partial",
    "adaptive_tea": "Adaptive TEA",
    "follow_tea": "Follow-TEA",
    "predict_tea": "Predict-TEA",
    "mpc_traj_h3": "MPC-H3 (receding horizon)",
    "mpc_traj_h5": "MPC-H5 (receding horizon)",
    "gdrl": "GDRL (ours)",
    "lyapunov": "Lyapunov",
    "genetic": "GA",
    "pso": "PSO",
    "sa": "SA",
    "aco": "ACO",
    "exhaustive_optimal": "Exhaustive",
    "ppo": "PPO",
    "dqn": "DQN",
    "d3qn": "P-D3QN",
    "gat_ppo": "GAT-PPO",
    "transformer_ppo": "Transformer-PPO",
    "ddqn": "DDQN",
    "ddpg": "DDPG",
    "td3": "TD3",
    "sac": "SAC",
}
COLORS = {
    "random": "#8c8c8c",
    "local_only": "#4c78a8",
    "uav_only": "#59a14f",
    "leo_only": "#f28e2b",
    "greedy_partial": "#b07aa1",
    "rate_aware": "#76b7b2",
    "full_offload_tea": "#edc948",
    "energy_guarded_tea": "#9c755f",
    "deadline_tea": "#ff9da7",
    "tea_partial": "#e15759",
    "adaptive_tea": "#d62728",
    "follow_tea": "#9467bd",
    "predict_tea": "#8c564b",
    "mpc_traj_h3": "#17becf",
    "mpc_traj_h5": "#17becf",
    "gdrl": "#e31a1c",
    "lyapunov": "#6a51a3",
    "genetic": "#33a02c",
    "pso": "#1f78b4",
    "sa": "#c7c7c7",
    "aco": "#2ca02c",
    "exhaustive_optimal": "#ff7f00",
    "ppo": "#a6cee3",
    "dqn": "#fb9a99",
    "d3qn": "#e377c2",
    "gat_ppo": "#8c564b",
    "transformer_ppo": "#7f7f7f",
    "ddqn": "#fdb462",
    "ddpg": "#ff9896",
    "td3": "#a6cee3",
    "sac": "#bcbd22",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def label(method):
    return METHOD_LABELS.get(method, method)


def color(method):
    return COLORS.get(method, "#666666")


def filter_rows(rows, ablation="none"):
    return [row for row in rows if row["ablation"] == ablation]


def methods_in(rows):
    return sorted({row["method"] for row in rows})


def difficulties_in(rows):
    present = {row["difficulty"] for row in rows}
    return [d for d in DIFFICULTY_ORDER if d in present]


def plot_metric_by_difficulty(rows, metric, ylabel, title, output_path, ablation="none", selected=None):
    rows = filter_rows(rows, ablation)
    methods = methods_in(rows)
    if selected:
        methods = [method for method in selected if method in methods]
    difficulties = difficulties_in(rows)
    by_key = {(row["difficulty"], row["method"]): f(row, metric) for row in rows}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(len(difficulties))
    for method in methods:
        y = [by_key.get((difficulty, method), np.nan) for difficulty in difficulties]
        ax.plot(x, y, marker="o", markersize=4.5, linewidth=1.6,
                label=label(method), color=color(method), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_ablation_for_method(rows, method, metric, ylabel, title, output_path):
    rows = [row for row in rows if row["method"] == method]
    difficulties = difficulties_in(rows)
    by_key = {(row["difficulty"], row["ablation"]): f(row, metric) for row in rows}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(difficulties))
    width = 0.18
    for idx, ablation in enumerate(ABLATION_ORDER):
        y = [by_key.get((difficulty, ablation), np.nan) for difficulty in difficulties]
        ax.bar(x + (idx - 1.5) * width, y, width=width, label=ablation, edgecolor="#222222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_latency_energy_scatter(rows, output_path, ablation="none"):
    rows = filter_rows(rows, ablation)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in rows:
        method = row["method"]
        difficulty = row["difficulty"]
        ax.scatter(f(row, "latency_mean"), f(row, "total_energy_mean"), color=color(method), s=52, alpha=0.82)
        if method in ("tea_partial", "deadline_tea", "energy_guarded_tea"):
            ax.annotate(difficulty, (f(row, "latency_mean"), f(row, "total_energy_mean")), fontsize=7)
    handles = []
    labels = []
    for method in methods_in(rows):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color(method), markersize=7))
        labels.append(label(method))
    ax.set_xlabel("Mean latency (s/user)")
    ax.set_ylabel("Mean total energy")
    ax.set_title("Latency-Energy Tradeoff")
    ax.grid(alpha=0.25)
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot UAV-LEO sweep metrics.")
    parser.add_argument("sweep_dir", help="Directory containing sweep_summary.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--selected", nargs="+", default=["local_only", "uav_only", "leo_only", "greedy_partial", "full_offload_tea", "deadline_tea", "tea_partial", "follow_tea", "predict_tea", "lyapunov", "gdrl", "ppo", "dqn", "td3", "sac"], help="Methods to show in main sweep line charts.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(sweep_dir / "sweep_summary.csv")

    selected = args.selected
    plot_metric_by_difficulty(rows, "reward_mean", "Mean episode reward", "Reward vs Difficulty", output_dir / "sweep_reward.png", selected=selected)
    plot_metric_by_difficulty(rows, "success_rate", "Success rate", "Success Rate vs Difficulty", output_dir / "sweep_success_rate.png", selected=selected)
    plot_metric_by_difficulty(rows, "latency_mean", "Mean latency (s/user)", "Latency vs Difficulty", output_dir / "sweep_latency.png", selected=selected)
    plot_metric_by_difficulty(rows, "total_energy_mean", "Mean total energy", "Energy vs Difficulty", output_dir / "sweep_energy.png", selected=selected)
    plot_metric_by_difficulty(rows, "offload_ratio_mean", "Mean offload ratio", "Offload Ratio vs Difficulty", output_dir / "sweep_offload_ratio.png", selected=selected)
    plot_latency_energy_scatter(rows, output_dir / "sweep_latency_energy_tradeoff.png")
    plot_ablation_for_method(rows, "tea_partial", "reward_mean", "Mean episode reward", "TEA Partial Ablation: Reward", output_dir / "ablation_tea_reward.png")
    plot_ablation_for_method(rows, "tea_partial", "success_rate", "Success rate", "TEA Partial Ablation: Success Rate", output_dir / "ablation_tea_success.png")
    plot_ablation_for_method(rows, "tea_partial", "total_energy_mean", "Mean total energy", "TEA Partial Ablation: Energy", output_dir / "ablation_tea_energy.png")
    print(f"Saved sweep figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
