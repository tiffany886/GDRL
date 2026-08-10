import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "random": "Random",
    "local_only": "Local",
    "uav_only": "UAV-only",
    "leo_only": "LEO-only",
    "greedy_partial": "Greedy partial",
    "tea_partial": "TEA partial",
}

COLORS = {
    "random": "#8c8c8c",
    "local_only": "#4c78a8",
    "uav_only": "#59a14f",
    "leo_only": "#f28e2b",
    "greedy_partial": "#b07aa1",
    "tea_partial": "#e15759",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def label(method):
    return METHOD_LABELS.get(method, method)


def color(method):
    return COLORS.get(method, "#666666")


def plot_bar(summary, metric, ylabel, title, output_path):
    methods = [row["method"] for row in summary]
    values = [as_float(row, metric) for row in summary]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, values, color=[color(m) for m in methods], edgecolor="#222222", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_target_stack(summary, output_path):
    methods = [row["method"] for row in summary]
    local = np.array([as_float(row, "target_local_rate") for row in summary])
    uav = np.array([as_float(row, "target_uav_rate") for row in summary])
    leo = np.array([as_float(row, "target_leo_rate") for row in summary])
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(x, local, label="Local", color="#4c78a8", edgecolor="#222222", linewidth=0.5)
    ax.bar(x, uav, bottom=local, label="UAV", color="#59a14f", edgecolor="#222222", linewidth=0.5)
    ax.bar(x, leo, bottom=local + uav, label="LEO", color="#f28e2b", edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m) for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Target selection ratio")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Offloading Target Distribution")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectory(steps, output_path):
    methods = sorted({row["method"] for row in steps})
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for method in methods:
        rows = [row for row in steps if row["method"] == method and int(row["episode"]) == 1]
        if not rows:
            continue
        xs = [as_float(row, "uav_pos_x") for row in rows]
        ys = [as_float(row, "uav_pos_y") for row in rows]
        ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2, label=label(method), color=color(method))
    ax.set_xlabel("UAV x position")
    ax.set_ylabel("UAV y position")
    ax.set_title("UAV Trajectory, Episode 1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot UAV-LEO experiment metrics.")
    parser.add_argument("run_dir", help="Directory containing uav_leo_summary.csv and uav_leo_steps.csv")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(run_dir / "uav_leo_summary.csv")
    steps = read_csv(run_dir / "uav_leo_steps.csv")

    plot_bar(summary, "reward_mean", "Mean episode reward", "Reward Comparison", output_dir / "reward_comparison.png")
    plot_bar(summary, "latency_mean", "Mean latency (s/user)", "Latency Comparison", output_dir / "latency_comparison.png")
    plot_bar(summary, "total_energy_mean", "Mean total energy", "Total Energy Comparison", output_dir / "energy_comparison.png")
    plot_bar(summary, "flight_energy_mean", "Mean UAV flight energy", "UAV Flight Energy Comparison", output_dir / "flight_energy_comparison.png")
    plot_bar(summary, "success_rate", "Success rate", "Task Success Rate", output_dir / "success_rate_comparison.png")
    plot_bar(summary, "offload_ratio_mean", "Mean offload ratio", "Partial Offloading Ratio", output_dir / "offload_ratio_comparison.png")
    plot_target_stack(summary, output_dir / "target_distribution.png")
    plot_trajectory(steps, output_dir / "uav_trajectory.png")
    print(f"Saved figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
