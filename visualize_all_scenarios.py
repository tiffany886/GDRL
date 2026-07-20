from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = [
    ("small", "small_U3_L8_N8_T100"),
    ("medium", "medium_U6_L12_N12_T100"),
    ("large", "large_U10_L16_N16_T100"),
]

METHOD_LABELS = {
    "random": "Random",
    "trpo_mlp": "TRPO-MLP",
    "ppo_mlp": "PPO-MLP",
    "gdrl": "GDRL",
}

METHODS = ["random", "trpo_mlp", "ppo_mlp", "gdrl"]

COLORS = {
    "random": "#8c8c8c",
    "trpo_mlp": "#e07a5f",
    "ppo_mlp": "#3d5a80",
    "gdrl": "#2a9d8f",
}


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "savefig.bbox": "tight",
    })


def load_results(project_dir):
    summary_frames = []
    episode_frames = []
    for scenario_label, scenario_dir in SCENARIOS:
        compare_dir = project_dir / "experiments" / scenario_dir / "compare_600ep_T100"
        summary_path = compare_dir / "baseline_summary.csv"
        episodes_path = compare_dir / "baseline_episodes.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary.insert(0, "scenario", scenario_label)
            summary_frames.append(summary)
        if episodes_path.exists():
            episodes = pd.read_csv(episodes_path)
            episodes.insert(0, "scenario", scenario_label)
            episode_frames.append(episodes)
    if not summary_frames:
        raise FileNotFoundError("No baseline_summary.csv files found under experiments/*/compare_600ep_T100.")
    summary_df = pd.concat(summary_frames, ignore_index=True)
    episodes_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    return summary_df, episodes_df


def savefig(path, dpi=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_grouped_bars(summary_df, metric, ylabel, title, output_path, scale=1.0):
    scenarios = [name for name, _ in SCENARIOS]
    x = np.arange(len(scenarios))
    width = 0.19

    plt.figure(figsize=(10, 5.4))
    for index, method in enumerate(METHODS):
        values = []
        for scenario in scenarios:
            row = summary_df[(summary_df["scenario"] == scenario) & (summary_df["method"] == method)]
            values.append(float(row[metric].iloc[0]) * scale if not row.empty else np.nan)
        offset = (index - (len(METHODS) - 1) / 2) * width
        plt.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            edgecolor="#222222",
            linewidth=0.7,
        )

    plt.xticks(x, [scenario.capitalize() for scenario in scenarios])
    plt.title(title)
    plt.xlabel("Scenario")
    plt.ylabel(ylabel)
    plt.legend(ncol=4, loc="upper left")
    savefig(output_path)


def plot_gdrl_gaps(summary_df, output_path):
    rows = []
    for scenario, _ in SCENARIOS:
        scenario_df = summary_df[summary_df["scenario"] == scenario]
        gdrl = scenario_df[scenario_df["method"] == "gdrl"].iloc[0]
        mlp_df = scenario_df[scenario_df["method"].isin(["trpo_mlp", "ppo_mlp"])]
        best_mlp_reward = mlp_df["reward_mean"].max()
        best_mlp_p95 = mlp_df["latency_p95"].min()
        rows.append({
            "scenario": scenario,
            "reward_gap": gdrl["reward_mean"] - best_mlp_reward,
            "p95_latency_gap_ms": (gdrl["latency_p95"] - best_mlp_p95) * 1000,
        })
    gap_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors_reward = ["#2a9d8f" if value >= 0 else "#e76f51" for value in gap_df["reward_gap"]]
    colors_latency = ["#2a9d8f" if value <= 0 else "#e76f51" for value in gap_df["p95_latency_gap_ms"]]

    axes[0].bar(gap_df["scenario"].str.capitalize(), gap_df["reward_gap"], color=colors_reward, edgecolor="#222222")
    axes[0].axhline(0, color="#222222", linewidth=1)
    axes[0].set_title("GDRL Reward Gap vs Best MLP")
    axes[0].set_ylabel("Reward gap")

    axes[1].bar(gap_df["scenario"].str.capitalize(), gap_df["p95_latency_gap_ms"], color=colors_latency, edgecolor="#222222")
    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].set_title("GDRL P95 Latency Gap vs Best MLP")
    axes[1].set_ylabel("P95 latency gap (ms)")

    fig.suptitle("GDRL Advantage/Disadvantage Summary", fontsize=15, fontweight="bold")
    savefig(output_path)
    return gap_df


def plot_reward_curves_grid(episodes_df, output_path):
    if episodes_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for axis, (scenario, _) in zip(axes, SCENARIOS):
        scenario_df = episodes_df[episodes_df["scenario"] == scenario]
        for method in METHODS:
            group = scenario_df[scenario_df["method"] == method].sort_values("episode")
            if group.empty:
                continue
            rewards = group["reward"].rolling(window=20, min_periods=1).mean()
            axis.plot(
                group["episode"],
                rewards,
                linewidth=1.7,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_title(f"{scenario.capitalize()} Reward Curve")
        axis.set_xlabel("Episode")
        axis.set_ylabel("20-episode moving average reward")
    axes[0].legend(loc="upper left")
    fig.suptitle("Reward Curves Across Scenarios", fontsize=15, fontweight="bold")
    savefig(output_path)


def write_rank_table(summary_df, output_path):
    rows = []
    for scenario, _ in SCENARIOS:
        scenario_df = summary_df[summary_df["scenario"] == scenario].copy()
        reward_best = scenario_df.loc[scenario_df["reward_mean"].idxmax()]
        p95_best = scenario_df.loc[scenario_df["latency_p95"].idxmin()]
        mean_latency_best = scenario_df.loc[scenario_df["latency_mean"].idxmin()]
        gdrl = scenario_df[scenario_df["method"] == "gdrl"].iloc[0]
        mlp_df = scenario_df[scenario_df["method"].isin(["trpo_mlp", "ppo_mlp"])]
        rows.append({
            "scenario": scenario,
            "best_reward_method": reward_best["method"],
            "best_reward_mean": reward_best["reward_mean"],
            "best_p95_latency_method": p95_best["method"],
            "best_p95_latency_ms": p95_best["latency_p95"] * 1000,
            "best_mean_latency_method": mean_latency_best["method"],
            "best_mean_latency_ms": mean_latency_best["latency_mean"] * 1000,
            "gdrl_reward_mean": gdrl["reward_mean"],
            "gdrl_p95_latency_ms": gdrl["latency_p95"] * 1000,
            "gdrl_reward_gap_vs_best_mlp": gdrl["reward_mean"] - mlp_df["reward_mean"].max(),
            "gdrl_p95_gap_vs_best_mlp_ms": (gdrl["latency_p95"] - mlp_df["latency_p95"].min()) * 1000,
        })
    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(output_path, index=False)
    return rank_df


def main():
    setup_style()
    project_dir = Path(".").resolve()
    output_dir = project_dir / "experiments" / "scenario_overview_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, episodes_df = load_results(project_dir)
    summary_df.to_csv(output_dir / "all_scenarios_summary.csv", index=False)
    rank_df = write_rank_table(summary_df, output_dir / "scenario_rank_table.csv")
    gap_df = plot_gdrl_gaps(summary_df, output_dir / "gdrl_gap_vs_best_mlp.png")

    plot_grouped_bars(
        summary_df,
        metric="reward_mean",
        ylabel="Average reward",
        title="Average Reward Across Scenarios",
        output_path=output_dir / "all_scenarios_reward_mean.png",
    )
    plot_grouped_bars(
        summary_df,
        metric="latency_p95",
        ylabel="P95 latency (ms)",
        title="P95 Latency Across Scenarios",
        output_path=output_dir / "all_scenarios_latency_p95.png",
        scale=1000,
    )
    plot_grouped_bars(
        summary_df,
        metric="latency_mean",
        ylabel="Mean latency (ms)",
        title="Mean Latency Across Scenarios",
        output_path=output_dir / "all_scenarios_latency_mean.png",
        scale=1000,
    )
    plot_reward_curves_grid(episodes_df, output_dir / "all_scenarios_reward_curves.png")

    print("Generated overview files:")
    for path in sorted(output_dir.iterdir()):
        print(path)
    print("\nScenario rank table:")
    print(rank_df.to_string(index=False))
    print("\nGDRL gap table:")
    print(gap_df.to_string(index=False))


if __name__ == "__main__":
    main()
