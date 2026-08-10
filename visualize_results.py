import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "random": "Random",
    "trpo_mlp": "TRPO-MLP",
    "ppo_mlp": "PPO-MLP",
    "gdrl": "GDRL (GCN+TRPO)",
    "gdrl_sac": "GDRL-SAC (GCN+SAC)",
}

COLORS = {
    "random": "#8c8c8c",
    "trpo_mlp": "#e07a5f",
    "ppo_mlp": "#3d5a80",
    "gdrl": "#2a9d8f",
    "gdrl_sac": "#f4a261",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GDRL training and baseline comparison plots.")
    parser.add_argument("--project_dir", type=str, default=".")
    parser.add_argument("--single_monitor", type=str, default="monitor_logs.monitor.csv")
    parser.add_argument("--latency_file", type=str, default="latency0.npy")
    parser.add_argument("--compare_dir", type=str, default="compare_results")
    parser.add_argument("--output_dir", type=str, default="visualization_results")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "savefig.bbox": "tight",
    })


def read_monitor(path):
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#")
    if df.empty:
        return None
    df = df.rename(columns={"r": "reward", "l": "episode_length", "t": "elapsed_sec"})
    df["episode"] = np.arange(1, len(df) + 1)
    return df


def select_gdrl_training_df(single_df, episodes_df):
    if episodes_df is not None and not episodes_df.empty and "method" in episodes_df.columns:
        gdrl_df = episodes_df.loc[episodes_df["method"] == "gdrl", ["episode", "reward"]].copy()
        if not gdrl_df.empty:
            gdrl_df["episode"] = np.arange(1, len(gdrl_df) + 1)
            return gdrl_df
    return single_df


def moving_average(values, window=3):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def savefig(path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_single_reward(df, output_dir, dpi):
    if df is None:
        return None
    plt.figure(figsize=(9, 5))
    marker = "o" if len(df) <= 50 else None
    plt.plot(df["episode"], df["reward"], marker=marker, linewidth=1.6, color=COLORS["gdrl"], label="Episode reward")
    window = 3 if len(df) < 50 else 20
    ma = moving_average(df["reward"], window=window)
    if len(ma) != len(df):
        x = df["episode"].iloc[len(df) - len(ma):]
    else:
        x = df["episode"]
    plt.plot(x, ma, linewidth=2.5, color="#264653", label=f"{window}-episode moving average")
    plt.title("GDRL Training Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    path = output_dir / "single_gdrl_reward_curve.png"
    savefig(path, dpi)
    return path


def plot_single_latency(latency_path, output_dir, dpi):
    if not latency_path.exists():
        return None
    lat = np.load(latency_path, allow_pickle=True).astype(float).reshape(-1)
    steps = np.arange(1, len(lat) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(steps, lat * 1000, linewidth=1.2, color="#457b9d")
    axes[0].set_title("Step Latency")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Latency (ms)")

    axes[1].hist(lat * 1000, bins=30, color="#a8dadc", edgecolor="#1d3557")
    axes[1].axvline(np.mean(lat) * 1000, color="#e63946", linewidth=2, label="Mean")
    axes[1].axvline(np.percentile(lat, 95) * 1000, color="#f4a261", linewidth=2, label="P95")
    axes[1].set_title("Latency Distribution")
    axes[1].set_xlabel("Latency (ms)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("GDRL Latency Analysis", fontweight="bold")
    path = output_dir / "single_gdrl_latency.png"
    savefig(path, dpi)
    return path


def plot_baseline_reward_curves(episodes_df, output_dir, dpi, smooth_window=50):
    if episodes_df is None or episodes_df.empty:
        return None
    plt.figure(figsize=(10, 5.5))
    for method, group in episodes_df.groupby("method", sort=False):
        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, None)
        rewards = group["reward"].values
        episodes = group["episode"].values
        # 原始曲线：低透明度背景
        plt.plot(episodes, rewards, linewidth=0.8, color=color, alpha=0.20)
        # 平滑曲线：粗线前景
        window = min(smooth_window, len(rewards) // 4)
        if window >= 3:
            ma = moving_average(rewards, window=window)
            x_ma = episodes[len(episodes) - len(ma):]
            plt.plot(x_ma, ma, linewidth=2.4, color=color, label=f"{label}")
        else:
            plt.plot(episodes, rewards, linewidth=2.2, color=color, label=label)
    plt.title(f"Baseline Reward Curves  (smoothed, window={smooth_window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    path = output_dir / "baseline_reward_curves.png"
    savefig(path, dpi)
    return path


def plot_reward_bar(summary_df, output_dir, dpi):
    if summary_df is None or summary_df.empty:
        return None
    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"]]
    colors = [COLORS.get(m, "#666666") for m in summary_df["method"]]
    x = np.arange(len(summary_df))

    plt.figure(figsize=(9, 5))
    plt.bar(x, summary_df["reward_mean"], yerr=summary_df["reward_std"], capsize=6, color=colors, edgecolor="#222222")
    plt.xticks(x, labels)
    plt.title("Average Reward Comparison")
    plt.xlabel("Method")
    plt.ylabel("Average reward")
    path = output_dir / "baseline_reward_bar.png"
    savefig(path, dpi)
    return path


def plot_latency_bar(summary_df, output_dir, dpi):
    if summary_df is None or summary_df.empty:
        return None
    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"]]
    x = np.arange(len(summary_df))
    width = 0.26

    plt.figure(figsize=(10, 5))
    plt.bar(x - width, summary_df["latency_mean"] * 1000, width, label="Mean", color="#2a9d8f")
    plt.bar(x, summary_df["latency_p95"] * 1000, width, label="P95", color="#e9c46a")
    plt.bar(x + width, summary_df["latency_max"] * 1000, width, label="Max", color="#e76f51")
    plt.xticks(x, labels)
    plt.title("Latency Comparison")
    plt.xlabel("Method")
    plt.ylabel("Latency (ms)")
    plt.legend()
    path = output_dir / "baseline_latency_bar.png"
    savefig(path, dpi)
    return path


def plot_energy_bar(summary_df, output_dir, dpi):
    if summary_df is None or summary_df.empty:
        return None
    if "energy_mean" not in summary_df.columns:
        return None
    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"]]
    colors = [COLORS.get(m, "#666666") for m in summary_df["method"]]
    x = np.arange(len(summary_df))
    width = 0.30

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, summary_df["energy_mean"] * 1e3, width,
            label="Mean (mJ)", color=colors, edgecolor="#222222")
    plt.bar(x + width / 2, summary_df["energy_p95"] * 1e3, width,
            label="P95 (mJ)", color=[c + "99" for c in colors], edgecolor="#222222")
    plt.xticks(x, labels)
    plt.title("Energy Consumption Comparison")
    plt.xlabel("Method")
    plt.ylabel("Energy per step (mJ)")
    plt.legend()
    path = output_dir / "baseline_energy_bar.png"
    savefig(path, dpi)
    return path


def plot_latency_cdf(compare_dir, summary_df, output_dir, dpi):
    """每个方法的延迟 CDF 对比图，从 latency_{method}.npy 读取原始数据。"""
    if summary_df is None or summary_df.empty:
        return None
    methods = summary_df["method"].tolist()
    data = {}
    for m in methods:
        f = compare_dir / f"latency_{m}.npy"
        if f.exists():
            arr = np.load(f, allow_pickle=True).astype(float).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                data[m] = arr
    if not data:
        return None

    plt.figure(figsize=(10, 5.5))
    for m, arr in data.items():
        sorted_lat = np.sort(arr) * 1000  # → ms
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        color = COLORS.get(m, "#666666")
        label = METHOD_LABELS.get(m, m)
        plt.plot(sorted_lat, cdf, linewidth=2.0, color=color, label=label)
        p95 = float(np.percentile(arr, 95)) * 1000
        plt.axvline(p95, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

    # 截断 x 轴到各方法 P99 的最大值，去除极端 outliers 避免曲线被压缩
    all_vals = np.concatenate([arr for arr in data.values()])
    x_max = float(np.percentile(all_vals, 99)) * 1000
    plt.title("Latency CDF Comparison")
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.xlim(0, x_max)
    plt.ylim(0, 1.02)
    plt.legend()
    path = output_dir / "baseline_latency_cdf.png"
    savefig(path, dpi)
    return path


def plot_tradeoff(summary_df, output_dir, dpi):
    if summary_df is None or summary_df.empty:
        return None
    plt.figure(figsize=(8, 5.5))
    for _, row in summary_df.iterrows():
        method = row["method"]
        plt.scatter(row["latency_p95"] * 1000, row["reward_mean"], s=150,
                    color=COLORS.get(method, "#666666"), edgecolor="#222222")
        plt.annotate(METHOD_LABELS.get(method, method),
                     (row["latency_p95"] * 1000, row["reward_mean"]),
                     textcoords="offset points", xytext=(8, 8))
    plt.title("Reward-Latency Tradeoff")
    plt.xlabel("P95 latency (ms)")
    plt.ylabel("Average reward")
    path = output_dir / "baseline_reward_latency_tradeoff.png"
    savefig(path, dpi)
    return path


def plot_dashboard(single_df, summary_df, episodes_df, latency_path, output_dir, dpi):
    if single_df is None or summary_df is None or episodes_df is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    marker = "o" if len(single_df) <= 50 else None
    axes[0, 0].plot(single_df["episode"], single_df["reward"], marker=marker, color=COLORS["gdrl"], linewidth=1.6)
    window = 3 if len(single_df) < 50 else 20
    ma = moving_average(single_df["reward"], window=window)
    if len(ma) != len(single_df):
        x = single_df["episode"].iloc[len(single_df) - len(ma):]
    else:
        x = single_df["episode"]
    axes[0, 0].plot(x, ma, color="#264653", linewidth=2.4, label=f"{window}-episode moving average")
    axes[0, 0].set_title("GDRL Training Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    for method, group in episodes_df.groupby("method", sort=False):
        axes[0, 1].plot(group["episode"], group["reward"], marker="o", linewidth=2,
                        color=COLORS.get(method), label=METHOD_LABELS.get(method, method))
    axes[0, 1].set_title("Baseline Reward Curves")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()

    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"]]
    colors = [COLORS.get(m, "#666666") for m in summary_df["method"]]
    axes[1, 0].bar(labels, summary_df["reward_mean"], color=colors, edgecolor="#222222")
    axes[1, 0].set_title("Average Reward")
    axes[1, 0].set_ylabel("Reward")

    if latency_path.exists():
        lat = np.load(latency_path, allow_pickle=True).astype(float).reshape(-1)
        axes[1, 1].hist(lat * 1000, bins=30, color="#a8dadc", edgecolor="#1d3557")
        axes[1, 1].axvline(np.mean(lat) * 1000, color="#e63946", linewidth=2, label="Mean")
        axes[1, 1].axvline(np.percentile(lat, 95) * 1000, color="#f4a261", linewidth=2, label="P95")
        axes[1, 1].legend()
        axes[1, 1].set_title("GDRL Latency Distribution")
        axes[1, 1].set_xlabel("Latency (ms)")
        axes[1, 1].set_ylabel("Count")
    else:
        x = np.arange(len(summary_df))
        width = 0.26
        axes[1, 1].bar(x - width, summary_df["latency_mean"] * 1000, width, label="Mean", color="#2a9d8f")
        axes[1, 1].bar(x, summary_df["latency_p95"] * 1000, width, label="P95", color="#e9c46a")
        axes[1, 1].bar(x + width, summary_df["latency_max"] * 1000, width, label="Max", color="#e76f51")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(labels)
        axes[1, 1].set_title("Latency Summary")
        axes[1, 1].set_xlabel("Method")
        axes[1, 1].set_ylabel("Latency (ms)")
        axes[1, 1].legend()

    fig.suptitle("GDRL Experiment Dashboard", fontsize=16, fontweight="bold")
    path = output_dir / "experiment_dashboard.png"
    savefig(path, dpi)
    return path


def main():
    args = parse_args()
    setup_style()

    project_dir = Path(args.project_dir).resolve()
    output_dir = (project_dir / args.output_dir).resolve()
    compare_dir = (project_dir / args.compare_dir).resolve()
    single_monitor = project_dir / args.single_monitor
    latency_path = project_dir / args.latency_file

    summary_path = compare_dir / "baseline_summary.csv"
    episodes_path = compare_dir / "baseline_episodes.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    episodes_df = pd.read_csv(episodes_path) if episodes_path.exists() else None
    single_df = select_gdrl_training_df(read_monitor(single_monitor), episodes_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for path in [
        plot_single_reward(single_df, output_dir, args.dpi),
        plot_single_latency(latency_path, output_dir, args.dpi),
        plot_baseline_reward_curves(episodes_df, output_dir, args.dpi),
        plot_reward_bar(summary_df, output_dir, args.dpi),
        plot_latency_bar(summary_df, output_dir, args.dpi),
        plot_latency_cdf(compare_dir, summary_df, output_dir, args.dpi),
        plot_tradeoff(summary_df, output_dir, args.dpi),
        plot_energy_bar(summary_df, output_dir, args.dpi),
        plot_dashboard(single_df, summary_df, episodes_df, latency_path, output_dir, args.dpi),
    ]:
        if path is not None:
            generated.append(path)

    print("Generated plots:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
