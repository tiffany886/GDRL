"""Produce the final comparison table and paired-significance figure for GDRL.

Usage:  python -m uav_leo_experiment.analyze_results <sweep_dir> [--output_dir DIR]

Reads sweep_summary.csv + sweep_episodes.csv written by run_sweep.py, computes
per-episode paired differences of the proposed method (gdrl) against every
baseline and prints a Markdown-ready table plus a significance bar figure.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

METHOD_ORDER = ["gdrl", "predict_tea", "follow_tea", "deadline_tea", "tea_partial",
                "energy_guarded_tea", "full_offload_tea", "greedy_partial", "lyapunov",
                "rate_aware", "uav_only", "leo_only", "ppo", "dqn", "td3", "sac",
                "random", "local_only"]
SHORT = {
    "gdrl": "GDRL", "predict_tea": "Predict-TEA", "follow_tea": "Follow-TEA",
    "deadline_tea": "Deadline-TEA", "tea_partial": "TEA", "energy_guarded_tea": "E-Guard TEA",
    "full_offload_tea": "Full-offload", "greedy_partial": "Greedy", "lyapunov": "Lyapunov",
    "rate_aware": "Rate-aware", "uav_only": "UAV-only", "leo_only": "LEO-only",
    "ppo": "PPO", "dqn": "DQN", "td3": "TD3", "sac": "SAC", "random": "Random",
    "local_only": "Local-only",
}


def read_rows(sweep_dir):
    summary = list(csv.DictReader((sweep_dir / "sweep_summary.csv").open(encoding="utf8")))
    episodes = list(csv.DictReader((sweep_dir / "sweep_episodes.csv").open(encoding="utf8")))
    return summary, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, episodes = read_rows(sweep_dir)

    lines = ["# GDRL vs baselines (paired over episodes)\n"]
    for difficulty in sorted({r["difficulty"] for r in summary}):
        rows = [r for r in summary if r["difficulty"] == difficulty]
        by_method = {}
        for r in rows:
            by_method[r["method"]] = r
        ep = defaultdict(dict)
        for r in episodes:
            if r["difficulty"] == difficulty:
                ep[r["method"]][int(r["episode"])] = float(r["reward"])
        g = ep.get("gdrl", {})
        lines.append(f"\n## {difficulty}\n")
        lines.append("| method | reward | success | latency | energy | diff vs GDRL | wins | p |")
        lines.append("|---|---|---|---|---|---|---|---|")
        ranked = sorted(rows, key=lambda r: float(r["reward_mean"]))
        for r in ranked:
            m = r["method"]
            diff = wins = p = ""
            if m != "gdrl" and g and m in ep and ep[m]:
                common = sorted(set(g) & set(ep[m]))
                if len(common) >= 5:
                    gv = np.array([g[e] for e in common])
                    mv = np.array([ep[m][e] for e in common])
                    d = gv - mv
                    diff = f"{d.mean():+.1f}"
                    wins = f"{int((d > 0).sum())}/{len(common)}"
                    p = f"{stats.ttest_rel(gv, mv).pvalue:.4f}"
            lines.append(
                f"| {SHORT.get(m, m)} | {float(r['reward_mean']):.1f} | "
                f"{float(r['success_rate']):.3f} | {float(r['latency_mean']):.4f} | "
                f"{float(r['total_energy_mean']):.1f} | {diff} | {wins} | {p} |"
            )

        # paired-difference figure (GDRL minus baseline; positive = GDRL better)
        if g and len(g) >= 5:
            names, means, pvals, colors = [], [], [], []
            for m in METHOD_ORDER:
                if m == "gdrl" or m not in ep or not ep[m]:
                    continue
                common = sorted(set(g) & set(ep[m]))
                if len(common) < 5:
                    continue
                gv = np.array([g[e] for e in common])
                mv = np.array([ep[m][e] for e in common])
                means.append(float((gv - mv).mean()))
                pvals.append(float(stats.ttest_rel(gv, mv).pvalue))
                colors.append("#2ca02c" if means[-1] > 0 and pvals[-1] < 0.05 else
                              "#d62728" if pvals[-1] < 0.05 else "#9e9e9e")
                names.append(SHORT.get(m, m))
            order = np.argsort(means)
            names = [names[i] for i in order]
            means = [means[i] for i in order]
            colors = [colors[i] for i in order]
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            bars = ax.barh(names, means, color=colors, edgecolor="#222222", linewidth=0.4)
            for b, (mm, pp) in zip(bars, zip(means, pvals)):
                ax.text(mm + (0.5 if mm >= 0 else -0.5), b.get_y() + b.get_height() / 2,
                        f"{mm:+.1f}" + ("" if pp >= 0.05 else "*"), va="center",
                        ha="left" if mm >= 0 else "right", fontsize=8)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_xlabel("Paired reward difference (GDRL - baseline); * p<0.05")
            ax.set_title(f"{difficulty}: per-episode gain of GDRL over baselines")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / f"gdrl_gain_{difficulty}.png", dpi=180)
            plt.close(fig)

    (sweep_dir / "gdrl_comparison.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))
    print(f"\nSaved figure + markdown to {sweep_dir}")


if __name__ == "__main__":
    main()
