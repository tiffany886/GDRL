import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path


def load_decisions(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def flatten_decisions(steps):
    nodes = []
    ratios = []
    local = []
    for step in steps:
        nodes.extend(step.get("node_types", []))
        ratios.extend(float(x) for x in step.get("offload_ratios", []))
        local.extend(bool(x) for x in step.get("is_local", []))
    return nodes, ratios, local


def distribution_distance(left, right):
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(k, 0.0) - right.get(k, 0.0)) for k in keys)


def normalize(counter):
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def read_summary(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize decision collapse from saved decisions_*.pkl files."
    )
    parser.add_argument(
        "experiment_dir",
        help="Experiment directory, for example experiments/hybrid_small_v2_1200ep",
    )
    parser.add_argument(
        "--decision-dir",
        default=None,
        help="Optional subdirectory containing decisions_*.pkl. Defaults to experiment_dir, then decision_figures.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if args.decision_dir:
        decision_dir = Path(args.decision_dir)
    elif list(exp_dir.glob("decisions_*.pkl")):
        decision_dir = exp_dir
    else:
        decision_dir = exp_dir / "decision_figures"

    summary_rows = read_summary(exp_dir / "baseline_summary.csv")
    if summary_rows:
        print("Metric summary")
        print("method,reward_mean,latency_mean,energy_mean")
        for row in summary_rows:
            print(
                f"{row.get('method')},"
                f"{float(row.get('reward_mean', 0.0)):.3f},"
                f"{float(row.get('latency_mean', 0.0)):.6g},"
                f"{float(row.get('energy_mean', 0.0)):.6g}"
            )
        print()

    paths = sorted(decision_dir.glob("decisions_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No decisions_*.pkl found in {decision_dir}")

    distributions = {}
    print(f"Decision summary: {decision_dir}")
    print("method,steps,users,node_distribution,ratio_mean,ratio_unique_3dp,local_rate")
    for path in paths:
        method = path.stem.replace("decisions_", "")
        steps = load_decisions(path)
        nodes, ratios, local = flatten_decisions(steps)
        counts = Counter(nodes)
        distributions[method] = normalize(counts)
        ratio_mean = sum(ratios) / len(ratios) if ratios else 0.0
        ratio_unique = len({round(value, 3) for value in ratios})
        local_rate = sum(local) / len(local) if local else 0.0
        print(
            f"{method},{len(steps)},{len(nodes)},"
            f"{dict(sorted(counts.items()))},"
            f"{ratio_mean:.4f},{ratio_unique},{local_rate:.4f}"
        )

    methods = sorted(distributions)
    print()
    print("Pairwise node distribution distance (0=same, 1=disjoint)")
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            dist = distribution_distance(distributions[left], distributions[right])
            print(f"{left} vs {right}: {dist:.4f}")


if __name__ == "__main__":
    main()
