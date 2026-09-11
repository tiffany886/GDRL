import argparse
import subprocess
import sys
from pathlib import Path

from docs.GDRL.experiment_config import SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate commands for small/medium/large SAGIN experiments.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--amn_epochs", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=["random", "trpo_mlp", "ppo_mlp", "gdrl"])
    parser.add_argument("--experiment_root", type=str, default="experiments")
    parser.add_argument("--execute", action="store_true", help="真正执行实验；不加时只打印命令。")
    parser.add_argument("--force_retrain_amn", action="store_true")
    return parser.parse_args()


def build_command(args, scenario):
    cmd = [
        sys.executable,
        "compare_baselines.py",
        "--scenario", scenario,
        "--methods", *args.methods,
        "--episodes", str(args.episodes),
        "--amn_epochs", str(args.amn_epochs),
        "--experiment_root", args.experiment_root,
    ]
    if args.force_retrain_amn:
        cmd.append("--force_retrain_amn")
    return cmd


def main():
    args = parse_args()
    print("Small / medium / large scenario commands:")
    for scenario in SCENARIOS:
        cmd = build_command(args, scenario)
        print(" ".join(cmd))
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
