"""Evaluate no-RL baselines B1-B3 (W2.8): one episode per (method, seed)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..baselines.no_rl import make_policy
from ..env.config import DEFAULT_CONFIG_PATH, load_config
from ..env.env import DisasterEnv
from ..env.tasks import APP_DONE, APP_FAILED, APP_PENDING


def run_one(method: str, seed: int, cfg_overrides=None) -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH, overrides=cfg_overrides or {})
    cfg.seed = seed
    env = DisasterEnv(cfg)
    policy = make_policy(method, seed=seed)
    done = False
    while not done:
        obs, reward, done, info = env.step(policy.act(env))
    counts = {APP_PENDING: 0, APP_DONE: 0, APP_FAILED: 0}
    for a in env.apps:
        counts[a.state] += 1
    arrived = len(env.apps)
    done_n = counts[APP_DONE]
    return {
        "method": method,
        "seed": seed,
        "apps_arrived": arrived,
        "apps_done": done_n,
        "apps_failed": counts[APP_FAILED],
        "apps_pending": counts[APP_PENDING],
        "success_rate": done_n / max(arrived, 1),
        "mean_delay_s": env.metrics["done_delay_sum_s"] / max(done_n, 1),
        "energy_j": env.metrics["energy_total_j"],
        "flight_energy_j": env.metrics["flight_energy_j"],
        "comm_energy_j": env.metrics["comm_energy_j"],
        "compute_energy_j": env.metrics["compute_energy_j"],
        "leo_placements": env.metrics["leo_placements"],
        "leo_invalid_attempts": env.metrics["leo_invalid_attempts"],
        "reward": env.metrics["reward_sum"],
    }


def main():
    ap = argparse.ArgumentParser(description="B1-B3 no-RL metrics")
    ap.add_argument("--methods", default="hover,chase,sweep,random")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parent.parent / "results" / "baselines_w2.csv"
    ))
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        for seed in seeds:
            rows.append(run_one(method, seed))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_path}")
    cols = ["method", "success_rate", "mean_delay_s", "energy_j", "leo_invalid_attempts"]
    print("\t".join(cols))
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        print("\t".join([
            method,
            f"{np.mean([r['success_rate'] for r in sub]):.3f}",
            f"{np.mean([r['mean_delay_s'] for r in sub]):.3f}",
            f"{np.mean([r['energy_j'] for r in sub]):.0f}",
            f"{np.mean([r['leo_invalid_attempts'] for r in sub]):.0f}",
        ]))


if __name__ == "__main__":
    main()
