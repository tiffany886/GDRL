﻿# -*- coding: utf-8 -*-
"""Ablation (nb/cc) + energy_weight sensitivity for PMEO-M-Eco at k3.

Usage:
  python _run_ablation_energy.py --mode ablation
  python _run_ablation_energy.py --mode energy
"""
import argparse, json, time
from pathlib import Path
from dataclasses import replace

from docs.GDRL.uav_leo_experiment.run_multi_uav import build_config
from docs.GDRL.uav_leo_experiment.run_experiment import run_policy, write_csv
from docs.GDRL.uav_leo_experiment.multi_uav import multi_uav_policies

SEEDS = [1, 7, 42, 73, 314, 555, 888, 999, 12345, 2024]
PRESET = "k3"
EPISODES = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablation", "energy"], required=True)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods = {p.name: p for p in multi_uav_policies()}
    base_out = Path("experiments/uav_leo_v2x_paper_final")

    if args.mode == "ablation":
        wanted = ["pmeo_m_eco_nb", "pmeo_m_eco_cc"]
        out = base_out / "multi_uav_ablation2"
        out.mkdir(parents=True, exist_ok=True)
        summaries, episodes = [], []
        t_all = time.time()
        for seed in seeds:
            cfg = build_config(PRESET, seed, args.episodes)
            tag = f"seed{seed}"
            for name in wanted:
                pol = methods[name]
                t0 = time.time()
                summary, ep_rows, _ = run_policy(pol, cfg)
                summary.update({"preset": PRESET, "seed": seed,
                                "runtime_s": round(time.time() - t0, 1)})
                summaries.append(summary)
                for e in ep_rows:
                    e.update({"preset": PRESET, "seed": seed})
                    episodes.append(e)
                print(f"[{tag}] {name}: reward={summary['reward_mean']:.1f} "
                      f"succ={summary['success_rate']:.3f} en={summary['total_energy_mean']:.1f} "
                      f"({summary['runtime_s']}s)", flush=True)
        write_csv(out / "multi_uav_summary.csv", summaries)
        write_csv(out / "multi_uav_episodes.csv", episodes)
        print(f"ablation done in {time.time()-t_all:.1f}s -> {out / 'multi_uav_summary.csv'}")

    else:
        weights = [0.0001, 0.001, 0.01]
        wanted = ["pmeo_m", "pmeo_m_eco", "mpc_m_h3"]
        out = base_out / "multi_uav_energy_sweep"
        out.mkdir(parents=True, exist_ok=True)
        summaries, episodes = [], []
        t_all = time.time()
        for w in weights:
            for seed in seeds:
                base = build_config(PRESET, seed, args.episodes)
                cfg = replace(base, energy_weight=w)
                tag = f"w{w:g}/seed{seed}"
                run_dir = out / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_config.json").write_text(
                    json.dumps(cfg.__dict__, indent=2, default=str), encoding="utf-8")
                for name in wanted:
                    pol = methods[name]
                    t0 = time.time()
                    summary, ep_rows, _ = run_policy(pol, cfg)
                    summary.update({"preset": PRESET, "energy_weight": w, "seed": seed,
                                    "runtime_s": round(time.time() - t0, 1)})
                    summaries.append(summary)
                    for e in ep_rows:
                        e.update({"preset": PRESET, "energy_weight": w, "seed": seed})
                        episodes.append(e)
                    print(f"[{tag}] {name}: reward={summary['reward_mean']:.1f} "
                          f"succ={summary['success_rate']:.3f} en={summary['total_energy_mean']:.1f} "
                          f"({summary['runtime_s']}s)", flush=True)
        write_csv(out / "multi_uav_summary.csv", summaries)
        write_csv(out / "multi_uav_episodes.csv", episodes)
        print(f"energy sweep done in {time.time()-t_all:.1f}s -> {out / 'multi_uav_summary.csv'}")


if __name__ == "__main__":
    main()
