"""Scale-sweep driver: area/road/hotspot scaling for the UAV-LEO hotspot line.

The user/LEO/hotspot geometry in env.py is anchored to ``road_half_len`` and
``area_size`` (see env.reset): enlarging only ``area_size`` does NOT change the
user distribution, so this script scales the whole geometry:

  line="geo"  (geometric / time-consistent):
      area_size, road_half_len, hotspot_radius, hotspot_speed, uav_speed_max,
      user_speed_max all scale by s -> identical trajectory shapes, larger
      absolute link distances (pure "map is bigger" effect).
  line="real" (realistic fixed UAV speed):
      only area_size, road_half_len, hotspot_radius scale by s; UAV/hotspot
      speeds stay at the physical 25/18 m/s baseline -> the UAV must cover a
      larger area at the same speed (harder coverage problem).

Outputs, per (line, scale, seed): summary row + episode rows, plus one merged
CSV for all runs (scale_summary.csv / scale_episodes.csv).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .baselines import default_policies
from .config import make_config
from .mpc_traj import MPCTrajPolicy
from .run_experiment import run_policy, write_csv

BASE_GEOM = {
    "v2x_hotspot_hard": dict(area_size=1000.0, road_half_len=350.0, hotspot_radius=100.0,
                             hotspot_speed=18.0, uav_speed_max=25.0, user_speed_max=16.0),
    "v2x_hotspot_stress": dict(area_size=1000.0, road_half_len=350.0, hotspot_radius=90.0,
                               hotspot_speed=20.0, uav_speed_max=25.0, user_speed_max=18.0),
}


def scale_overrides(difficulty, s, line):
    b = BASE_GEOM[difficulty]
    ov = {
        "area_size": b["area_size"] * s,
        "road_half_len": b["road_half_len"] * s,
        "hotspot_radius": b["hotspot_radius"] * s,
    }
    if line == "geo":
        ov["hotspot_speed"] = b["hotspot_speed"] * s
        ov["uav_speed_max"] = b["uav_speed_max"] * s
        ov["user_speed_max"] = b["user_speed_max"] * s
    return ov


def build_policies(method_names):
    by_name = {pol.name: pol for pol in default_policies()}
    by_name["mpc_traj_h3"] = MPCTrajPolicy(horizon=3, offload="exact")
    if not method_names:
        names = ["postmove_exact", "current_exact", "predict_tea", "follow_tea",
                 "pmeo_e", "mpc_traj_h3", "random", "local_only", "uav_only", "leo_only"]
    else:
        names = [n.strip() for n in method_names.split(",") if n.strip()]
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit("unknown methods: %s (available: %s)" % (missing, sorted(by_name)))
    return [by_name[n] for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", default="v2x_hotspot_hard",
                    choices=["v2x_hotspot_hard", "v2x_hotspot_stress"])
    ap.add_argument("--lines", default="geo,real")
    ap.add_argument("--scales", default="1.0,1.5,2.0,3.0,5.0")
    ap.add_argument("--seeds", default="73")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--output_dir", default="experiments/uav_leo_v2x_paper_final/scale_sweep")
    args = ap.parse_args()

    lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_episodes = []
    running_csv = out / "scale_summary_running.csv"
    running_written = False
    total_t0 = time.time()
    for line in lines:
        for s in scales:
            for seed in seeds:
                cfg = make_config(args.difficulty, ablation="none", seed=seed,
                                  **scale_overrides(args.difficulty, s, line))
                cfg.episodes = args.episodes
                tag = f"{line}/s{s:g}/seed{seed}"
                run_dir = out / tag
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "run_config.json").write_text(
                    json.dumps(cfg.__dict__, indent=2, default=str), encoding="utf-8")
                for pol in build_policies(args.methods):
                    t0 = time.time()
                    summary, episodes, _ = run_policy(pol, cfg)
                    summary.update({"line": line, "scale": s, "seed": seed,
                                    "area_km": cfg.area_size / 1000.0,
                                    "road_km": 2.0 * cfg.road_half_len / 1000.0,
                                    "uav_speed": cfg.uav_speed_max,
                                    "runtime_s": round(time.time() - t0, 1)})
                    all_summaries.append(summary)
                    for e in episodes:
                        e.update({"line": line, "scale": s, "seed": seed,
                                  "area_km": cfg.area_size / 1000.0})
                        all_episodes.append(e)
                    print(f"[{tag}] {pol.name}: reward={summary['reward_mean']:.3f} "
                          f"success={summary['success_rate']:.3f} "
                          f"energy={summary['total_energy_mean']:.1f} "
                          f"({summary['runtime_s']}s)", flush=True)
                    if not running_written:
                        with running_csv.open("w", newline="") as h:
                            csv.DictWriter(h, fieldnames=list(summary.keys())).writeheader()
                        running_written = True
                    with running_csv.open("a", newline="") as h:
                        csv.DictWriter(h, fieldnames=list(summary.keys())).writerow(summary)
    write_csv(out / "scale_summary.csv", all_summaries)
    write_csv(out / "scale_episodes.csv", all_episodes)
    print(f"total {time.time() - total_t0:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()