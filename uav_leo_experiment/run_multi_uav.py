"""Multi-UAV configuration sweep driver.

Scans UAV count x user count x area x hotspot count for the v2x hotspot line
and evaluates the multi-UAV policies from :mod:`multi_uav` (PMEO-M and
baselines).  Geometry follows the "real" scaling line of scale_sweep.py:
area_size / road_half_len / hotspot_radius scale together while UAV/hotspot
speeds stay physical.

Outputs one merged CSV (multi_uav_summary.csv) plus per-config run dirs with
run_config.json, summary CSV and episode rows.
"""
import argparse
import csv
import json
import time
from pathlib import Path

from .config import make_config
from .multi_uav import multi_uav_policies
from .run_experiment import run_policy, write_csv

# (uavs, users, hotspots, area_km) presets
PRESETS = {
    # density scan: ~17 users per UAV at 2 km
    "k2": dict(uavs=2, users=34, hotspots=2, area_km=2.0),
    "k3": dict(uavs=3, users=51, hotspots=3, area_km=2.0),
    "k4": dict(uavs=4, users=68, hotspots=4, area_km=2.0),
    # user-scale scan at K=3, 2 km
    "u30": dict(uavs=3, users=30, hotspots=3, area_km=2.0),
    "u50": dict(uavs=3, users=50, hotspots=3, area_km=2.0),
    "u80": dict(uavs=3, users=80, hotspots=3, area_km=2.0),
    # area scan at K=3, 50 users
    "a1": dict(uavs=3, users=50, hotspots=3, area_km=1.0),
    "a15": dict(uavs=3, users=50, hotspots=3, area_km=1.5),
    "a2": dict(uavs=3, users=50, hotspots=3, area_km=2.0),
    "a3": dict(uavs=3, users=50, hotspots=3, area_km=3.0),
    # extra combos
    "k2u50": dict(uavs=2, users=50, hotspots=2, area_km=2.0),
    "k4u50": dict(uavs=4, users=50, hotspots=4, area_km=2.0),
    "k3u80a3": dict(uavs=3, users=80, hotspots=3, area_km=3.0),
    "k2u20a15": dict(uavs=2, users=20, hotspots=2, area_km=1.5),
}

BASE_GEOM = {
    "area_size": 1000.0,
    "road_half_len": 350.0,
    "hotspot_radius": 100.0,
    "hotspot_speed": 18.0,
    "uav_speed_max": 25.0,
    "user_speed_max": 16.0,
}


def build_config(preset, seed, episodes):
    p = PRESETS[preset]
    s = p["area_km"]
    cfg = make_config(
        "v2x_hotspot_hard", ablation="none", seed=seed,
        uavs=p["uavs"], users=p["users"], hotspots=p["hotspots"],
        leos=4, episodes=episodes,
        area_size=BASE_GEOM["area_size"] * s,
        road_half_len=BASE_GEOM["road_half_len"] * s,
        hotspot_radius=BASE_GEOM["hotspot_radius"] * s,
        hotspot_speed=BASE_GEOM["hotspot_speed"],
        uav_speed_max=BASE_GEOM["uav_speed_max"],
        user_speed_max=BASE_GEOM["user_speed_max"],
    )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="k2,k3,k4,u30,u50,u80,a1,a15,a2,a3,k2u50,k4u50,k3u80a3,k2u20a15")
    ap.add_argument("--seeds", default="73,1,7,42,2024")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--methods", default="pmeo_m,current_exact_m,follow_tea_m,random_m")
    ap.add_argument("--output_dir", default="experiments/uav_leo_v2x_paper_final/multi_uav")
    args = ap.parse_args()

    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    methods = {p.name: p for p in multi_uav_policies()}
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    missing = [m for m in wanted if m not in methods]
    if missing:
        raise SystemExit("unknown methods: %s (available: %s)" % (missing, sorted(methods)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    all_episodes = []
    total_t0 = time.time()
    for preset in presets:
        for seed in seeds:
            cfg = build_config(preset, seed, args.episodes)
            p = PRESETS[preset]
            tag = f"{preset}/seed{seed}"
            run_dir = out / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(cfg.__dict__, indent=2, default=str), encoding="utf-8")
            for pol_name in wanted:
                pol = methods[pol_name]
                t0 = time.time()
                summary, episodes, _ = run_policy(pol, cfg)
                summary.update({
                    "preset": preset,
                    "uavs": p["uavs"], "users": p["users"], "hotspots": p["hotspots"],
                    "area_km": p["area_km"], "seed": seed,
                    "runtime_s": round(time.time() - t0, 1),
                })
                all_summaries.append(summary)
                for e in episodes:
                    e.update({"preset": preset, "uavs": p["uavs"], "users": p["users"],
                              "hotspots": p["hotspots"], "area_km": p["area_km"], "seed": seed})
                    all_episodes.append(e)
                print(f"[{tag}] {pol_name}: reward={summary['reward_mean']:.3f} "
                      f"success={summary['success_rate']:.3f} "
                      f"energy={summary['total_energy_mean']:.1f} "
                      f"({summary['runtime_s']}s)", flush=True)
    write_csv(out / "multi_uav_summary.csv", all_summaries)
    write_csv(out / "multi_uav_episodes.csv", all_episodes)
    print("total time", round(time.time() - total_t0, 1), "s")
    print("summary:", out / "multi_uav_summary.csv")


if __name__ == "__main__":
    main()
