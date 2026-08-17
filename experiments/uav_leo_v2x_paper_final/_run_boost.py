
"""k3: eco H5+future enhancement and deadline stress scenario."""
import argparse, json, time
from pathlib import Path
from dataclasses import replace
from uav_leo_experiment.run_multi_uav import build_config
from uav_leo_experiment.run_experiment import run_policy, write_csv
from uav_leo_experiment.multi_uav import (PmeoMPolicy, PmeoMEcoPolicy, MpcMPolicy)

def _eco_f5():
    pol = PmeoMEcoPolicy(horizon=5, include_future=True)
    pol.name = "pmeo_m_eco_f5"
    return pol

POLS = {
    "pmeo_m":      PmeoMPolicy(),
    "pmeo_m_eco":  PmeoMEcoPolicy(horizon=3),
    "pmeo_m_eco_f5": _eco_f5(),
    "mpc_m_h3":    MpcMPolicy(),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="k3")
    ap.add_argument("--seeds", default="1,7,42,73,314,555,888,999,12345,2024")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--methods", default="pmeo_m,pmeo_m_eco,pmeo_m_eco_f5,mpc_m_h3")
    ap.add_argument("--deadlines", default="0.65,0.5")
    ap.add_argument("--output_dir", default="experiments/uav_leo_v2x_paper_final/multi_uav_boost")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    missing = [m for m in wanted if m not in POLS]
    if missing:
        raise SystemExit(f"unknown methods: {missing}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_summaries, all_episodes = [], []
    t0_all = time.time()
    for dl in [float(x) for x in args.deadlines.split(",")]:
        for seed in seeds:
            base = build_config(args.preset, seed, args.episodes)
            cfg = replace(base, success_deadline_s=dl)
            tag = f"dl{dl:g}/seed{seed}"
            run_dir = out / tag; run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(
                json.dumps(cfg.__dict__, indent=2, default=str), encoding="utf-8")
            for pol_name in wanted:
                pol = POLS[pol_name]
                t0 = time.time()
                summary, episodes, _ = run_policy(pol, cfg)
                summary.update({"preset": args.preset, "success_deadline_s": dl,
                                "seed": seed, "runtime_s": round(time.time() - t0, 1)})
                all_summaries.append(summary)
                for e in episodes:
                    e.update({"preset": args.preset, "success_deadline_s": dl, "seed": seed})
                    all_episodes.append(e)
                print(f"[{tag}] {pol_name}: reward={summary['reward_mean']:.3f} "
                      f"succ={summary['success_rate']:.3f} en={summary['total_energy_mean']:.1f} "
                      f"lat={summary['latency_mean']:.4f} ({summary['runtime_s']}s)", flush=True)
    write_csv(out / "multi_uav_summary.csv", all_summaries)
    write_csv(out / "multi_uav_episodes.csv", all_episodes)
    print("total", round(time.time() - t0_all, 1), "s ->", out / "multi_uav_summary.csv")

if __name__ == "__main__":
    main()
