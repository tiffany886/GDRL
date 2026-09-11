
"""k3 drop-penalty sensitivity: does PMEO-M-Eco's advantage widen when drops are expensive?"""
import argparse, json, time
from pathlib import Path
from docs.GDRL.uav_leo_experiment.run_multi_uav import build_config, PRESETS
from docs.GDRL.uav_leo_experiment.run_experiment import run_policy, write_csv
from docs.GDRL.uav_leo_experiment.config import make_config
from docs.GDRL.uav_leo_experiment.multi_uav import multi_uav_policies
from dataclasses import replace

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="k3")
    ap.add_argument("--seeds", default="1,7,42,73,314,555,888,999,12345,2024")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--methods", default="pmeo_m,pmeo_m_eco,mpc_m_h3,pmeo_m,current_exact_m,follow_tea_m,random_m")
    ap.add_argument("--drop_penalties", default="6,30,60")
    ap.add_argument("--output_dir", default="experiments/uav_leo_v2x_paper_final/multi_uav_dropsweep")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    penalties = [float(x) for x in args.drop_penalties.split(",")]
    methods = {p.name: p for p in multi_uav_policies()}
    wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    all_summaries, all_episodes = [], []
    t0_all = time.time()
    for dp in penalties:
        for seed in seeds:
            base = build_config(args.preset, seed, args.episodes)
            cfg = replace(base, drop_penalty=dp)
            tag = f"dp{dp:g}/seed{seed}"
            run_dir = out / tag; run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str), encoding="utf-8")
            for pol_name in wanted:
                pol = methods[pol_name]
                t0 = time.time()
                summary, episodes, _ = run_policy(pol, cfg)
                summary.update({"preset": args.preset, "drop_penalty": dp, "seed": seed,
                                "runtime_s": round(time.time() - t0, 1)})
                all_summaries.append(summary)
                for e in episodes:
                    e.update({"preset": args.preset, "drop_penalty": dp, "seed": seed})
                    all_episodes.append(e)
                print(f"[{tag}] {pol_name}: reward={summary['reward_mean']:.3f} "
                      f"succ={summary['success_rate']:.3f} en={summary['total_energy_mean']:.1f} "
                      f"({summary['runtime_s']}s)", flush=True)
    write_csv(out / "multi_uav_summary.csv", all_summaries)
    write_csv(out / "multi_uav_episodes.csv", all_episodes)
    print("total", round(time.time() - t0_all, 1), "s ->", out / "multi_uav_summary.csv")

if __name__ == "__main__":
    main()
