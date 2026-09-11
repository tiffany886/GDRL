
"""Train decentralized DRL baselines (m_dqn / m_td3 / m_sac) on a multi-UAV preset."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
from docs.GDRL.uav_leo_experiment.run_multi_uav import build_config
from docs.GDRL.uav_leo_experiment.env import UavLeoEnv
from docs.GDRL.uav_leo_experiment.multi_uav_drl import (train_multi_uav_drl, evaluate_multi_uav,
                                              make_trainer)
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="k3")
    ap.add_argument("--method", default="m_td3", choices=["m_dqn", "m_td3", "m_sac"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--output_dir", default="experiments/uav_leo_v2x_paper_final/mdrl_models")
    args = ap.parse_args()

    cfg = build_config(args.preset, args.seed, episodes=10)
    eval_env = UavLeoEnv(cfg)
    out = Path(args.output_dir) / f"{args.preset}_{args.method}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    trainer = train_multi_uav_drl(
        args.method, cfg, steps=args.steps, eval_env=eval_env,
        eval_every=args.eval_every, log_path=out, seed=args.seed)
    # final evaluation
    score = evaluate_multi_uav(trainer, eval_env, cfg, episodes=10, seed=args.seed + 999)
    print(f"FINAL {args.method} seed{args.seed}: eval_reward={score:.1f} "
          f"({time.time()-t0:.1f}s) -> {out}")

if __name__ == "__main__":
    main()
