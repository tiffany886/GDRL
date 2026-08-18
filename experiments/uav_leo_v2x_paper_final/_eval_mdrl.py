# -*- coding: utf-8 -*-
"""Evaluate trained multi-UAV DRL baselines on the k3 10-seed x 10-episode protocol.

Usage:
  python _eval_mdrl.py --method m_td3 --model_dir experiments/uav_leo_v2x_paper_final/mdrl_models/k3_m_td3_seed1
"""
import argparse, csv, time
from pathlib import Path

import numpy as np
import torch

from uav_leo_experiment.run_multi_uav import build_config
from uav_leo_experiment.env import UavLeoEnv
from uav_leo_experiment.multi_uav_drl import make_trainer, act_global, lmax_for
from uav_leo_experiment.multi_uav import associate_users

SEEDS = [1, 7, 42, 73, 314, 555, 888, 999, 12345, 2024]


def load_trainer(method, config, model_path):
    trainer = make_trainer(method, config)
    ckpt = torch.load(model_path, map_location="cpu")
    if method == "m_dqn":
        trainer.q_net.load_state_dict(ckpt["state_dict"])
    else:
        trainer.actor.load_state_dict(ckpt["state_dict"])
    return trainer


def evaluate_seed(trainer, cfg, seed, episodes=10):
    env = UavLeoEnv(cfg)
    lmax = lmax_for(cfg)
    rows = []
    for ep in range(1, episodes + 1):
        obs = env.reset(seed=cfg.seed + ep)
        ep_reward, ep_lat, ep_task, ep_flight, ep_total = 0.0, 0.0, 0.0, 0.0, 0.0
        succs, drops, offs = [], [], []
        done = False
        while not done:
            assoc = associate_users(obs, cfg)
            action, _, _ = act_global(trainer, obs, cfg, assoc, lmax, train=False)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_lat += info["latency_mean"]
            ep_task += info["task_energy"]
            ep_total += info["total_energy"]
            ep_flight += info["flight_energy"]
            succs.append(info["success_rate"])
            drops.append(info.get("drop_rate", 0.0))
            offs.append(info["offload_ratio_mean"])
        rows.append({
            "episode": ep, "reward": ep_reward,
            "latency_mean": ep_lat / cfg.horizon,
            "task_energy_mean": ep_task / cfg.horizon,
            "flight_energy_mean": ep_flight / cfg.horizon,
            "total_energy_mean": ep_total / cfg.horizon,
            "success_rate": float(np.mean(succs)),
            "drop_rate": float(np.mean(drops)),
            "offload_ratio_mean": float(np.mean(offs)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["m_dqn", "m_td3", "m_sac"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--preset", default="k3")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    model_dir = Path(args.model_dir)
    model_path = model_dir / "model_best.pt"
    cfg = build_config(args.preset, 1, args.episodes)  # seed field will be overwritten per seed
    trainer = load_trainer(args.method, cfg, model_path)
    out_csv = args.out_csv or (model_dir / "eval_summary.csv")
    out_rows = []
    t0 = time.time()
    for seed in seeds:
        cfg_s = build_config(args.preset, seed, args.episodes)
        ep_rows = evaluate_seed(trainer, cfg_s, seed, args.episodes)
        r = np.array([x["reward"] for x in ep_rows])
        lat = np.array([x["latency_mean"] for x in ep_rows])
        te = np.array([x["task_energy_mean"] for x in ep_rows])
        fe = np.array([x["flight_energy_mean"] for x in ep_rows])
        tot = np.array([x["total_energy_mean"] for x in ep_rows])
        succ = np.array([x["success_rate"] for x in ep_rows])
        drop = np.array([x["drop_rate"] for x in ep_rows])
        off = np.array([x["offload_ratio_mean"] for x in ep_rows])
        out_rows.append({
            "difficulty": cfg_s.difficulty, "ablation": cfg_s.ablation,
            "method": args.method, "episodes": args.episodes,
            "reward_mean": float(r.mean()), "reward_std": float(r.std()),
            "latency_mean": float(lat.mean()),
            "task_energy_mean": float(te.mean()), "flight_energy_mean": float(fe.mean()),
            "total_energy_mean": float(tot.mean()),
            "success_rate": float(succ.mean()), "drop_rate": float(drop.mean()),
            "offload_ratio_mean": float(off.mean()),
            "target_local": 0, "target_uav": 0, "target_leo": 0,
            "target_local_rate": 0.0, "target_uav_rate": 0.0, "target_leo_rate": 0.0,
            "preset": args.preset, "seed": seed,
            "runtime_s": round((time.time() - t0) / len(seeds), 1),
        })
        print(f"seed {seed}: reward={r.mean():.1f} succ={succ.mean():.3f} "
              f"en={tot.mean():.1f} lat={lat.mean():.4f}", flush=True)
    fields = list(out_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
