"""Train the VA-DAG-HO MAPPO trajectory policy (W3.5) or the flat B4
baseline (W3.6) and log training curves to CSV."""
from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path

import numpy as np
import torch

from ..agents.mappo import Mappo
from ..agents.trainer import train
from ..env.config import DEFAULT_CONFIG_PATH, load_config
from ..env.env import DisasterEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="embed", choices=["embed", "flat"])
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--out", default="")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    args = ap.parse_args()

    cfg = load_config(DEFAULT_CONFIG_PATH, overrides={
        "seed": args.seed,
        "obs_terminal_features": args.mode == "flat",
    })
    env = DisasterEnv(cfg)
    num_slots = ceil(cfg.num_terminals / cfg.num_uavs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Mappo(
        obs_dim=env.obs_dim, num_uavs=cfg.num_uavs, num_slots=num_slots,
        mode=args.mode, seed=args.seed, lr=args.lr,
        entropy_coef=args.entropy_coef, device=device,
    )
    eval_seeds = tuple(int(s) for s in args.eval_seeds.split(",") if s.strip())
    rows = train(policy, env, episodes=args.episodes, seed=args.seed,
                 eval_every=args.eval_every, eval_seeds=eval_seeds)

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent
        / "results" / f"train_{args.mode}_seed{args.seed}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")

    tr = [r for r in rows if r["kind"] == "train"]
    ev = [r for r in rows if r["kind"] == "eval"]
    if tr:
        head = np.mean([r["reward"] for r in tr[: max(1, len(tr) // 10)]])
        tail = np.mean([r["reward"] for r in tr[-max(1, len(tr) // 10):]])
        print(f"reward head {head:.2f} -> tail {tail:.2f}")
    if ev:
        first, last = ev[: len(eval_seeds)], ev[-len(eval_seeds):]
        f_s = np.mean([r["success_rate"] for r in first])
        l_s = np.mean([r["success_rate"] for r in last])
        print(f"eval success first {f_s:.3f} -> last {l_s:.3f} "
              f"(n_eval={len(eval_seeds)})")


if __name__ == "__main__":
    main()

