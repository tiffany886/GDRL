"""Per-episode training curves for the paper (one seed per method).

Writes results/train_curves/*.csv consumed by scripts/make_figures.py:
  embed_main_seed100.csv  (本文, 200 ep)
  b4_flat_seed100.csv     (B4 flat MAPPO, 250 ep)
  b5_sac_seed100.csv      (B5 SAC-flat, 400 ep)
"""
from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path

import torch

from ..agents.mappo import Mappo
from ..agents.trainer import train
from ..baselines.sac_flat import SacFlat
from ..env.config import load_config_variant
from ..env.env import DisasterEnv

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "train_curves"


def _episodes_for(name: str) -> int:
    return {"embed": 200, "flat": 250, "sac": 400}[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    eval_seeds = tuple(range(5))
    jobs = [
        ("embed", None, "embed", "VA-DAG-HO-MAPPO"),
        ("flat", "b4_flat.yaml", "flat", "B4-flat"),
    ]
    for name, delta, mode, _label in jobs:
        env = DisasterEnv(load_config_variant(delta, overrides={
            "seed": args.seed}))
        slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
        policy = Mappo(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                       num_slots=slots, mode=mode, seed=args.seed,
                       device=device)
        rows = train(policy, env, episodes=_episodes_for(name),
                     seed=args.seed, eval_every=25, eval_seeds=eval_seeds)
        _dump(OUT / f"{name}_seed{args.seed}.csv", rows)
        print(f"{name}: {len(rows)} rows")

    # B5 SAC-flat curve
    name = "sac"
    env = DisasterEnv(load_config_variant("b5_flat.yaml", overrides={
        "seed": args.seed}))
    slots = ceil(env.cfg.num_terminals / env.cfg.num_uavs)
    policy = SacFlat(obs_dim=env.obs_dim, num_uavs=env.cfg.num_uavs,
                     num_slots=slots, seed=args.seed, device=device,
                     warmup=400, batch=256, alpha=0.2)
    rows = policy.fit(env, episodes=_episodes_for(name), eval_every=50,
                      eval_seeds=eval_seeds)
    _dump(OUT / f"{name}_seed{args.seed}.csv", rows)
    print(f"{name}: {len(rows)} rows")


def _dump(path: Path, rows):
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
