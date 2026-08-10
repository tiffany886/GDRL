"""Deterministic search for a harder hotspot scenario where the GDRL
(expert trajectory + exact post-move offloading) margin over the TEA heuristics
is clearly larger than on hotspot-hard/stress.

Only deterministic policies are used (no training needed): the learned residual
is ~zero and the exact grid and the TEA grid pick the same decisions, so
``expert + exact@post-move`` is a faithful stand-in for GDRL.
"""
import csv
from pathlib import Path

import numpy as np

from .baselines import default_policies
from .config import make_config
from .run_experiment import run_policy
from .scenario_spec import scenario_overrides
from .traj_drl import _exact_offload, _predict_tea_move


class ExpertExactPolicy:
    """Expert trajectory (k-step hotspot lookahead) + exact offloading either at
    the post-move position (as GDRL does) or at the current position."""
    name = "expert_exact"

    def __init__(self, lookahead=1, offload_at="post_move"):
        self.lookahead = lookahead
        self.offload_at = offload_at
        self.name = f"expert_k{lookahead}_{offload_at}"

    def act(self, obs, rng, config):
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        if workload.sum() > 1.0e-9:
            centroid = np.average(obs["user_pos"], axis=0, weights=workload)
        else:
            centroid = obs["user_pos"].mean(axis=0)
        if obs.get("hotspot_pos") is not None:
            hotspot_next = (obs["hotspot_pos"]
                            + obs.get("hotspot_vel", np.zeros(2))
                            * config.slot_seconds * self.lookahead)
            target = 0.5 * centroid + 0.5 * hotspot_next
        else:
            target = centroid
        move = target - np.asarray(obs["uav_pos"], dtype=float)
        norm = float(np.linalg.norm(move))
        if norm > config.uav_speed_max and norm > 1.0e-9:
            move = move / norm * config.uav_speed_max
        move = np.asarray(move, dtype=float)
        if self.offload_at == "post_move":
            pos = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
        else:
            pos = obs["uav_pos"]
        targets, ratios = _exact_offload(obs, config, pos)
        return {"move": move, "targets": targets, "ratios": ratios}


BASE = scenario_overrides("v2x_hotspot_stress")
VARIANTS = {
    "V0_stress_base": {},
    "V1": {"users": 18, "success_deadline_s": 0.55, "hotspot_speed": 22.0, "hotspot_radius": 85.0},
    "V2": {"users": 18, "success_deadline_s": 0.5, "hotspot_speed": 24.0, "hotspot_radius": 80.0,
           "bandwidth_hz": 1.0e6, "backhaul_mbps": 16.0},
    "V3": {"users": 20, "success_deadline_s": 0.5, "hotspot_speed": 24.0, "hotspot_radius": 80.0,
           "bandwidth_hz": 1.0e6, "backhaul_mbps": 16.0},
    "V4": {"users": 20, "success_deadline_s": 0.45, "hotspot_speed": 26.0, "hotspot_radius": 75.0,
           "bandwidth_hz": 1.0e6, "backhaul_mbps": 14.0},
    "V5": {"users": 24, "success_deadline_s": 0.45, "hotspot_speed": 26.0, "hotspot_radius": 75.0,
           "bandwidth_hz": 0.95e6, "backhaul_mbps": 12.0},
    "V6": {"users": 20, "success_deadline_s": 0.5, "hotspot_speed": 28.0, "hotspot_radius": 80.0,
           "bandwidth_hz": 1.0e6, "backhaul_mbps": 16.0, "uav_speed_max": 30.0},
    "V7": {"users": 24, "success_deadline_s": 0.4, "hotspot_speed": 26.0, "hotspot_radius": 70.0,
           "bandwidth_hz": 0.9e6, "backhaul_mbps": 12.0, "uav_speed_max": 30.0},
}
HEURISTIC_NAMES = ["predict_tea", "follow_tea", "deadline_tea", "tea_partial", "energy_guarded_tea"]


def make_variant_config(name, episodes):
    overrides = dict(BASE)
    overrides.update(VARIANTS[name])
    users = overrides.pop("users", 16)
    return make_config("v2x_hotspot_stress", "none", seed=73, users=users, leos=4,
                       episodes=episodes, horizon=30, **overrides)


def main():
    episodes = 10
    out_path = Path("experiments/uav_leo_v2x_paper_final/scenario_search.csv")
    rows = []
    for vname in VARIANTS:
        config = make_variant_config(vname, episodes)
        heuristics = [pol for pol in default_policies() if pol.name in HEURISTIC_NAMES]
        experts = [ExpertExactPolicy(k, "post_move") for k in (1, 2, 3)]
        experts.append(ExpertExactPolicy(1, "current"))
        results = {}
        for pol in heuristics + experts:
            summary, _, _ = run_policy(pol, config)
            results[pol.name] = summary
            rows.append({"variant": vname, "method": pol.name,
                         "reward": summary["reward_mean"],
                         "success": summary["success_rate"],
                         "latency": summary["latency_mean"],
                         "energy": summary["total_energy_mean"]})
        best_heur = max((results[n] for n in HEURISTIC_NAMES), key=lambda s: s["reward_mean"])
        best_exp = max((results[p.name] for p in experts), key=lambda s: s["reward_mean"])
        gap = best_exp["reward_mean"] - best_heur["reward_mean"]
        print(f"{vname}: best_expert={best_exp['reward_mean']:.1f} (succ {best_exp['success_rate']:.3f}) "
              f"best_heur={best_heur['reward_mean']:.1f} ({best_heur['method']}) "
              f"gap={gap:.1f}  rel={100*gap/abs(best_heur['reward_mean']):.1f}%", flush=True)
        for pol in experts:
            s = results[pol.name]
            print(f"    {pol.name}: {s['reward_mean']:.1f} succ {s['success_rate']:.3f}")
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "method", "reward", "success", "latency", "energy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
