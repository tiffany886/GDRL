"""Probe smarter expert trajectories (blend alpha, user-lookahead) with exact
post-move offloading, to see if a better trajectory widens GDRL's margin."""
import numpy as np

from .config import make_config
from .run_experiment import run_policy
from .scenario_spec import scenario_overrides
from .traj_drl import _exact_offload


class BlendExpertPolicy:
    def __init__(self, alpha=0.5, lookahead=1, user_lookahead=False):
        self.alpha = alpha
        self.lookahead = lookahead
        self.user_lookahead = user_lookahead
        tag = f"a{alpha}_k{lookahead}"
        if user_lookahead:
            tag += "_ul"
        self.name = "blend_" + tag

    def act(self, obs, rng, config):
        pos = np.asarray(obs["user_pos"], dtype=float)
        if self.user_lookahead and obs.get("user_vel") is not None:
            pos = pos + np.asarray(obs["user_vel"], dtype=float) * config.slot_seconds * self.lookahead
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        if workload.sum() > 1.0e-9:
            centroid = np.average(pos, axis=0, weights=workload)
        else:
            centroid = pos.mean(axis=0)
        if obs.get("hotspot_pos") is not None:
            hotspot_next = (obs["hotspot_pos"]
                            + obs.get("hotspot_vel", np.zeros(2)) * config.slot_seconds * self.lookahead)
            target = self.alpha * centroid + (1.0 - self.alpha) * hotspot_next
        else:
            target = centroid
        move = target - np.asarray(obs["uav_pos"], dtype=float)
        norm = float(np.linalg.norm(move))
        if norm > config.uav_speed_max and norm > 1.0e-9:
            move = move / norm * config.uav_speed_max
        move = np.asarray(move, dtype=float)
        post = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
        targets, ratios = _exact_offload(obs, config, post)
        return {"move": move, "targets": targets, "ratios": ratios}


VARIANTS = {
    "stress": {},
    "V2": {"users": 18, "success_deadline_s": 0.5, "hotspot_speed": 24.0, "hotspot_radius": 80.0,
           "bandwidth_hz": 1.0e6, "backhaul_mbps": 16.0},
}


def main():
    from .baselines import default_policies
    cands = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        cands.append(BlendExpertPolicy(alpha=a, lookahead=1))
    cands += [BlendExpertPolicy(alpha=0.5, lookahead=2),
              BlendExpertPolicy(alpha=0.5, lookahead=1, user_lookahead=True),
              BlendExpertPolicy(alpha=0.25, lookahead=1, user_lookahead=True),
              BlendExpertPolicy(alpha=0.5, lookahead=2, user_lookahead=True)]
    for vname, over in VARIANTS.items():
        base = scenario_overrides("v2x_hotspot_stress")
        base.update(over)
        users = base.pop("users", 16)
        config = make_config("v2x_hotspot_stress", "none", seed=73, users=users, leos=4,
                             episodes=10, horizon=30, **base)
        heurs = {p.name: p for p in default_policies()}
        for name in ("predict_tea", "follow_tea", "deadline_tea", "tea_partial"):
            s, _, _ = run_policy(heurs[name], config)
            print(f"{vname} {name}: {s['reward_mean']:.1f}", flush=True)
        best_heur = None
        for name in ("predict_tea", "follow_tea", "deadline_tea", "tea_partial"):
            s, _, _ = run_policy(heurs[name], config)
            if best_heur is None or s["reward_mean"] > best_heur[1]["reward_mean"]:
                best_heur = (name, s)
        print(f"{vname} best_heur: {best_heur[0]} {best_heur[1]['reward_mean']:.1f}", flush=True)
        for pol in cands:
            s, _, _ = run_policy(pol, config)
            print(f"{vname} {pol.name}: {s['reward_mean']:.1f} succ {s['success_rate']:.3f} "
                  f"gap {s['reward_mean'] - best_heur[1]['reward_mean']:+.1f}", flush=True)


if __name__ == "__main__":
    main()
