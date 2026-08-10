"""Probe: does speed budgeting help when the UAV battery binds?

If full-speed chasing drains the battery and grounds the UAV for the last
slots, a battery-aware trajectory (lower speed cap or adaptive speed) should
beat the myopic full-speed expert.  This would give the learned trajectory
layer a real, physically meaningful job and could widen GDRL's margin.
"""
import numpy as np

from .config import make_config
from .run_experiment import run_policy
from .scenario_spec import scenario_overrides
from .traj_drl import _exact_offload, _predict_tea_move


class SpeedCappedExpert:
    """Expert trajectory with a fixed speed cap (fraction of uav_speed_max)."""

    def __init__(self, cap=1.0, adaptive=False):
        self.cap = cap
        self.adaptive = adaptive
        self.name = f"expert_cap{cap}" if not adaptive else "expert_adaptive"

    def act(self, obs, rng, config):
        move = _predict_tea_move(obs, config)
        speed_max = config.uav_speed_max
        if self.adaptive:
            frac = 1.0 if float(obs.get("uav_battery_norm", 1.0)) > 0.35 else 0.6
            cap = frac * speed_max
        else:
            cap = self.cap * speed_max
        norm = float(np.linalg.norm(move))
        if norm > cap and norm > 1.0e-9:
            move = move / norm * cap
        move = np.asarray(move, dtype=float)
        post = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
        targets, ratios = _exact_offload(obs, config, post)
        return {"move": move, "targets": targets, "ratios": ratios}


VARIANTS = {
    "stress_battery150": {},
    "stress_battery100": {"uav_battery_per_slot": 100.0},
    "stress_battery80": {"uav_battery_per_slot": 80.0},
}


def main():
    episodes = 20
    base = scenario_overrides("v2x_hotspot_stress")
    for vname, over in VARIANTS.items():
        cfg = dict(base)
        cfg.update(over)
        users = cfg.pop("users", 16)
        config = make_config("v2x_hotspot_stress", "none", seed=73, users=users, leos=4,
                             episodes=episodes, horizon=30, **cfg)
        policies = ([SpeedCappedExpert(1.0)] +
                    [SpeedCappedExpert(c) for c in (0.9, 0.8, 0.7, 0.6, 0.5)] +
                    [SpeedCappedExpert(0.0), SpeedCappedExpert(adaptive=True)])
        print(f"=== {vname} (battery={cfg.get('uav_battery_per_slot', 150)}*30) ===", flush=True)
        for pol in policies:
            s, _, _ = run_policy(pol, config)
            print(f"  {pol.name}: reward={s['reward_mean']:.1f} success={s['success_rate']:.3f} "
                  f"latency={s['latency_mean']:.4f} flight={s['flight_energy_mean']:.1f} "
                  f"drop={s['drop_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
