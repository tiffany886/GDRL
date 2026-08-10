"""Temporary probe: does any fixed trajectory residual beat the expert on stress?"""
import numpy as np

from .config import make_config
from .run_experiment import run_policy
from .scenario_spec import scenario_overrides
from .traj_drl import GDRLPolicy, TrajPPOTrainer


class FixedDeltaGDRL(GDRLPolicy):
    def __init__(self, model_path, delta_norm, name):
        super().__init__(model_path=model_path)
        self.name = name
        self._delta = np.asarray(delta_norm, dtype=float)

    def act(self, obs, rng, config):
        if self.trainer is None:
            self.trainer = TrajPPOTrainer.load(self.model_path, config)
        move = self.trainer._compose_move(obs, self._delta)
        post_pos = self.trainer._post_move_pos(obs, move)
        targets, ratios = self.trainer._offload_action(obs, post_pos)
        return {"move": move, "targets": targets, "ratios": ratios}


def main():
    model = "experiments/uav_leo_v2x_paper_final/models/gdrl_stress.pt"
    config = make_config("v2x_hotspot_stress", "none", seed=73, users=16, leos=4,
                         episodes=20, horizon=30, **scenario_overrides("v2x_hotspot_stress"))
    candidates = [
        (0.0, 0.0), (0.2, 0.0), (-0.2, 0.0), (0.0, 0.2), (0.0, -0.2),
        (0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15),
        (0.3, 0.0), (0.3, 0.3), (-0.3, 0.3),
    ]
    for cand in candidates:
        pol = FixedDeltaGDRL(model, cand, "fixed_delta")
        summary, _, _ = run_policy(pol, config)
        print(f"delta_norm={cand}: reward={summary['reward_mean']:.2f} "
              f"success={summary['success_rate']:.3f} latency={summary['latency_mean']:.4f} "
              f"energy={summary['total_energy_mean']:.2f}", flush=True)


if __name__ == "__main__":
    main()
