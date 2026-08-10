"""Run UAV-LEO experiment baselines + optional trained DRL policies."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .algorithms import (AntColonyPolicy, ExhaustiveOptimalPolicy, GeneticPolicy,
                           LyapunovPolicy, PSOPolicy, SimulatedAnnealingPolicy)
from .baselines import default_policies
from .config import ABLATION_PRESETS, DIFFICULTY_PRESETS, make_config
from .env import UavLeoEnv
from .scenario_spec import scenario_overrides
from .learning import (D3QNPolicy, DDPGPolicy, DDQNPolicy, DQNPolicy, GraphPPOPolicy,
                           PPOPolicy, SACPolicy, TD3Policy)
from .traj_drl import GDRLPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Run standalone UAV-LEO offloading baselines.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="easy")
    parser.add_argument("--ablation", choices=sorted(ABLATION_PRESETS), default="none")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--leos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--task_bits_min", type=float, default=None)
    parser.add_argument("--task_bits_max", type=float, default=None)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--bandwidth_hz", type=float, default=None)
    parser.add_argument("--energy_weight", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--ppo_model", type=str, default=None, help="Path to trained PPO model.pt")
    parser.add_argument("--dqn_model", type=str, default=None, help="Path to trained DQN model.pt")
    parser.add_argument("--ddqn_model", type=str, default=None, help="Path to trained DDQN model.pt")
    parser.add_argument("--d3qn_model", type=str, default=None, help="Path to trained P-D3QN model.pt")
    parser.add_argument("--gat_ppo_model", type=str, default=None, help="Path to trained GAT-PPO model.pt")
    parser.add_argument("--transformer_ppo_model", type=str, default=None,
                        help="Path to trained Transformer-PPO model.pt")
    parser.add_argument("--td3_model", type=str, default=None, help="Path to trained TD3 model.pt")
    parser.add_argument("--ddpg_model", type=str, default=None, help="Path to trained DDPG model.pt")
    parser.add_argument("--sac_model", type=str, default=None, help="Path to trained SAC model.pt")
    parser.add_argument("--gdrl_model", type=str, default=None, help="Path to trained GDRL trajectory model.pt")
    parser.add_argument("--gdrl_ablate_traj", action="store_true",
                        help="GDRL ablation: learned offloading, frozen expert trajectory (delta=0).")
    parser.add_argument("--gdrl_ablate_offload", action="store_true",
                        help="GDRL ablation: learned trajectory, TEA heuristic offloading.")
    parser.add_argument("--mpc_horizon", type=int, default=0,
                        help="Receding-horizon (MPC) trajectory planner lookahead slots (0 = disabled).")
    parser.add_argument("--mpc_offload", type=str, default="exact", choices=["exact", "tea"],
                        help="Offloading layer used inside the MPC trajectory planner.")
    parser.add_argument("--skip_slow", action="store_true", help="Skip genetic/exhaustive policies")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated policy names to run (default: all)")
    return parser.parse_args()


def build_config(args):
    cli = {
        "seed": args.seed,
        "users": args.users,
        "leos": args.leos,
        "horizon": args.horizon,
        "episodes": args.episodes,
        "task_bits_min": args.task_bits_min,
        "task_bits_max": args.task_bits_max,
        "success_deadline_s": args.deadline,
        "bandwidth_hz": args.bandwidth_hz,
        "energy_weight": args.energy_weight,
    }
    overrides = {key: value for key, value in cli.items() if value is not None}
    overrides = {**scenario_overrides(args.difficulty), **overrides}
    return make_config(args.difficulty, args.ablation, **overrides)


def build_policies(args):
    policies = default_policies() + [LyapunovPolicy()]
    if getattr(args, "mpc_horizon", 0) and args.mpc_horizon > 0:
        from .mpc_traj import MPCTrajPolicy
        policies.append(MPCTrajPolicy(horizon=args.mpc_horizon, offload=args.mpc_offload))
    if not args.skip_slow:
        policies += [GeneticPolicy(pop_size=40, generations=10), PSOPolicy(),
                     AntColonyPolicy(), SimulatedAnnealingPolicy(),
                     ExhaustiveOptimalPolicy()]
    if args.ppo_model:
        policies.append(PPOPolicy(model_path=args.ppo_model))
    if args.dqn_model:
        policies.append(DQNPolicy(model_path=args.dqn_model))
    if getattr(args, "ddqn_model", None):
        policies.append(DDQNPolicy(model_path=args.ddqn_model))
    if getattr(args, "d3qn_model", None):
        policies.append(D3QNPolicy(model_path=args.d3qn_model))
    if getattr(args, "gat_ppo_model", None):
        policies.append(GraphPPOPolicy(model_path=args.gat_ppo_model, name="gat_ppo"))
    if getattr(args, "transformer_ppo_model", None):
        policies.append(GraphPPOPolicy(model_path=args.transformer_ppo_model, name="transformer_ppo"))
    if getattr(args, "td3_model", None):
        policies.append(TD3Policy(model_path=args.td3_model))
    if getattr(args, "ddpg_model", None):
        policies.append(DDPGPolicy(model_path=args.ddpg_model))
    if getattr(args, "sac_model", None):
        policies.append(SACPolicy(model_path=args.sac_model))
    if getattr(args, "gdrl_model", None):
        policies.append(GDRLPolicy(model_path=args.gdrl_model))
        if getattr(args, "gdrl_ablate_traj", False):
            policies.append(GDRLPolicy(model_path=args.gdrl_model, ablate_traj=True))
        if getattr(args, "gdrl_ablate_offload", False):
            policies.append(GDRLPolicy(model_path=args.gdrl_model, ablate_offload=True))
    if getattr(args, "methods", None):
        allowed = {name.strip() for name in args.methods.split(",") if name.strip()}
        policies = [pol for pol in policies if pol.name in allowed]
    return policies


def run_policy(policy, config):
    rng = np.random.default_rng(config.seed)
    env = UavLeoEnv(config)
    episode_rows = []
    step_rows = []
    target_counter = Counter()

    for episode in range(1, config.episodes + 1):
        obs = env.reset(seed=config.seed + episode)
        ep_reward = 0.0
        ep_latency = 0.0
        ep_task_energy = 0.0
        ep_total_energy = 0.0
        ep_flight = 0.0
        ep_offload_ratio = []
        ep_success = []
        ep_drop = []
        done = False
        step = 0
        while not done:
            action = policy.act(obs, rng, config)
            obs, reward, done, info = env.step(action)
            step += 1
            ep_reward += reward
            ep_latency += info["latency_mean"]
            ep_task_energy += info["task_energy"]
            ep_total_energy += info["total_energy"]
            ep_flight += info["flight_energy"]
            ep_offload_ratio.append(info["offload_ratio_mean"])
            ep_success.append(info["success_rate"])
            ep_drop.append(info.get("drop_rate", 0.0))
            target_counter.update(info["target_names"])
            step_rows.append({
                "difficulty": config.difficulty,
                "ablation": config.ablation,
                "method": policy.name,
                "episode": episode,
                "step": step,
                "reward": reward,
                "latency_mean": info["latency_mean"],
                "task_energy": info["task_energy"],
                "flight_energy": info["flight_energy"],
                "total_energy": info["total_energy"],
                "success_rate": info["success_rate"],
                "drop_rate": info.get("drop_rate", 0.0),
                "offload_ratio_mean": info["offload_ratio_mean"],
                "uav_pos_x": info["uav_pos_x"],
                "uav_pos_y": info["uav_pos_y"],
            })
        episode_rows.append({
            "difficulty": config.difficulty,
            "ablation": config.ablation,
            "method": policy.name,
            "episode": episode,
            "reward": ep_reward,
            "latency_mean": ep_latency / config.horizon,
            "task_energy_mean": ep_task_energy / config.horizon,
            "flight_energy_mean": ep_flight / config.horizon,
            "total_energy_mean": ep_total_energy / config.horizon,
            "success_rate": float(np.mean(ep_success)),
            "drop_rate": float(np.mean(ep_drop)),
            "offload_ratio_mean": float(np.mean(ep_offload_ratio)),
        })

    rewards = np.array([row["reward"] for row in episode_rows], dtype=float)
    latencies = np.array([row["latency_mean"] for row in episode_rows], dtype=float)
    task_energies = np.array([row["task_energy_mean"] for row in episode_rows], dtype=float)
    flight_energies = np.array([row["flight_energy_mean"] for row in episode_rows], dtype=float)
    total_energies = np.array([row["total_energy_mean"] for row in episode_rows], dtype=float)
    successes = np.array([row["success_rate"] for row in episode_rows], dtype=float)
    drops = np.array([row["drop_rate"] for row in episode_rows], dtype=float)
    offload_ratios = np.array([row["offload_ratio_mean"] for row in episode_rows], dtype=float)
    target_total = max(sum(target_counter.values()), 1)
    summary = {
        "difficulty": config.difficulty,
        "ablation": config.ablation,
        "method": policy.name,
        "episodes": config.episodes,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "latency_mean": float(latencies.mean()),
        "task_energy_mean": float(task_energies.mean()),
        "flight_energy_mean": float(flight_energies.mean()),
        "total_energy_mean": float(total_energies.mean()),
        "success_rate": float(successes.mean()),
        "drop_rate": float(drops.mean()),
        "offload_ratio_mean": float(offload_ratios.mean()),
        "target_local": target_counter.get("local", 0),
        "target_uav": target_counter.get("uav", 0),
        "target_leo": target_counter.get("leo", 0),
        "target_local_rate": target_counter.get("local", 0) / target_total,
        "target_uav_rate": target_counter.get("uav", 0) / target_total,
        "target_leo_rate": target_counter.get("leo", 0) / target_total,
    }
    return summary, episode_rows, step_rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config, output_dir, policies=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="ascii")

    summaries = []
    all_episodes = []
    all_steps = []
    for policy in policies:
        print(f"Running {config.difficulty}/{config.ablation}/{policy.name} ...", flush=True)
        summary, episodes, steps = run_policy(policy, config)
        summaries.append(summary)
        all_episodes.extend(episodes)
        all_steps.extend(steps)
        print(
            f"{policy.name}: reward={summary['reward_mean']:.3f} "
            f"latency={summary['latency_mean']:.6f} "
            f"energy={summary['total_energy_mean']:.3f} "
            f"success={summary['success_rate']:.3f} drop={summary['drop_rate']:.3f} "
            f"offload={summary['offload_ratio_mean']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "uav_leo_summary.csv", summaries)
    write_csv(output_dir / "uav_leo_episodes.csv", all_episodes)
    write_csv(output_dir / "uav_leo_steps.csv", all_steps)
    print(f"Saved results to {output_dir}", flush=True)
    return summaries, all_episodes, all_steps


def default_output_dir(config, output_root=None):
    root = Path(output_root or config.output_root)
    return root / config.difficulty / config.ablation / f"U{config.users}_L{config.leos}_{config.episodes}ep_T{config.horizon}"


def main():
    args = parse_args()
    config = build_config(args)
    output_dir = default_output_dir(config, args.output_dir)
    policies = build_policies(args)
    run_experiment(config, output_dir, policies)


if __name__ == "__main__":
    main()