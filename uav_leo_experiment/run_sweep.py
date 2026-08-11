"""Run UAV-LEO difficulty sweep with heuristics, optimization baselines, and DRL."""
import argparse
import csv
import json
from pathlib import Path

from .config import ABLATION_PRESETS, DIFFICULTY_PRESETS, make_config
from .run_experiment import build_policies, default_output_dir, run_experiment
from .scenario_spec import scenario_overrides

# User count used by each difficulty when --users is not given. The v2x line
# scales the user population with difficulty so the DRL models (trained at the
# same counts) are evaluated at the matching scale. Hotspot scenarios read the
# authoritative user count from scenario_spec.MY_SCENARIOS so sweep and
# training always use the same population.
from .scenario_spec import MY_SCENARIOS

DIFFICULTY_USERS = {
    "v2x_easy": 5,
    "v2x_medium": 8,
    "v2x_hard": 12,
    "v2x_stress": 16,
}
def users_for(difficulty, default=5):
    return MY_SCENARIOS.get(difficulty, {}).get("users", DIFFICULTY_USERS.get(difficulty, default))


def parse_args():
    parser = argparse.ArgumentParser(description="Run UAV-LEO difficulty and ablation sweep.")
    parser.add_argument("--difficulties", nargs="+",
                        default=["v2x_easy", "v2x_medium", "v2x_hard", "v2x_stress"],
                        choices=sorted(DIFFICULTY_PRESETS))
    parser.add_argument("--ablations", nargs="+", default=["none"], choices=sorted(ABLATION_PRESETS))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--leos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output_dir", type=str, default="experiments/uav_leo_v2x_sweep")
    parser.add_argument("--ppo_easy", type=str, default="experiments/uav_leo_v2x/drl_models/ppo/v2x_easy/model.pt")
    parser.add_argument("--ppo_medium", type=str, default="experiments/uav_leo_v2x/drl_models/ppo/v2x_medium/model.pt")
    parser.add_argument("--ppo_hard", type=str, default="experiments/uav_leo_v2x/drl_models/ppo/v2x_hard/model.pt")
    parser.add_argument("--ppo_stress", type=str, default="experiments/uav_leo_v2x/drl_models/ppo/v2x_stress/model.pt")
    parser.add_argument("--dqn_easy", type=str, default="experiments/uav_leo_v2x/drl_models/dqn/v2x_easy/model.pt")
    parser.add_argument("--dqn_medium", type=str, default="experiments/uav_leo_v2x/drl_models/dqn/v2x_medium/model.pt")
    parser.add_argument("--dqn_hard", type=str, default="experiments/uav_leo_v2x/drl_models/dqn/v2x_hard/model.pt")
    parser.add_argument("--dqn_stress", type=str, default="experiments/uav_leo_v2x/drl_models/dqn/v2x_stress/model.pt")
    parser.add_argument("--ddqn_easy", type=str, default="experiments/uav_leo_v2x/drl_models/ddqn/v2x_easy/model.pt")
    parser.add_argument("--ddqn_medium", type=str, default="experiments/uav_leo_v2x/drl_models/ddqn/v2x_medium/model.pt")
    parser.add_argument("--ddqn_hard", type=str, default="experiments/uav_leo_v2x/drl_models/ddqn/v2x_hard/model.pt")
    parser.add_argument("--ddqn_stress", type=str, default="experiments/uav_leo_v2x/drl_models/ddqn/v2x_stress/model.pt")
    parser.add_argument("--td3_easy", type=str, default="experiments/uav_leo_v2x/drl_models/td3/v2x_easy/model.pt")
    parser.add_argument("--td3_medium", type=str, default="experiments/uav_leo_v2x/drl_models/td3/v2x_medium/model.pt")
    parser.add_argument("--td3_hard", type=str, default="experiments/uav_leo_v2x/drl_models/td3/v2x_hard/model.pt")
    parser.add_argument("--td3_stress", type=str, default="experiments/uav_leo_v2x/drl_models/td3/v2x_stress/model.pt")
    parser.add_argument("--ddpg_easy", type=str, default="experiments/uav_leo_v2x/drl_models/ddpg/v2x_easy/model.pt")
    parser.add_argument("--ddpg_medium", type=str, default="experiments/uav_leo_v2x/drl_models/ddpg/v2x_medium/model.pt")
    parser.add_argument("--ddpg_hard", type=str, default="experiments/uav_leo_v2x/drl_models/ddpg/v2x_hard/model.pt")
    parser.add_argument("--ddpg_stress", type=str, default="experiments/uav_leo_v2x/drl_models/ddpg/v2x_stress/model.pt")
    parser.add_argument("--sac_easy", type=str, default="experiments/uav_leo_v2x/drl_models/sac/v2x_easy/model.pt")
    parser.add_argument("--sac_medium", type=str, default="experiments/uav_leo_v2x/drl_models/sac/v2x_medium/model.pt")
    parser.add_argument("--sac_hard", type=str, default="experiments/uav_leo_v2x/drl_models/sac/v2x_hard/model.pt")
    parser.add_argument("--sac_stress", type=str, default="experiments/uav_leo_v2x/drl_models/sac/v2x_stress/model.pt")
    parser.add_argument("--d3qn_hard", type=str, default="experiments/uav_leo_v2x/drl_models_v2/d3qn/v2x_hotspot_hard/model.pt")
    parser.add_argument("--d3qn_stress", type=str, default="experiments/uav_leo_v2x/drl_models_v2/d3qn/v2x_hotspot_stress/model.pt")
    parser.add_argument("--gat_ppo_hard", type=str, default="experiments/uav_leo_v2x/drl_models_v2/gat_ppo/v2x_hotspot_hard/model.pt")
    parser.add_argument("--gat_ppo_stress", type=str, default="experiments/uav_leo_v2x/drl_models_v2/gat_ppo/v2x_hotspot_stress/model.pt")
    parser.add_argument("--transformer_ppo_hard", type=str, default="experiments/uav_leo_v2x/drl_models_v2/transformer_ppo/v2x_hotspot_hard/model.pt")
    parser.add_argument("--transformer_ppo_stress", type=str, default="experiments/uav_leo_v2x/drl_models_v2/transformer_ppo/v2x_hotspot_stress/model.pt")
    parser.add_argument("--gdrl_hard", type=str, default="experiments/uav_leo_v2x/drl_models_v2/gdrl/gdrl/v2x_hotspot_stress/model.pt",
                        help="GDRL model (trained on stress, zero-shot transferred to hard)")
    parser.add_argument("--gdrl_stress", type=str, default="experiments/uav_leo_v2x/drl_models_v2/gdrl/gdrl/v2x_hotspot_stress/model.pt")
    parser.add_argument("--energy_weight", type=float, default=None, help="Override objective energy weight")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated policy names to run (default: all)")
    parser.add_argument("--skip_slow", action="store_true")
    parser.add_argument("--mpc_horizon", type=int, default=0,
                        help="Receding-horizon trajectory planner lookahead slots (0 = off)")
    parser.add_argument("--mpc_horizons", type=str, default=None,
                        help="Extra MPC lookahead horizons to run, comma separated (e.g. 3,10,30)")
    parser.add_argument("--mpc_offload", type=str, default="exact", choices=["exact", "tea"])
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_config.json").write_text(json.dumps(vars(args), indent=2), encoding="ascii")
    all_summary = []
    all_episodes = []
    all_steps = []

    ppo_paths = {
        "v2x_easy": args.ppo_easy,
        "v2x_medium": args.ppo_medium,
        "v2x_hard": args.ppo_hard,
        "v2x_stress": args.ppo_stress,
        "v2x_hotspot_hard": args.ppo_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.ppo_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    dqn_paths = {
        "v2x_easy": args.dqn_easy,
        "v2x_medium": args.dqn_medium,
        "v2x_hard": args.dqn_hard,
        "v2x_stress": args.dqn_stress,
        "v2x_hotspot_hard": args.dqn_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.dqn_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    ddqn_paths = {
        "v2x_easy": args.ddqn_easy,
        "v2x_medium": args.ddqn_medium,
        "v2x_hard": args.ddqn_hard,
        "v2x_stress": args.ddqn_stress,
        "v2x_hotspot_hard": args.ddqn_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.ddqn_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    td3_paths = {
        "v2x_easy": args.td3_easy,
        "v2x_medium": args.td3_medium,
        "v2x_hard": args.td3_hard,
        "v2x_stress": args.td3_stress,
        "v2x_hotspot_hard": args.td3_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.td3_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    ddpg_paths = {
        "v2x_easy": args.ddpg_easy,
        "v2x_medium": args.ddpg_medium,
        "v2x_hard": args.ddpg_hard,
        "v2x_stress": args.ddpg_stress,
        "v2x_hotspot_hard": args.ddpg_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.ddpg_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    sac_paths = {
        "v2x_easy": args.sac_easy,
        "v2x_medium": args.sac_medium,
        "v2x_hard": args.sac_hard,
        "v2x_stress": args.sac_stress,
        "v2x_hotspot_hard": args.sac_hard.replace("v2x_hard", "v2x_hotspot_hard"),
        "v2x_hotspot_stress": args.sac_stress.replace("v2x_stress", "v2x_hotspot_stress"),
    }
    d3qn_paths = {
        "v2x_hotspot_hard": args.d3qn_hard,
        "v2x_hotspot_stress": args.d3qn_stress,
    }
    gat_ppo_paths = {
        "v2x_hotspot_hard": args.gat_ppo_hard,
        "v2x_hotspot_stress": args.gat_ppo_stress,
    }
    transformer_ppo_paths = {
        "v2x_hotspot_hard": args.transformer_ppo_hard,
        "v2x_hotspot_stress": args.transformer_ppo_stress,
    }
    gdrl_paths = {
        "v2x_hotspot_hard": args.gdrl_hard,
        "v2x_hotspot_stress": args.gdrl_stress,
    }
    for difficulty in args.difficulties:
        for ablation in args.ablations:
            users = args.users if args.users is not None else users_for(difficulty)
            scenario = scenario_overrides(difficulty)
            if args.energy_weight is not None:
                scenario = {k: v for k, v in scenario.items() if k != "energy_weight"}
            ew_kwargs = {} if args.energy_weight is None else {"energy_weight": args.energy_weight}
            config = make_config(
                difficulty,
                ablation,
                seed=args.seed,
                users=users,
                leos=args.leos,
                episodes=args.episodes,
                horizon=args.horizon,
                **ew_kwargs,
                **scenario,
            )
            # pick the DRL model appropriate for the difficulty (if present)
            ppo_model = ppo_paths.get(difficulty)
            dqn_model = dqn_paths.get(difficulty)
            ddqn_model = ddqn_paths.get(difficulty)
            td3_model = td3_paths.get(difficulty)
            ddpg_model = ddpg_paths.get(difficulty)
            sac_model = sac_paths.get(difficulty)
            if ppo_model and not Path(ppo_model).exists():
                ppo_model = None
            if dqn_model and not Path(dqn_model).exists():
                dqn_model = None
            if ddqn_model and not Path(ddqn_model).exists():
                ddqn_model = None
            if td3_model and not Path(td3_model).exists():
                td3_model = None
            if ddpg_model and not Path(ddpg_model).exists():
                ddpg_model = None
            if sac_model and not Path(sac_model).exists():
                sac_model = None
            d3qn_model = d3qn_paths.get(difficulty)
            gat_ppo_model = gat_ppo_paths.get(difficulty)
            transformer_ppo_model = transformer_ppo_paths.get(difficulty)
            gdrl_model = gdrl_paths.get(difficulty)
            if d3qn_model and not Path(d3qn_model).exists():
                d3qn_model = None
            if gat_ppo_model and not Path(gat_ppo_model).exists():
                gat_ppo_model = None
            if transformer_ppo_model and not Path(transformer_ppo_model).exists():
                transformer_ppo_model = None
            if gdrl_model and not Path(gdrl_model).exists():
                gdrl_model = None
            sweep_args = argparse.Namespace(
                difficulty=difficulty,
                ablation=ablation,
                episodes=args.episodes,
                horizon=args.horizon,
                users=users,
                leos=args.leos,
                seed=args.seed,
                ppo_model=ppo_model,
                dqn_model=dqn_model,
                ddqn_model=ddqn_model,
                d3qn_model=d3qn_model,
                gat_ppo_model=gat_ppo_model,
                transformer_ppo_model=transformer_ppo_model,
                td3_model=td3_model,
                ddpg_model=ddpg_model,
                sac_model=sac_model,
                gdrl_model=gdrl_model,
                skip_slow=args.skip_slow,
                methods=args.methods,
                mpc_horizon=args.mpc_horizon,
                mpc_offload=args.mpc_offload,
            )
            policies = build_policies(sweep_args)
            if getattr(args, "mpc_horizons", None):
                from .mpc_traj import MPCTrajPolicy
                extra = [int(h) for h in args.mpc_horizons.split(",") if h.strip()]
                existing = {pol.name for pol in policies}
                for h in extra:
                    name = f"mpc_traj_h{h}"
                    if h > 0 and name not in existing:
                        policies.append(MPCTrajPolicy(horizon=h, offload=args.mpc_offload))
                        existing.add(name)
            run_dir = default_output_dir(config, root)
            summaries, episodes, steps = run_experiment(config, run_dir, policies)
            all_summary.extend(summaries)
            all_episodes.extend(episodes)
            all_steps.extend(steps)

    write_csv(root / "sweep_summary.csv", all_summary)
    write_csv(root / "sweep_episodes.csv", all_episodes)
    write_csv(root / "sweep_steps.csv", all_steps)
    print(f"Saved sweep results to {root}", flush=True)


if __name__ == "__main__":
    main()