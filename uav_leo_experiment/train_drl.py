"""Train DRL agents (PPO / DQN / TD3 / DDPG / SAC / GDRL) on the UAV-LEO v2x environment."""
import argparse
import json
from pathlib import Path

import torch

from .config import DIFFICULTY_PRESETS, make_config
from .env import UavLeoEnv
from .scenario_spec import MY_SCENARIOS, scenario_overrides

def users_for(difficulty, default=5):
    return MY_SCENARIOS.get(difficulty, {}).get("users", default)
from .learning import (DDPGTrainer, DDQNTrainer, DEVICE, DQNTrainer, PPOTrainer,
                         SACTrainer, TD3Trainer, evaluate_policy)
from .traj_drl import TrajPPOTrainer, evaluate_traj_policy

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO/DQN on UAV-LEO v2x.")
    parser.add_argument("--method", choices=["ppo", "dqn", "ddqn", "td3", "ddpg", "sac", "gdrl"], default="ppo")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_PRESETS), default="v2x_hard")
    parser.add_argument("--users", type=int, default=None)
    parser.add_argument("--leos", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--move_std", type=float, default=0.25,
                        help="std of the Gaussian UAV-move exploration head")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--pretrain_steps", type=int, default=3000)
    parser.add_argument("--pretrain_expert", type=str, default=None,
                        choices=["tea", "deadline_tea", "energy_guarded_tea", "full_offload_tea",
                                 "follow_tea", "tea_q"])
    parser.add_argument("--kl_coef", type=float, default=0.0,
                        help="KL penalty to the pretrained (BC) policy during PPO fine-tuning")
    parser.add_argument("--critic_warmup_updates", type=int, default=0,
                        help="PPO updates with the policy head frozen (critic-only warmup)")
    parser.add_argument("--output_dir", type=str, default="experiments/uav_leo_v2x/drl_models")
    parser.add_argument("--overrides", type=str, default="{}",
                        help="JSON dict of extra UavLeoConfig fields (explicit scenario parameters)")
    return parser.parse_args()


def main():
    args = parse_args()
    user_overrides = json.loads(args.overrides) if args.overrides else {}
    overrides = {**scenario_overrides(args.difficulty), **user_overrides}
    config = make_config(args.difficulty, "none",
                         users=(args.users if args.users is not None else users_for(args.difficulty)), leos=args.leos,
                         episodes=1, horizon=args.horizon, seed=args.seed, **overrides)
    out = Path(args.output_dir) / args.method / args.difficulty
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(str(args), encoding="utf-8")

    env = UavLeoEnv(config)
    eval_env = UavLeoEnv(make_config(args.difficulty, "none",
                                     users=(args.users if args.users is not None else users_for(args.difficulty)), leos=args.leos,
                                     episodes=5, horizon=args.horizon, seed=args.seed, **overrides))

    if args.method == "ppo":
        trainer = PPOTrainer(config, lr=args.lr, rollout_steps=args.rollout_steps,
                             minibatch=args.batch_size,
                             entropy_coef=args.entropy_coef if args.entropy_coef is not None else 0.003,
                             move_std=args.move_std)
        if args.pretrain_steps > 0:
            trainer.pretrain_expert(env, steps=args.pretrain_steps, seed=args.seed,
                                     expert=args.pretrain_expert)
            bc_score = evaluate_policy(trainer, eval_env, episodes=5, deterministic=True)
            bc_state = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
            best_path = out / "model_best.pt"
            if not best_path.exists() or bc_score > float(torch.load(best_path, map_location="cpu")["eval_reward"]):
                torch.save({"state_dict": trainer.policy.state_dict(), "U": trainer.U,
                            "eval_reward": bc_score}, best_path)
            print(f"BC eval = {bc_score:.2f} (seeded model_best)", flush=True)
        kl_anchor = None
        if (args.kl_coef > 0.0 or args.critic_warmup_updates > 0) and args.pretrain_steps > 0:
            kl_anchor = bc_state
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed,
                      kl_anchor=kl_anchor, kl_coef=args.kl_coef,
                      critic_warmup_updates=args.critic_warmup_updates)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.policy.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.policy.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = evaluate_policy(trainer, eval_env, episodes=10, deterministic=True)
        trainer.save(out / "model.pt")

    if args.method == "gdrl":
        # Residual trajectory PPO: delta = 0 reproduces the demand-predictive
        # expert exactly, so we start from a zero-initialized policy.
        trainer = TrajPPOTrainer(config, lr=args.lr, rollout_steps=args.rollout_steps,
                                 minibatch=args.batch_size,
                                 entropy_coef=args.entropy_coef if args.entropy_coef is not None else 0.01,
                                 move_std=args.move_std)
        with torch.no_grad():
            for p in trainer.policy.delta_head.parameters():
                p.zero_()
        if args.pretrain_steps > 0:
            bc_score = evaluate_traj_policy(trainer, eval_env, episodes=5, deterministic=True)
            bc_state = {key: value.clone() for key, value in trainer.policy.state_dict().items()}
            best_path = out / "model_best.pt"
            if not best_path.exists() or bc_score > float(torch.load(best_path, map_location="cpu")["eval_reward"]):
                torch.save({"state_dict": trainer.policy.state_dict(),
                            "eval_reward": bc_score}, best_path)
            print(f"BC eval = {bc_score:.2f} (seeded model_best)", flush=True)
        kl_anchor = None
        if (args.kl_coef > 0.0 or args.critic_warmup_updates > 0) and args.pretrain_steps > 0:
            kl_anchor = bc_state
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed,
                      kl_anchor=kl_anchor, kl_coef=args.kl_coef,
                      critic_warmup_updates=args.critic_warmup_updates)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.policy.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.policy.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = evaluate_traj_policy(trainer, eval_env, episodes=10, deterministic=True)
        trainer.save(out / "model.pt")

    elif args.method == "td3":
        trainer = TD3Trainer(config, lr=args.lr, batch_size=args.batch_size)
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.actor.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.actor.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = trainer.evaluate(eval_env, episodes=10)
        trainer.save(out / "model.pt")
    elif args.method == "ddpg":
        trainer = DDPGTrainer(config, lr=args.lr, batch_size=args.batch_size)
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.actor.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.actor.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = trainer.evaluate(eval_env, episodes=10)
        trainer.save(out / "model.pt")
    elif args.method == "sac":
        trainer = SACTrainer(config, lr=args.lr, batch_size=args.batch_size)
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.actor.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.actor.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = trainer.evaluate(eval_env, episodes=10)
        trainer.save(out / "model.pt")
    elif args.method == "ddqn":
        trainer = DDQNTrainer(config, lr=args.lr, batch_size=args.batch_size)
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.q_net.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.q_net.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = trainer.evaluate(eval_env, episodes=10)
        trainer.save(out / "model.pt")
    else:
        trainer = DQNTrainer(config, lr=args.lr, batch_size=args.batch_size)
        trainer.train(env, steps=args.steps, eval_env=eval_env,
                      eval_every=args.eval_every, log_path=out, seed=args.seed)
        best_path = out / "model_best.pt"
        if best_path.exists():
            trainer.q_net.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
            trainer.q_net.to(DEVICE)
            print("Loaded best eval checkpoint", flush=True)
        final = trainer.evaluate(eval_env, episodes=10)
        trainer.save(out / "model.pt")
    print(f"Saved model to {out / 'model.pt'}; final eval reward = {final:.2f}", flush=True)


if __name__ == "__main__":
    main()
