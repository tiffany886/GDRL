import argparse
import csv
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import TRPO
from torch_geometric.utils import to_edge_index
from experiment_config import (
    add_scenario_arguments,
    amn_paths,
    apply_scenario,
    default_compare_dir,
    scenario_tag,
    validate_scenario,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GDRL with lightweight baselines.")
    parser.add_argument("--methods", nargs="+", default=["random", "trpo_mlp", "gdrl"],
                        choices=["random", "trpo_mlp", "ppo_mlp", "gdrl", "gdrl_sac"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--U", type=int, default=3)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amn_epochs", type=int, default=100)
    parser.add_argument("--force_retrain_amn", action="store_true")
    parser.add_argument(
        "--online_amn_update",
        action="store_true",
        help="在 GDRL 训练期间用真实 latent action 继续微调 AMN；默认关闭，避免干扰基线结果。",
    )
    parser.add_argument(
        "--energy_weight", type=float, default=0.1,
        help="奖励函数中能耗惩罚系数 γ₃（默认 0.1）。设为 0 可禁用能耗项。"
    )
    parser.add_argument("--sac_buffer_size", type=int, default=50000,
                        help="SAC replay buffer size")
    parser.add_argument("--sac_batch_size", type=int, default=256,
                        help="SAC mini-batch size")
    parser.add_argument("--sac_tau", type=float, default=0.005,
                        help="SAC soft update coefficient")
    parser.add_argument(
        "--use_gat", action="store_true", default=True,
        help="使用 GATv2Conv（默认）。传 --no-use_gat 可还原 GCN 做消融对比。"
    )
    parser.add_argument("--no-use_gat", dest="use_gat", action="store_false")
    add_scenario_arguments(parser)
    args = parser.parse_args()
    args = apply_scenario(args)
    validate_scenario(args)
    return args


def configure_project_argv(args):
    # Feature.py 会读取 arg_parser.get_args() 的结果，这里同步当前实验参数。
    sys.argv = [
        sys.argv[0],
        "--U", str(args.U),
        "--L", str(args.L),
        "--N", str(args.N),
        "--T", str(args.T),
        "--total_step", str(args.episodes * args.T),
        "--energy_weight", str(getattr(args, "energy_weight", 0.1)),
    ] + (["--use_gat"] if getattr(args, "use_gat", True) else ["--no-use_gat"])


class MetricsCallback(BaseCallback):
    def __init__(self, autoencoder_dis=None, autoencoder_con=None, online_update=False,
                 update_epochs=1, update_batch_size=100, online_update_interval=10,
                 online_warmup_episodes=20, replay_episode_capacity=20):
        super().__init__()
        self.episode_rewards = []
        self.step_latencies = []
        self.step_energies = []
        self._current_episode_latents = []
        self._recent_episode_latents = deque(maxlen=max(1, int(replay_episode_capacity)))
        self.online_update_losses = []
        self._episode_count = 0
        self.autoencoder_dis = autoencoder_dis
        self.autoencoder_con = autoencoder_con
        self.online_update = bool(online_update and autoencoder_dis is not None and autoencoder_con is not None)
        self.update_epochs = max(1, int(update_epochs))
        self.update_batch_size = max(1, int(update_batch_size))
        self.online_update_interval = max(1, int(online_update_interval))
        self.online_warmup_episodes = max(1, int(online_warmup_episodes))
        self.criterion = torch.nn.MSELoss()
        self.optimizer_dis = None
        self.optimizer_con = None
        if self.online_update:
            self.optimizer_dis = torch.optim.Adam(self.autoencoder_dis.parameters(), lr=1e-4)
            self.optimizer_con = torch.optim.Adam(self.autoencoder_con.parameters(), lr=1e-4)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions")
        if actions is not None:
            action_array = np.asarray(actions, dtype=np.float32)
            if action_array.ndim == 1:
                action_array = action_array.reshape(1, -1)
        else:
            action_array = None
        for info in infos:
            latency = info.get("latency")
            if latency is not None:
                self.step_latencies.append(to_float(latency))
            energy = info.get("energy", 0.0)
            self.step_energies.append(float(energy) if energy is not None else 0.0)
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        if self.online_update and action_array is not None:
            for index, info in enumerate(infos):
                if index < len(action_array):
                    self._current_episode_latents.append(action_array[index].reshape(-1))
                if "episode" in info:
                    self._episode_count += 1
                    if self._current_episode_latents:
                        episode_reward = float(info["episode"]["r"])
                        episode_latents = np.asarray(self._current_episode_latents, dtype=np.float32)
                        self._recent_episode_latents.append((episode_reward, episode_latents))
                    self._current_episode_latents.clear()
                    if (
                        self._episode_count >= self.online_warmup_episodes
                        and self._episode_count % self.online_update_interval == 0
                    ):
                        self._update_online_amn()
        return True

    def _update_online_amn(self):
        if not self._recent_episode_latents:
            return
        episode_rewards = np.asarray([item[0] for item in self._recent_episode_latents], dtype=np.float32)
        reward_threshold = float(np.percentile(episode_rewards, 60))
        selected_latents = [latents for reward, latents in self._recent_episode_latents if reward >= reward_threshold]
        if not selected_latents:
            selected_latents = [latents for _, latents in self._recent_episode_latents]
        latent_actions = np.concatenate(selected_latents, axis=0).astype(np.float32, copy=False)
        if latent_actions.ndim != 2 or latent_actions.shape[1] < 32:
            return

        dis_inputs = torch.as_tensor(latent_actions[:, :16], dtype=torch.float32)
        con_inputs = torch.as_tensor(latent_actions[:, 16:32], dtype=torch.float32)
        batch_dis = min(self.update_batch_size, len(dis_inputs))
        batch_con = min(self.update_batch_size, len(con_inputs))
        if batch_dis <= 0 or batch_con <= 0:
            return

        loader_dis = DataLoader(
            TensorDataset(dis_inputs, dis_inputs),
            batch_size=batch_dis,
            shuffle=True,
        )
        loader_con = DataLoader(
            TensorDataset(con_inputs, con_inputs),
            batch_size=batch_con,
            shuffle=True,
        )

        self.autoencoder_dis.train()
        self.autoencoder_con.train()
        dis_losses = []
        con_losses = []
        for _ in range(self.update_epochs):
            for inputs, targets in loader_dis:
                self.optimizer_dis.zero_grad()
                loss = self.criterion(self.autoencoder_dis(inputs), targets)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder_dis.parameters(), max_norm=1.0)
                self.optimizer_dis.step()
                dis_losses.append(float(loss.item()))
            for inputs, targets in loader_con:
                self.optimizer_con.zero_grad()
                loss = self.criterion(self.autoencoder_con(inputs), targets)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder_con.parameters(), max_norm=1.0)
                self.optimizer_con.step()
                con_losses.append(float(loss.item()))
        self.autoencoder_dis.eval()
        self.autoencoder_con.eval()
        if dis_losses or con_losses:
            self.online_update_losses.append({
                "episode": len(self.episode_rewards),
                "loss_dis": float(np.mean(dis_losses)) if dis_losses else float("nan"),
                "loss_con": float(np.mean(con_losses)) if con_losses else float("nan"),
            })


def to_float(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value, dtype=float).mean())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_state_store(args, ResetFunction):
    counter = 0
    LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
        user_status, A_u, A_u_ori, A_resource = ResetFunction(args.N, args.L, args.T, args.U)

    def save_variable(new_LEO_status, new_HAPS_status, new_LEO_resource_status, new_HAPS_resource_status,
                      new_user_status, new_A_u, new_A_u_ori, new_A_resource, new_counter):
        nonlocal LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status
        nonlocal user_status, A_u, A_u_ori, A_resource, counter
        LEO_status = new_LEO_status
        HAPS_status = new_HAPS_status
        LEO_resource_status = new_LEO_resource_status
        HAPS_resource_status = new_HAPS_resource_status
        user_status = new_user_status
        A_u = new_A_u
        A_u_ori = new_A_u_ori
        A_resource = new_A_resource
        counter = new_counter

    def load_variable():
        return LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status, \
            user_status, A_u, A_u_ori, A_resource, counter

    return save_variable, load_variable


def build_components(args, output_dir):
    from Generate_adj_matrix import GenerateAdjacency
    from Generate_inital_environment import ResetFunction
    from UserRequest import all_user_feature
    from amp import AutoencoderCon, AutoencoderDis, M1DecoderCon, M1DecoderDis, M1EncoderCon, M1EncoderDis

    Uf, Pu, Su, Ou, Vu, Lu = all_user_feature(args.U, args.T)
    user_requests = {"Uf": Uf, "Pu": Pu, "Su": Su, "Ou": Ou, "Vu": Vu, "Lu": Lu}
    np.savez(output_dir / "user_requests.npz", **user_requests)
    # 兼容少数旧模块；主要实验数据仍保存在 output_dir。
    np.savez("user_requests.npz", **user_requests)

    action_space_len, adj_matrix, user_lists = GenerateAdjacency(args.U, args.L, args.N)
    sparse_adj_matrix = torch.Tensor(adj_matrix).to_sparse()
    edge_index, _ = to_edge_index(sparse_adj_matrix)
    torch.save(edge_index, output_dir / "edge_index.pt")
    # Feature.py 当前从项目根目录读取 edge_index.pt，因此保留一份运行时副本。
    torch.save(edge_index, "edge_index.pt")

    # 保存静态边特征（归一化节点间距离），供 GATv2Conv 使用。
    # 必须在 build_components 中生成，以确保边数与当前场景匹配。
    from gdrl.core.graph import compute_edge_attr as _cef
    from UserStatus import all_user_status as _aus
    from gdrl.core.nodes import all_LEO_status as _als, all_HAPS_status as _ahs
    _, _U_place = _aus(args.U)
    _, _LEO_place, _ = _als(args.L)
    _, _HAPS_place, _ = _ahs(args.N)
    _ei_np = np.stack(np.where(adj_matrix > 0), axis=0)
    _edge_attr = _cef(_ei_np, _U_place, _LEO_place, _HAPS_place, args.U, args.L, args.N)
    torch.save(_edge_attr, output_dir / "edge_attr.pt")
    torch.save(_edge_attr, "edge_attr.pt")

    autoencoder_dis = AutoencoderDis(16, action_space_len, args.U, user_lists, M1EncoderDis, M1DecoderDis)
    autoencoder_con = AutoencoderCon(16, 2 * args.U, M1EncoderCon, M1DecoderCon)
    return user_requests, user_lists, autoencoder_dis, autoencoder_con, ResetFunction


def train_or_load_amn(args, autoencoder_dis, autoencoder_con, output_dir):
    """每个场景单独训练/加载 AMN，防止不同 U/L/N 的权重互相覆盖。"""
    from torch.utils.data import DataLoader, TensorDataset
    from amp import M1DecoderCon, M1DecoderDis

    paths = amn_paths(args)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if not args.force_retrain_amn:
        try:
            autoencoder_dis.load_state_dict(torch.load(paths["dis"], map_location="cpu"))
            autoencoder_con.load_state_dict(torch.load(paths["con"], map_location="cpu"))
            print(f"Loaded AMN weights for {scenario_tag(args)}")
            report_amn_range(autoencoder_con.encoder, args.U)
            return autoencoder_dis, autoencoder_con, autoencoder_dis.encoder, autoencoder_con.encoder
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"AMN will be trained for {scenario_tag(args)}: {exc}")
    else:
        print(f"Force retrain AMN for {scenario_tag(args)}")

    # 预训练样本不需要随强化学习轮数无限增长；1 万个样本足够覆盖初始动作分布。
    total_samples = min(max(args.episodes * args.T, args.T), 10000)
    # 初始策略动作主要集中在 0 附近，截断正态分布比 [-10, 10] 均匀分布更符合实际。
    # 后续 main.py 的单算法训练还会用真实策略动作继续微调 AMN。
    latent_actions = torch.randn(total_samples, 32).mul_(2.0).clamp_(-10.0, 10.0)
    actions_dis = latent_actions[:, :16]
    actions_con = latent_actions[:, 16:32]
    loader_dis = DataLoader(TensorDataset(actions_dis, actions_dis), batch_size=args.T, shuffle=True)
    loader_con = DataLoader(TensorDataset(actions_con, actions_con), batch_size=args.T, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer_dis = torch.optim.Adam(autoencoder_dis.parameters(), lr=0.001)
    optimizer_con = torch.optim.Adam(autoencoder_con.parameters(), lr=0.001)
    loss_rows = []

    for epoch in range(args.amn_epochs):
        dis_losses = []
        con_losses = []
        for inputs, targets in loader_dis:
            optimizer_dis.zero_grad()
            loss = criterion(autoencoder_dis(inputs), targets)
            loss.backward()
            optimizer_dis.step()
            dis_losses.append(loss.item())
        for inputs, targets in loader_con:
            optimizer_con.zero_grad()
            loss = criterion(autoencoder_con(inputs), targets)
            loss.backward()
            optimizer_con.step()
            con_losses.append(loss.item())
        loss_rows.append({
            "epoch": epoch + 1,
            "loss_dis": float(np.mean(dis_losses)),
            "loss_con": float(np.mean(con_losses)),
        })

    write_csv(paths["loss"], loss_rows)
    torch.save(autoencoder_dis.state_dict(), paths["dis"])
    torch.save(autoencoder_con.state_dict(), paths["con"])
    print(f"Saved AMN weights to {paths['dir']}")
    report_amn_range(autoencoder_con.encoder, args.U)
    return autoencoder_dis, autoencoder_con, autoencoder_dis.encoder, autoencoder_con.encoder


def report_amn_range(encoder_con, user_count):
    """在强化学习开始前检查 AMN 输出，便于尽早发现异常权重。"""
    with torch.no_grad():
        probe = torch.linspace(-10.0, 10.0, steps=16).repeat(64, 1)
        output = encoder_con(probe)
        if output.ndim == 2:
            output = output.unsqueeze(0)
        if not torch.isfinite(output).all():
            raise ValueError("AMN continuous encoder produced NaN or Inf values.")
        compute_min = float(output[:, :, 0].min())
        channel_min = float(output[:, :, 1].min())
        channel_sum_error = float((output[:, :, 1].sum(dim=1) - 1.0).abs().max())
        print(
            "AMN range check: "
            f"compute_min={compute_min:.6f}, channel_min={channel_min:.6f}, "
            f"channel_sum_error={channel_sum_error:.2e}"
        )
        if compute_min < 0.049 or channel_min < 0.009:
            raise ValueError(
                f"AMN allocation is unsafe for U={user_count}: "
                f"compute_min={compute_min}, channel_min={channel_min}"
            )


def make_raw_env(args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction, monitor_path=None):
    from Environment_baseline import NetworkEnvironment

    save_var, load_var = make_state_store(args, ResetFunction)
    env = NetworkEnvironment(
        args.U, args.L, args.N, args.T, user_requests, user_lists,
        save_var, load_var, encoder_dis, encoder_con
    )
    if monitor_path is not None:
        env = Monitor(env, filename=str(monitor_path), allow_early_resets=True)
    return env


def make_vec_env(args, method, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction, output_dir):
    monitor_path = output_dir / f"{method}.monitor.csv"
    raw_env = make_raw_env(args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction, monitor_path)
    env = DummyVecEnv([lambda: raw_env])
    return VecNormalize(env, norm_obs=False, norm_reward=True)


def evaluate_random(args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction):
    env = make_raw_env(args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction)
    episode_rewards = []
    step_latencies = []
    step_energies = []
    obs, _ = env.reset()

    current_reward = 0.0
    for _ in range(args.episodes * args.T):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        current_reward += float(reward)
        if info.get("latency") is not None:
            step_latencies.append(to_float(info["latency"]))
        step_energies.append(float(info.get("energy", 0.0)))
        if terminated or truncated:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            obs, _ = env.reset()

    env.close()
    return episode_rewards, step_latencies, step_energies, 0.0


def train_method(args, method, user_requests, user_lists, encoder_dis, encoder_con,
                 autoencoder_dis, autoencoder_con, ResetFunction, output_dir):
    from Feature import CustomFeaturesExtractor

    env = make_vec_env(args, method, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction, output_dir)
    callback = MetricsCallback(
        autoencoder_dis=autoencoder_dis if method == "gdrl" else None,
        autoencoder_con=autoencoder_con if method == "gdrl" else None,
        online_update=(method == "gdrl" and args.online_amn_update),
        update_epochs=1,
        update_batch_size=args.T,
    )
    total_timesteps = args.episodes * args.T

    if method == "gdrl":
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4]),
        )
        model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T,
                     batch_size=args.T, learning_rate=3e-4, cg_damping=0.2,
                     target_kl=0.005, verbose=0, device=args.device, seed=args.seed)
    elif method == "trpo_mlp":
        policy_kwargs = dict(
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4]),
        )
        model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T,
                     batch_size=args.T, verbose=0, device=args.device, seed=args.seed)
    elif method == "ppo_mlp":
        policy_kwargs = dict(
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4]),
        )
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T,
                    batch_size=args.T, verbose=0, device=args.device, seed=args.seed)
    elif method == "gdrl_sac":
        from stable_baselines3 import SAC
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64], qf=[64, 64]),
        )
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs,
                    learning_rate=3e-4,
                    buffer_size=getattr(args, 'sac_buffer_size', 50000),
                    batch_size=getattr(args, 'sac_batch_size', 256),
                    tau=getattr(args, 'sac_tau', 0.005),
                    gamma=0.995,
                    verbose=0, device=args.device, seed=args.seed)
    else:
        raise ValueError(f"Unknown trainable method: {method}")

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - start
    model.save(str(output_dir / f"{method}_model"))
    if method == "gdrl" and autoencoder_dis is not None and autoencoder_con is not None:
        paths = amn_paths(args)
        torch.save(autoencoder_dis.state_dict(), paths["dis"])
        torch.save(autoencoder_con.state_dict(), paths["con"])
        if callback.online_update:
            print(f"Saved online-adapted AMN weights to {paths['dir']}")
        else:
            print(f"Saved AMN weights to {paths['dir']}")
    env.close()
    return callback.episode_rewards, callback.step_latencies, callback.step_energies, elapsed


def summarize(method, episode_rewards, step_latencies, step_energies, elapsed):
    rewards = np.asarray(episode_rewards, dtype=float)
    latencies = np.asarray(step_latencies, dtype=float)
    return {
        "method": method,
        "episodes": int(len(rewards)),
        "reward_mean": float(rewards.mean()) if rewards.size else float("nan"),
        "reward_std": float(rewards.std(ddof=0)) if rewards.size else float("nan"),
        "reward_min": float(rewards.min()) if rewards.size else float("nan"),
        "reward_max": float(rewards.max()) if rewards.size else float("nan"),
        "reward_last": float(rewards[-1]) if rewards.size else float("nan"),
        "latency_mean": float(latencies.mean()) if latencies.size else float("nan"),
        "latency_p95": float(np.percentile(latencies, 95)) if latencies.size else float("nan"),
        "latency_max": float(latencies.max()) if latencies.size else float("nan"),
        "energy_mean": float(np.mean(step_energies)) if len(step_energies) > 0 else 0.0,
        "energy_total": float(np.sum(step_energies)) if len(step_energies) > 0 else 0.0,
        "energy_p95": float(np.percentile(step_energies, 95)) if len(step_energies) > 0 else 0.0,
        "elapsed_sec": float(elapsed),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    configure_project_argv(args)
    set_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else default_compare_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "run_config.json", args, {
        "mode": "compare_baselines",
        "methods": args.methods,
        "episodes": args.episodes,
        "amn_epochs": args.amn_epochs,
    })
    print(f"Scenario: {scenario_tag(args)}")
    print(f"Output directory: {output_dir}")

    rows = []
    detail_rows = []
    user_requests, user_lists, autoencoder_dis, autoencoder_con, ResetFunction = build_components(args, output_dir)
    autoencoder_dis, autoencoder_con, encoder_dis, encoder_con = train_or_load_amn(
        args, autoencoder_dis, autoencoder_con, output_dir
    )

    for index, method in enumerate(args.methods):
        set_seed(args.seed + index)
        torch.cuda.empty_cache()

        if method == "random":
            episode_rewards, step_latencies, step_energies, elapsed = evaluate_random(
                args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction
            )
        else:
            episode_rewards, step_latencies, step_energies, elapsed = train_method(
                args, method, user_requests, user_lists, encoder_dis, encoder_con,
                autoencoder_dis, autoencoder_con, ResetFunction, output_dir
            )

        np.save(output_dir / f"latency_{method}.npy", np.asarray(step_latencies, dtype=float))
        np.save(output_dir / f"energy_{method}.npy", np.asarray(step_energies, dtype=float))
        row = summarize(method, episode_rewards, step_latencies, step_energies, elapsed)
        rows.append(row)
        for episode, reward in enumerate(episode_rewards, start=1):
            detail_rows.append({"method": method, "episode": episode, "reward": float(reward)})

        print(
            f"{method:8s} reward_mean={row['reward_mean']:.3f} "
            f"reward_last={row['reward_last']:.3f} "
            f"latency_mean={row['latency_mean']:.6f} "
            f"latency_p95={row['latency_p95']:.6f}"
        )

    write_csv(output_dir / "baseline_summary.csv", rows)
    write_csv(output_dir / "baseline_episodes.csv", detail_rows)
    print(f"Saved summary to {output_dir / 'baseline_summary.csv'}")


if __name__ == "__main__":
    main()
