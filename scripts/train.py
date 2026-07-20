"""
scripts/train.py — GDRL 单场景训练入口（重构后版本）
====================================================
用法：
    python scripts/train.py --U 3 --L 16 --N 16 --T 100 --episodes 600
"""
import os
import sys
import csv
from pathlib import Path

# 确保项目根目录在 Python 路径中（scripts/ 子目录运行时需要）
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_edge_index
from sb3_contrib import TRPO
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from arg_parser import get_args
from experiment_config import amn_paths, default_single_dir, scenario_tag, write_manifest
from gdrl.core.user_request import all_user_feature
from gdrl.core.graph import GenerateAdjacency
from gdrl.core.update import updatevalue
from gdrl.core.callback import CustomCallback
from gdrl.models.amn import AutoencoderDis, AutoencoderCon, M1EncoderDis, M1DecoderDis, M1EncoderCon, M1DecoderCon
from gdrl.models.feature import CustomFeaturesExtractor
from gdrl.envs.environment import NetworkEnvironment


if __name__ == "__main__":
    args = get_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_single_dir(args)
    model_dir = output_dir / "model_save"
    tensorboard_dir = output_dir / "trpo_tensorboard"
    autoencoder_log_dir = output_dir / "autoencoder_status"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    amn_info = amn_paths(args)
    amn_info["dir"].mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "run_config.json", args, {"mode": "single_gdrl"})
    print(f"Scenario: {scenario_tag(args)}")
    print(f"Output directory: {output_dir}")

    Uf, Pu, Su, Ou, Vu, Lu = all_user_feature(args.U, args.T)
    user_requests = {'Uf': Uf, 'Pu': Pu, 'Su': Su, 'Ou': Ou, 'Vu': Vu, 'Lu': Lu}
    np.savez(output_dir / 'user_requests.npz', **user_requests)
    np.savez('user_requests.npz', **user_requests)  # 兼容根目录读取

    action_space_len, Adj_Matrix, user_lists = GenerateAdjacency(args.U, args.L, args.N)
    save_var, load_var = updatevalue()
    custom_callback = CustomCallback()

    autoencoder_dis = AutoencoderDis(16, action_space_len, args.U, user_lists, M1EncoderDis, M1DecoderDis)
    autoencoder_con = AutoencoderCon(16, 2*args.U, M1EncoderCon, M1DecoderCon)
    encoder_dis = autoencoder_dis.encoder
    encoder_con = autoencoder_con.encoder
    optimizer1 = torch.optim.Adam(autoencoder_dis.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(autoencoder_con.parameters(), lr=0.001)
    criterion1 = nn.MSELoss(); criterion2 = nn.MSELoss()

    sparse_Adj_Matrix = torch.Tensor(Adj_Matrix).to_sparse()
    edge_index, _ = to_edge_index(sparse_Adj_Matrix)
    torch.save(edge_index, output_dir / 'edge_index.pt')
    torch.save(edge_index, 'edge_index.pt')   # Feature.py 从根目录读取

    amn_loaded = False
    if not args.force_retrain_amn:
        try:
            autoencoder_dis.load_state_dict(torch.load(amn_info["dis"], map_location="cpu"))
            autoencoder_con.load_state_dict(torch.load(amn_info["con"], map_location="cpu"))
            encoder_dis = autoencoder_dis.encoder
            encoder_con = autoencoder_con.encoder
            amn_loaded = True
            print(f"Loaded scenario AMN weights from {amn_info['dir']}")
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Will train AMN for this scenario: {exc}")
    else:
        print("Force retrain AMN is enabled; saved AMN weights will be ignored.")

    for i in range(1):
        env = NetworkEnvironment(args.U, args.L, args.N, args.T, user_requests, user_lists,
                                 save_var, load_var, encoder_dis, encoder_con)
        env = Monitor(env, filename=str(output_dir / "monitor_logs"), allow_early_resets=True)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=32),
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4])
        )
        model_file = model_dir / f"trpo_trained_{scenario_tag(args)}.zip"
        if os.path.isfile(model_file):
            model = TRPO.load(str(model_file)); model.set_env(env)
        else:
            model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T,
                         batch_size=args.T, learning_rate=3e-4, cg_damping=0.2,
                         target_kl=0.005, verbose=1, device="auto",
                         tensorboard_log=str(tensorboard_dir))
        model.learn(total_timesteps=args.total_step, tb_log_name="first_run", callback=custom_callback)
        model.save(str(model_file))

        actions, rewards, latency = custom_callback.get_training_data()
        np.save(output_dir / f'latency{i}.npy', latency)
        np.save(output_dir / f'action{i}.npy', actions)

        actions = torch.from_numpy(actions).squeeze()
        dataset_dis = TensorDataset(actions[:, 0:16], actions[:, 0:16])
        dataset_con = TensorDataset(actions[:, 16:32], actions[:, 16:32])
        dataloader_dis = DataLoader(dataset_dis, batch_size=args.T, shuffle=False)
        dataloader_con = DataLoader(dataset_con, batch_size=args.T, shuffle=False)
        writer = SummaryWriter(str(autoencoder_log_dir)); loss_rows = []

        for epoch in range(args.amn_epochs):
            dis_losses, con_losses = [], []
            for inp, tgt in dataloader_dis:
                optimizer1.zero_grad()
                loss1 = criterion1(autoencoder_dis(inp), tgt)
                loss1.backward(); optimizer1.step()
                writer.add_scalar('Loss/Autoencoder_DIS', loss1.item(), epoch * len(dataloader_dis) + i)
                dis_losses.append(loss1.item())
            for inp, tgt in dataloader_con:
                optimizer2.zero_grad()
                loss2 = criterion2(autoencoder_con(inp), tgt)
                loss2.backward(); optimizer2.step()
                writer.add_scalar('Loss/Autoencoder_CON', loss2.item(), epoch * len(dataloader_con) + i)
                con_losses.append(loss2.item())
            loss_rows.append({"epoch": epoch+1, "loss_dis": float(np.mean(dis_losses)),
                               "loss_con": float(np.mean(con_losses))})

        print('Finish Training autoencoder')
        with open(amn_info["loss"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss_dis", "loss_con"])
            w.writeheader(); w.writerows(loss_rows)
        torch.save(autoencoder_dis.state_dict(), amn_info["dis"])
        torch.save(autoencoder_con.state_dict(), amn_info["con"])
        print(f"Saved scenario AMN weights to {amn_info['dir']}")
        writer.close()
