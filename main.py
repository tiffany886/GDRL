import os
import csv
from pathlib import Path

import torch.nn

from Environment_baseline import NetworkEnvironment
from Generate_adj_matrix import *
from UserRequest import all_user_feature
from UserStatus import *
from torch_geometric.utils import to_edge_index
from arg_parser import get_args
from UpdateVariable import updatevalue
from amp import *
from sb3_contrib import TRPO
from CallBack import CustomCallback
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Feature import CustomFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from experiment_config import amn_paths, default_single_dir, scenario_tag, write_manifest



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
    user_requests = {
        'Uf': Uf,
        'Pu': Pu,
        'Su': Su,
        'Ou': Ou,
        'Vu': Vu,
        'Lu': Lu,
    }
    np.savez(output_dir / 'user_requests.npz', **user_requests)
    # 兼容 Environment_baseline.reset 里的旧读取逻辑，同时主要结果放到场景目录。
    np.savez('user_requests.npz', **user_requests)
    action_space_len, Adj_Matrix, user_lists = GenerateAdjacency(args.U, args.L, args.N)
    save_var, load_var = updatevalue()
    #encoder_dis = M1EncoderDis(16, action_space_len)  # The output dim is the total number of possible actions foe all users
    #encoder_con = M1EncoderCon(16, 2*args.U)  # The output dim is the number of continious actions
    custom_callback = CustomCallback()  # Define the callback function for recording the reward and action
    # Defination related to autoencoder
    autoencoder_dis = AutoencoderDis(16, action_space_len, args.U, user_lists, M1EncoderDis, M1DecoderDis)
    autoencoder_con = AutoencoderCon(16, 2*args.U, M1EncoderCon, M1DecoderCon)
    encoder_dis = autoencoder_dis.encoder
    encoder_con = autoencoder_con.encoder
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer1 = torch.optim.Adam(autoencoder_dis.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(autoencoder_con.parameters(), lr=0.001)

    sparse_Adj_Matrix = torch.Tensor(Adj_Matrix).to_sparse()
    edge_index, _ = to_edge_index(sparse_Adj_Matrix)
    torch.save(edge_index, output_dir / 'edge_index.pt')
    # Feature.py 当前从项目根目录读取 edge_index.pt，因此保留一份运行时副本。
    torch.save(edge_index, 'edge_index.pt')
    train_model_num = 1
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

    # 先训练/加载 GDRL，再用采集到的 latent actions 训练当前场景专属 AMN。
    for i in range(train_model_num):
        env = NetworkEnvironment(args.U, args.L, args.N, args.T, user_requests, user_lists, save_var,
                                 load_var, encoder_dis, encoder_con)
        env = Monitor(env, filename=str(output_dir / "monitor_logs"), allow_early_resets=True)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)
        #policy_kwargs = dict(activation_fn=torch.nn.Sigmoid,
                             #net_arch=dict(pi=[32, 32], vf=[16, 4]))
        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=32
            ),
            activation_fn=torch.nn.Sigmoid,
            net_arch=dict(pi=[32, 32], vf=[16, 4])
        )
        model_file = model_dir / f"trpo_trained_{scenario_tag(args)}.zip"
        # Check if the model file exists
        if os.path.isfile(model_file):
            # Load the existing model
            model = TRPO.load(str(model_file))
            model.set_env(env)
        else:
            # Initialize a new model because the saved model does not exist
            model = TRPO("MlpPolicy", env, policy_kwargs=policy_kwargs, n_steps=args.T, batch_size=args.T,
                         learning_rate=3e-4, cg_damping=0.2, target_kl=0.005,
                         verbose=1, device="auto", tensorboard_log=str(tensorboard_dir))
        model.learn(total_timesteps=args.total_step, tb_log_name="first_run", callback=custom_callback)
        model.save(str(model_file))
        actions, rewards, latency = custom_callback.get_training_data()
        np.save(output_dir / f'latency{i}.npy', latency)
        np.save(output_dir / f'action{i}.npy', actions)
        # The shape of action is [total_timesteps, 1, 32], reward: [total_timesteps, 1]
        actions = torch.from_numpy(actions).squeeze()
        actions_dis = actions[:, 0:16]
        actions_con = actions[:, 16:32]
        # For autoencoders, input data is also the target. So, both inputs and targets are X_tensor.
        dataset_dis = TensorDataset(actions_dis, actions_dis)
        dataloader_dis = DataLoader(dataset_dis, batch_size=args.T, shuffle=False)
        dataset_con = TensorDataset(actions_con, actions_con)
        dataloader_con = DataLoader(dataset_con, batch_size=args.T, shuffle=False)
        writer = SummaryWriter(str(autoencoder_log_dir))
        loss_rows = []

        for epoch in range(args.amn_epochs):
            epoch_dis_losses = []
            epoch_con_losses = []
            for input, target in dataloader_dis:
                # 训练离散 AMN：把 latent action 重构回来，loss 越小说明动作映射越可靠。
                optimizer1.zero_grad()
                output1 = autoencoder_dis(input)
                loss1 = criterion1(output1, target)
                loss1.backward()
                optimizer1.step()
                writer.add_scalar('Loss/Autoencoder_DIS', loss1.item(), epoch * len(dataloader_dis) + i)
                epoch_dis_losses.append(loss1.item())
            for input, target in dataloader_con:
                # 训练连续 AMN：学习连续资源分配比例的重构。
                optimizer2.zero_grad()
                output2 = autoencoder_con(input)
                loss2 = criterion2(output2, target)
                loss2.backward()
                optimizer2.step()
                writer.add_scalar('Loss/Autoencoder_CON', loss2.item(), epoch * len(dataloader_con) + i)
                epoch_con_losses.append(loss2.item())
            loss_rows.append({
                "epoch": epoch + 1,
                "loss_dis": float(np.mean(epoch_dis_losses)),
                "loss_con": float(np.mean(epoch_con_losses)),
            })

        print('Finish Training autoencoder')
        with open(amn_info["loss"], "w", newline="") as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=["epoch", "loss_dis", "loss_con"])
            writer_csv.writeheader()
            writer_csv.writerows(loss_rows)
        torch.save(autoencoder_dis.state_dict(), amn_info["dis"])
        torch.save(autoencoder_con.state_dict(), amn_info["con"])
        print(f"Saved scenario AMN weights to {amn_info['dir']}")
        writer.close()



