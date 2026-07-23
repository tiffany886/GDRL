"""
gdrl/models/amn.py — Action Mapping Network（AMN / 动作映射网络）
=================================================================
对应论文 Section IV-C：LSTM Autoencoder 将 TRPO 输出的连续潜在动作
映射为结构化的离散（卸载决策）+ 连续（资源分配）动作对。

结构：
  AutoencoderDis  — 离散动作 AE：LSTM Encoder → softmax → 硬离散 + STE 梯度
  AutoencoderCon  — 连续动作 AE：LSTM Encoder → sigmoid → 归一化资源比例

straight-through estimator (STE)：
  训练时通过 soft_action - soft_action.detach() + hard_action 让梯度可以
  回传到 encoder，推理时直接用 argmax 得到硬离散动作。
"""
from torch import nn
import torch
import torch.nn.functional as F


class M1EncoderDis(nn.Module):
    """离散动作编码器：LSTM + FNN → 每个用户的卸载目标 softmax 分布。"""

    def __init__(self, input_dim, output_dim):
        super(M1EncoderDis, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, 128, batch_first=True)
        self.fnn_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, latent_action, user_lists, differentiable=False):
        x = latent_action
        batch_size = x.size(0)
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.view(batch_size, -1)[:, -128:]
        x = self.fnn_net(x)

        # 将 logits 按用户分割并做 softmax 归一化
        x_out = []
        index = 0
        for u in range(len(user_lists)):
            x1 = F.softmax(x[:, index:index + len(user_lists[u])], dim=1)
            index += len(user_lists[u])
            x_out.append(x1)

        all_action = []
        for u, probs in enumerate(x_out):
            action_values = torch.tensor(user_lists[u], device=x.device, dtype=x.dtype)
            max_idx = torch.argmax(probs, dim=1)
            hard_action = action_values[max_idx]
            if differentiable:
                # STE：前向用硬动作，反向用软期望梯度
                soft_action = torch.sum(probs * action_values.unsqueeze(0), dim=1)
                final_action = hard_action.detach() + soft_action - soft_action.detach()
            else:
                final_action = hard_action.to(torch.long)
            all_action.append(final_action)
        return torch.stack(all_action, dim=1)


class M1DecoderDis(nn.Module):
    """离散动作解码器：FNN + LSTM → 重构输入空间（用于 AE 训练）。"""

    def __init__(self, input_dim, output_dim):
        super(M1DecoderDis, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fnn_net = nn.Sequential(
            nn.Linear(self.input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.lstm_out = nn.LSTM(16, self.output_dim, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.float32)
        x = self.fnn_net(x).unsqueeze(1)
        x, _ = self.lstm_out(x)
        return x.view(batch_size, -1)


class AutoencoderDis(nn.Module):
    """离散动作自编码器（训练 AMN 离散分支）。"""

    def __init__(self, input_dim, latent_dim, U, user_lists, Encoder, Decoder):
        super(AutoencoderDis, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(U, input_dim)
        self.user_lists = user_lists

    def forward(self, x):
        # 训练时使用 STE（可微），推理时 encoder 直接返回硬离散动作
        encoded = self.encoder(x, self.user_lists, differentiable=True)
        return self.decoder(encoded)


class M1EncoderCon(nn.Module):
    """连续动作编码器：LSTM + Sigmoid → 归一化的资源分配比例 [U, 2]。"""
    MIN_COMPUTE_SHARE = 0.05   # 计算资源最小份额，防止除零
    MIN_CHANNEL_SHARE = 0.01   # 子信道最小份额

    def __init__(self, input_dim, output_dim):
        super(M1EncoderCon, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.input_dim, 128, batch_first=True)
        self.fnn_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, latent_action):
        x = latent_action
        x, _ = self.lstm(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.fnn_net(x).reshape(x.size(0), -1, 2)

        # 计算资源份额：[MIN, 1.0]，防止 process_time 数值爆炸
        compute_share = self.MIN_COMPUTE_SHARE + (1.0 - self.MIN_COMPUTE_SHARE) * x[:, :, 0]

        # 子信道份额：softmax 归一化后加下限，保证每用户非零
        user_count = x.size(1)
        channel_floor = min(self.MIN_CHANNEL_SHARE, 0.5 / user_count)
        channel_probs = F.softmax(x[:, :, 1], dim=1)
        channel_share = channel_floor + (1.0 - channel_floor * user_count) * channel_probs

        x_out = torch.stack([compute_share, channel_share], dim=2)
        return x_out.squeeze(0) if x_out.size(0) == 1 else x_out


class M1DecoderCon(nn.Module):
    """连续动作解码器：FNN + LSTM → 重构输入（用于 AE 训练）。"""

    def __init__(self, input_dim, output_dim):
        super(M1DecoderCon, self).__init__()
        self.fnn_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.lstm = nn.LSTM(16, output_dim, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.fnn_net(x).unsqueeze(1)
        x, _ = self.lstm(x)
        return x.view(batch_size, -1)


class AutoencoderCon(nn.Module):
    """连续动作自编码器（训练 AMN 连续分支）。"""

    def __init__(self, input_dim, latent_dim, Encoder, Decoder):
        super(AutoencoderCon, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
