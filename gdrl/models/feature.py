"""
gdrl/models/feature.py — GDRL 图特征提取网络（GFEN）
=====================================================
对应论文 Section IV-B：Graph Feature Extraction Network。

核心思路：
  观测空间（obs）= [普通数值状态] + [图节点特征（每个节点 2 维）]
  - 普通数值状态：通过 FNN（前馈网络）处理
  - 图节点特征：通过 2 层 GCN 提取拓扑依赖关系
  - 两路特征拼接后输出固定维度的特征向量（默认 32 维）

GCN 结构：2 → 16 → 1（节点维度逐层压缩，最终聚合为全局特征）
FNN 结构：var_input_dim → 128 → 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from arg_parser import get_args

args = get_args()


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    # 输入和激活的截断阈值，防止数值爆炸（LEO/HAPS VM 数量可能达数百）
    INPUT_CLIP = 1e6
    ACTIVATION_CLIP = 100.0

    def __init__(self, observation_space, features_dim: int = 32):
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)

        # 图中节点总数 = 用户数 + LEO 数 + HAPS 数
        self.output_dim = args.U + args.L + args.N
        self._L = args.L
        self._N = args.N
        # edge_index.pt 由 main.py 根据邻接矩阵动态生成，保存在项目根目录
        self.edge_index = torch.load('edge_index.pt').long()

        # 观测向量分两段：前半为普通数值状态，后 2*(U+L+N) 为图节点特征
        var_input_dim = observation_space.shape[0] - self.output_dim

        # ── GNN 部分：GATv2Conv（默认）或 GCNConv（--no-use_gat 消融）
        _use_gat = getattr(args, 'use_gat', True)
        _EDGE_DIM = 2          # 静态距离(1) + 动态负载比例(1)
        _GAT_HEADS = 4
        self.node_norm = nn.LayerNorm(2)
        if _use_gat:
            self.gnn1 = GATv2Conv(2, 4, heads=_GAT_HEADS, concat=True,
                                  dropout=0.1, add_self_loops=False,
                                  edge_dim=_EDGE_DIM)
            # gnn1 输出：[B, N, 16]（4 heads × 4 dims）
            self.gat_mid_norm = nn.LayerNorm(_GAT_HEADS * 4)
            self.gnn2 = GATv2Conv(_GAT_HEADS * 4, 1, heads=1, concat=False,
                                  dropout=0.0, add_self_loops=False,
                                  edge_dim=_EDGE_DIM)
        else:
            self.gnn1 = GCNConv(2, 16)
            self.gnn2 = GCNConv(16, 1)
            self.gat_mid_norm = nn.LayerNorm(16)
        self._use_gat = _use_gat
        self._edge_dim = _EDGE_DIM
        self.graph_norm = nn.LayerNorm(self.output_dim)

        # 静态边特征（传播距离），由 main.py 运行时保存
        import os as _os
        _ea_path = 'edge_attr.pt'
        if _os.path.exists(_ea_path):
            self.register_buffer('edge_attr', torch.load(_ea_path).float())
        else:
            self.edge_attr = None

        # ── FNN 前馈网络部分（处理数值型状态）
        self.var_norm = nn.LayerNorm(var_input_dim)
        self.fnn1 = nn.Sequential(
            nn.Linear(var_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # 最终输出层：将两路特征（FNN + GCN）融合为 features_dim 维
        self.fnn_out = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, features_dim), nn.ReLU()
        )

    def _scale_observations(self, observations):
        """
        对数压缩变换：sign(x) * log(1 + |x|)
        消除 LEO/HAPS VM 数量（150-200）与信道增益（1e-8 量级）的数量级差异。
        """
        x = observations.to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=self.INPUT_CLIP, neginf=-self.INPUT_CLIP)
        x = torch.clamp(x, min=-self.INPUT_CLIP, max=self.INPUT_CLIP)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _safe_activations(self, values):
        """中间激活值安全截断，防止 GCN 消息传递中梯度爆炸。"""
        values = torch.nan_to_num(values, nan=0.0,
                                  posinf=self.ACTIVATION_CLIP, neginf=-self.ACTIVATION_CLIP)
        return torch.clamp(values, min=-self.ACTIVATION_CLIP, max=self.ACTIVATION_CLIP)

    def _build_edge_attr(self, scaled_obs, edge_index, batch_size):
        """
        构建 [B, E, 2] 边特征：
          dim-0: 静态归一化距离（edge_attr.pt）
          dim-1: 动态目标节点负载比例（从 log-scale 观测提取）

        如果 edge_attr.pt 未加载，返回 None。
        """
        if self.edge_attr is None:
            return None

        E = self.edge_attr.shape[0]
        device = scaled_obs.device
        U = self.output_dim - self._L - self._N
        L = self._L
        N = self._N

        # 静态边特征 [E, 1] -> [B, E, 1]
        static_ea = self.edge_attr.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        # 动态特征：目标节点当前负载（log-scale 下的代理值）
        var_state = scaled_obs[:, :-2 * self.output_dim]   # [B, var_dim]
        cl_curr = var_state[:, 6 * U: 6 * U + L]          # [B, L]
        cn_curr = var_state[:, 6 * U + L: 6 * U + L + N]  # [B, N]

        con_state_raw = scaled_obs[:, -2 * self.output_dim:].reshape(
            batch_size, self.output_dim, 2)
        cl_ori = con_state_raw[:, U: U + L, 0]            # [B, L]
        cn_ori = con_state_raw[:, U + L: U + L + N, 0]    # [B, N]

        leo_load = cl_curr / (cl_ori.abs() + 1e-6)        # [B, L]
        haps_load = cn_curr / (cn_ori.abs() + 1e-6)       # [B, N]
        ue_load = torch.zeros(batch_size, U, device=device)
        node_load = torch.cat([ue_load, leo_load, haps_load], dim=1)  # [B, U+L+N]

        dst = edge_index[1]                                 # [E]
        dynamic_ea = node_load[:, dst].unsqueeze(2)        # [B, E, 1]

        return torch.cat([static_ea, dynamic_ea], dim=2)   # [B, E, 2]

    def forward(self, observations):
        """
        GFEN 前向传播：双路特征提取 → 融合 → 输出 features_dim 维特征向量。

        数据流：
          observations [B, obs_dim]
              ├─ 前段（普通数值）→ FNN → [B, 64]
              └─ 后段（图节点特征）reshape → [B, U+L+N, 2]
                      → GCN layer1 [B, U+L+N, 16]
                      → GCN layer2 [B, U+L+N, 1] squeeze → [B, U+L+N]
          拼接 → [B, var_input_dim + (U+L+N)]
              → LayerNorm → fnn1 → [B, 64]
              → fnn_out → [B, features_dim=32]
        """
        # 1. 对数压缩观测值，消除量级差异
        x = self._scale_observations(observations)
        edge_index = self.edge_index.to(x.device)
        batch_size = x.size(0)

        # 2. 分离两路输入
        var_state = x[:, :-2 * self.output_dim]   # 普通数值状态
        con_state = x[:, -2 * self.output_dim:].reshape(batch_size, self.output_dim, 2)

        # 3. GNN 图特征提取路径（GAT 或 GCN，由 _use_gat 控制）
        con_state = self.node_norm(con_state)
        edge_attr = self._build_edge_attr(x, edge_index, batch_size)
        if self._use_gat:
            # GATv2Conv 需要 2D 输入 [N, F]，用 PyG 标准批图格式展开：
            #   节点: [B, N, F] → [B*N, F]
            #   边:   [2, E]    → [2, B*E]（每批次偏移 N）
            #   edge_attr: [B, E, 2] → [B*E, 2]
            num_nodes = self.output_dim
            num_edges = edge_index.shape[1]
            device = x.device

            x_flat = con_state.reshape(batch_size * num_nodes, -1)          # [B*N, 2]

            offsets = (torch.arange(batch_size, device=device) * num_nodes
                       ).view(batch_size, 1, 1)                              # [B, 1, 1]
            edge_index_b = (edge_index.unsqueeze(0).expand(batch_size, -1, -1)
                            + offsets).permute(1, 0, 2).reshape(2, -1)      # [2, B*E]

            ea_flat = (edge_attr.reshape(batch_size * num_edges, -1)
                       if edge_attr is not None else None)                   # [B*E, 2] | None

            con_output = self.gnn1(x_flat, edge_index_b, ea_flat)           # [B*N, 16]
            con_output = F.elu(self._safe_activations(con_output))
            con_output = self.gat_mid_norm(con_output)
            con_output = self.gnn2(con_output, edge_index_b, ea_flat)       # [B*N, 1]
            con_output = con_output.reshape(batch_size, num_nodes)          # [B, N]
        else:
            con_output = self.gnn1(con_state, edge_index)                # 2->16
            con_output = F.relu(self._safe_activations(con_output))
            con_output = F.dropout(con_output, training=self.training)
            con_output = self.gnn2(con_output, edge_index).squeeze(2)    # 16->1
        con_output = self.graph_norm(self._safe_activations(con_output))

        # 4. 融合两路特征 → FNN 进一步提取
        var_input = torch.cat((var_state, con_output), dim=1)
        var_input = self.var_norm(var_input)
        fnn_input = self.fnn1(var_input)

        # 5. 最终输出，安全截断防止后续 TRPO 策略更新数值不稳定
        return self._safe_activations(self.fnn_out(fnn_input))
