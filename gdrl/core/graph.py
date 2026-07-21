"""
gdrl/core/graph.py — SAGIN 图拓扑生成与离散动作空间编码
=======================================================
GenerateAdjacency(U, L, N) 生成：
  1. 邻接矩阵 Adj_Matrix [U+L+N, U+L+N]：用户-LEO-HAPS 三层图
  2. user_lists：每个用户可达节点的 7-bit 动作编码（十进制）
  3. action_space_len：所有用户可用动作总数

动作编码格式（7 bit）：
  bit[6]: 0=LEO, 1=HAPS
  bit[5]: 0=直连, 1=间接
  bit[4]: 0=非本地, 1=本地卸载
  bit[3:0]: 节点编号（0-15）
"""
import numpy as np


def _node_bits(index):
    """把 LEO/HAPS 节点编号编码为 4 bit，当前动作格式最多支持 0-15。"""
    if index < 0 or index > 15:
        raise ValueError("当前 7 bit 离散动作格式最多支持 LEO/HAPS 编号 0-15。")
    return bin(index)[2:].zfill(4)


def GenerateAdjacency(U, L, N):
    """
    生成 SAGIN 三层图的邻接矩阵和用户动作空间。

    参数：
        U : 用户数
        L : LEO 卫星数
        N : HAPS 平台数

    返回：
        action_space_len : 所有用户可用动作总数（int）
        Adj_Matrix       : 对称邻接矩阵 [U+L+N, U+L+N]
        user_lists_new   : dict {用户索引: [动作十进制列表]}
    """
    Adj_Matrix = np.zeros((U + L + N, U + L + N))

    # 每个用户直接连接两颗 LEO（user u → LEO u, LEO u+1）
    for u in range(U):
        Adj_Matrix[u, U + u] = 1
        Adj_Matrix[u, U + u + 1] = 1

    # 接入 LEO 随机连接 1-3 颗非接入 LEO（模拟星间链路）
    for u in range(U, U + U + 1):
        numbers_LEO = np.arange(U + U + 1, U + L)
        number_connection_L = np.random.choice(np.arange(1, 4), 1, replace=False)[0]
        index = np.random.choice(numbers_LEO, number_connection_L, replace=False)
        for i in range(number_connection_L):
            Adj_Matrix[u, index[i]] = 1

    # 每个用户直接连接两个 HAPS（user u → HAPS u, HAPS u+1）
    for u in range(U):
        Adj_Matrix[u, U + L + u] = 1
        Adj_Matrix[u, U + L + u + 1] = 1

    # 接入 HAPS 随机连接 1-3 个非接入 HAPS（模拟平流层平台间通信）
    for u in range(U + L, U + U + L + 1):
        numbers_HAPS = np.arange(U + U + L + 1, U + L + N)
        number_connection_N = np.random.choice(np.arange(1, 4), 1, replace=False)[0]
        index_n = np.random.choice(numbers_HAPS, number_connection_N, replace=False)
        for i in range(number_connection_N):
            Adj_Matrix[u, index_n[i]] = 1

    # 对称化邻接矩阵（无向图）
    Adj_Matrix = np.triu(Adj_Matrix, 1) + np.triu(Adj_Matrix, 1).T

    # 构建每个用户的可达动作集合（7-bit 二进制 → 十进制）
    user_lists = {}
    for u in range(U):
        binary_u = []
        # 直连 LEO（本地 access LEO）
        binary_u.extend([bin(u)[2:].zfill(7), bin(u + 1)[2:].zfill(7)])
        # 间接 LEO（通过 access LEO u 可达）
        for i in np.where(Adj_Matrix[u + U, U:U + L] == 1)[0]:
            binary_u.append("010" + _node_bits(i))
        for i in np.where(Adj_Matrix[u + U + 1, U:U + L] == 1)[0]:
            binary_u.append("010" + _node_bits(i))
        # 直连 HAPS
        binary_u.extend(["100" + _node_bits(u), "100" + _node_bits(u + 1)])
        # 间接 HAPS（通过 access HAPS u 可达）
        for i in np.where(Adj_Matrix[u + U + L, U + L:U + L + N] == 1)[0]:
            binary_u.append("110" + _node_bits(i))
        for i in np.where(Adj_Matrix[u + U + L + 1, U + L:U + L + N] == 1)[0]:
            binary_u.append("110" + _node_bits(i))
        binary_u.append("0010000")   # 本地执行选项
        user_lists[u] = list(set(binary_u))

    # 二进制字符串 → 十进制整数
    user_lists_new = {u: [int(b, 2) for b in user_lists[u]] for u in range(U)}

    # 所有用户动作总数（GCN 输出维度）
    action_space_len = sum(len(user_lists_new[u]) for u in range(U))

    return action_space_len, Adj_Matrix, user_lists_new


def compute_edge_attr(edge_index_np, U_place, LEO_place, HAPS_place, U, L, N):
    """
    为每条有向边计算归一化节点间距离（传播延迟代理），shape [E, 1]。

    参数：
        edge_index_np : np.ndarray [2, E]，有向边 (src, dst) 列表
        U_place       : array-like [U]，用户 1D 坐标
        LEO_place     : array-like [L]，LEO 卫星 1D 坐标
        HAPS_place    : array-like [N]，HAPS 平台 1D 坐标
        U, L, N       : 各层节点数

    返回：
        torch.FloatTensor [E, 1]，归一化距离，范围 [0, 1]
    """
    import numpy as np
    import torch

    all_pos = np.concatenate([
        np.asarray(U_place, dtype=float).reshape(-1),
        np.asarray(LEO_place, dtype=float).reshape(-1),
        np.asarray(HAPS_place, dtype=float).reshape(-1),
    ])  # [U+L+N]

    src = edge_index_np[0]
    dst = edge_index_np[1]
    dist = np.abs(all_pos[src] - all_pos[dst]).astype(np.float32)  # [E]

    max_dist = float(dist.max()) if dist.max() > 0 else 1.0
    dist_norm = (dist / max_dist).reshape(-1, 1)  # [E, 1]

    return torch.tensor(dist_norm, dtype=torch.float32)
