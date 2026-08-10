"""
gdrl/core/graph.py — SAGIN 图拓扑生成与离散动作空间编码
=======================================================

GenerateAdjacency(U, L, N) — 原版 3 层图（用户-LEO-HAPS），7-bit 动作编码
GenerateAdjacency_hybrid(U, G, V, L, M) — 新版 5 层图（用户-gNB-UAV-LEO-MEC），12-bit 动作编码

动作编码格式（7 bit，原版）：
  bit[6]: 0=LEO, 1=HAPS
  bit[5]: 0=直连, 1=间接
  bit[4]: 0=非本地, 1=本地卸载
  bit[3:0]: 节点编号（0-15）

动作编码格式（12 bit，hybrid）：
  bit[11:10]: 节点类型 (00=gNB, 01=UAV, 10=LEO, 11=MEC)
  bit[9]:     0=直连, 1=间接
  bit[8]:     0=非本地, 1=本地卸载
  bit[7:0]:   节点编号（0-255）
"""
import numpy as np


def _node_bits(index):
    """把 LEO/HAPS 节点编号编码为 4 bit，当前动作格式最多支持 0-15。"""
    if index < 0 or index > 15:
        raise ValueError("当前 7 bit 离散动作格式最多支持 LEO/HAPS 编号 0-15。")
    return bin(index)[2:].zfill(4)


def _node_bits_8bit(index):
    """把节点编号编码为 8 bit，最多支持 0-255。"""
    if index < 0 or index > 255:
        raise ValueError("当前 12 bit 动作格式最多支持节点编号 0-255。")
    return bin(index)[2:].zfill(8)


# ── 12-bit 动作编码常量 ─────────────────────────────────────────
_TYPE_GNB  = "00"
_TYPE_UAV  = "01"
_TYPE_LEO  = "10"
_TYPE_MEC  = "11"
_LOCAL_BIT = "1"
_NONLOCAL_BIT = "0"
_DIRECT_BIT = "0"
_INDIRECT_BIT = "1"


def _encode_action_12bit(node_type, direct, local, node_idx):
    """
    构造 12-bit 动作字符串。
      node_type : "00"|"01"|"10"|"11"
      direct    : 0=直连, 1=间接
      local     : 0=非本地, 1=本地
      node_idx  : 0-255
    """
    bits = node_type + (_INDIRECT_BIT if direct else _DIRECT_BIT) + \
           (_LOCAL_BIT if local else _NONLOCAL_BIT) + _node_bits_8bit(node_idx)
    return bits


def _decode_action_12bit(action_int):
    """
    解码 12-bit 动作整数 → (node_type, direct, local, node_idx)。
    返回: (type_str, is_indirect, is_local, node_index)
    """
    bits = bin(action_int)[2:].zfill(12)
    node_type = bits[0:2]
    is_indirect = bits[2] == "1"
    is_local = bits[3] == "1"
    node_idx = int(bits[4:12], 2)
    return node_type, is_indirect, is_local, node_idx


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
        # Keep a stable candidate order so AMN logits map to the same actions
        # across runs and Python hash seeds.
        user_lists[u] = sorted(set(binary_u))

    # 二进制字符串 → 十进制整数
    user_lists_new = {u: [int(b, 2) for b in user_lists[u]] for u in range(U)}

    # 所有用户动作总数（GCN 输出维度）
    action_space_len = sum(len(user_lists_new[u]) for u in range(U))

    return action_space_len, Adj_Matrix, user_lists_new


def GenerateAdjacency_hybrid(U, G, V, L, M):
    """
    生成五层异构 SAGIN 图（用户-gNB-UAV-LEO-MEC）和 12-bit 动作空间。

    参数：
        U : 用户数
        G : gNB 数
        V : UAV 数
        L : LEO 卫星数
        M : MEC 中心站数

    返回：
        action_space_len : 所有用户可用动作总数
        Adj_Matrix       : 对称邻接矩阵 [U+G+V+L+M, U+G+V+L+M]
        user_lists_new   : dict {用户索引: [动作十进制列表]}
        node_order       : dict {类型: 起始索引}，用于环境 step() 解析
    """
    total = U + G + V + L + M
    Adj_Matrix = np.zeros((total, total))

    # 节点索引布局：UE[0:U] → gNB[U:U+G] → UAV[U+G:U+G+V] → LEO[U+G+V:U+G+V+L] → MEC[U+G+V+L:U+G+V+L+M]
    off_g = U
    off_v = U + G
    off_l = U + G + V
    off_m = U + G + V + L

    # 每个用户连接 2 个 gNB（直连）
    for u in range(U):
        g1 = u % G
        g2 = (u + 1) % G
        Adj_Matrix[u, off_g + g1] = 1
        Adj_Matrix[u, off_g + g2] = 1

    # 每个用户连接 2 个 UAV（直连）
    for u in range(U):
        v1 = u % V
        v2 = (u + 1) % V
        Adj_Matrix[u, off_v + v1] = 1
        Adj_Matrix[u, off_v + v2] = 1

    # 每个用户连接 2 颗 LEO（直连）
    for u in range(U):
        Adj_Matrix[u, off_l + u % L] = 1
        Adj_Matrix[u, off_l + (u + 1) % L] = 1

    # 每个用户连接 1 个 MEC（直连）
    for u in range(U):
        Adj_Matrix[u, off_m + u % M] = 1

    # gNB 随机连接 1-2 个其他 gNB（模拟光纤互联）
    for g in range(G):
        if G <= 2:
            continue
        others = [x for x in range(G) if x != g]
        n_conn = min(2, len(others))
        conn = np.random.choice(others, n_conn, replace=False)
        for c in conn:
            Adj_Matrix[off_g + g, off_g + c] = 1

    # UAV 随机连接 1-2 个其他 UAV（空中转发）
    for v in range(V):
        if V <= 2:
            continue
        others = [x for x in range(V) if x != v]
        n_conn = min(2, len(others))
        conn = np.random.choice(others, n_conn, replace=False)
        for c in conn:
            Adj_Matrix[off_v + v, off_v + c] = 1

    # LEO 随机星间链路（1-3 颗）
    for l in range(L):
        if L <= 3:
            continue
        others = [x for x in range(L) if x != l]
        n_conn = min(3, len(others))
        conn = np.random.choice(others, n_conn, replace=False)
        for c in conn:
            Adj_Matrix[off_l + l, off_l + c] = 1

    # gNB-MEC 光纤直连
    for g in range(G):
        Adj_Matrix[off_g + g, off_m + g % M] = 1

    # UAV-gNB 连接（空中-地面）
    for v in range(V):
        Adj_Matrix[off_v + v, off_g + v % G] = 1

    # LEO-gNB 连接（卫星-地面）
    for l in range(L):
        Adj_Matrix[off_l + l, off_g + l % G] = 1

    # 对称化
    Adj_Matrix = np.triu(Adj_Matrix, 1) + np.triu(Adj_Matrix, 1).T

    # 构建 12-bit 动作集合
    user_lists = {}
    for u in range(U):
        actions = []

        # 直连 gNB
        g1 = u % G
        g2 = (u + 1) % G
        actions.append(_encode_action_12bit(_TYPE_GNB, 0, 0, g1))
        actions.append(_encode_action_12bit(_TYPE_GNB, 0, 0, g2))

        # 间接 gNB（通过直连 gNB 可达）
        for i in np.where(Adj_Matrix[off_g + g1, off_g:off_g + G] == 1)[0]:
            if i != g1:
                actions.append(_encode_action_12bit(_TYPE_GNB, 1, 0, int(i)))
        for i in np.where(Adj_Matrix[off_g + g2, off_g:off_g + G] == 1)[0]:
            if i != g2:
                actions.append(_encode_action_12bit(_TYPE_GNB, 1, 0, int(i)))

        # 直连 UAV
        v1 = u % V
        v2 = (u + 1) % V
        actions.append(_encode_action_12bit(_TYPE_UAV, 0, 0, v1))
        actions.append(_encode_action_12bit(_TYPE_UAV, 0, 0, v2))

        # 间接 UAV
        for i in np.where(Adj_Matrix[off_v + v1, off_v:off_v + V] == 1)[0]:
            if i != v1:
                actions.append(_encode_action_12bit(_TYPE_UAV, 1, 0, int(i)))

        # 直连 LEO
        l1 = u % L
        l2 = (u + 1) % L
        actions.append(_encode_action_12bit(_TYPE_LEO, 0, 0, l1))
        actions.append(_encode_action_12bit(_TYPE_LEO, 0, 0, l2))

        # 间接 LEO
        for i in np.where(Adj_Matrix[off_l + l1, off_l:off_l + L] == 1)[0]:
            if i != l1:
                actions.append(_encode_action_12bit(_TYPE_LEO, 1, 0, int(i)))

        # 直连 MEC
        m1 = u % M
        actions.append(_encode_action_12bit(_TYPE_MEC, 0, 0, m1))

        # 本地执行
        actions.append(_encode_action_12bit(_TYPE_GNB, 0, 1, 0))  # 本地用 gNB type + local bit

        # Keep a stable candidate order so AMN logits map to the same actions
        # across runs and Python hash seeds.
        user_lists[u] = sorted(set(actions))

    user_lists_new = {u: [int(b, 2) for b in user_lists[u]] for u in range(U)}
    action_space_len = sum(len(user_lists_new[u]) for u in range(U))

    node_order = {
        "UE": 0,
        "gNB": off_g,
        "UAV": off_v,
        "LEO": off_l,
        "MEC": off_m,
    }

    return action_space_len, Adj_Matrix, user_lists_new, node_order


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
