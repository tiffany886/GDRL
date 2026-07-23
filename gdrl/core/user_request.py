"""
gdrl/core/user_request.py — 用户业务请求特征生成
================================================
每个用户每个时隙生成 6 维特征：
  uf  : 业务类型（1-6）
  pu  : 发射功率（线性，W）
  s_u : 任务大小（bit）
  o_u : CPU 需求（cycle/bit）
  v_u : 计算结果大小（bit）
  l_u : 时延约束（ms）

uf=1/2/5 → 低时延类（LEO Sub-6G）；其余 → 高带宽类（HAPS mmWave）
"""
import numpy as np


def all_user_feature(U, T):
    """生成 (U, T) 形状的所有用户请求特征矩阵。"""
    Uf, Pu, Su, Ou, Vu, Lu = [], [], [], [], [], []
    for u in range(U):
        Uft, Put, Sut, Out, Vut, Lut = [], [], [], [], [], []
        for t in range(T):
            uf, pu, s_u, o_u, v_u, l_u = user_feature(t)
            Uft.append(uf); Put.append(pu); Sut.append(s_u)
            Out.append(o_u); Vut.append(v_u); Lut.append(l_u)
        Uf.append(Uft); Pu.append(Put); Su.append(Sut)
        Ou.append(Out); Vu.append(Vut); Lu.append(Lut)
    return np.array(Uf), np.array(Pu), np.array(Su), np.array(Ou), np.array(Vu), np.array(Lu)


def user_feature(t):
    """生成单个时隙的用户特征。每 10 个时隙允许生成所有类型（含类型 1/2）。"""
    # 业务类型：每 10 个时隙切换一次低时延业务
    if t % 10 == 0:
        uf = np.random.randint(1, 7)
    else:
        uf = np.random.randint(3, 7)

    pu_dBW = np.random.randint(10, 26)
    pu = 10 ** (pu_dBW / 10)          # dBW → 线性（W）

    # 任务大小：eMBB/eMTC 大包（9000-10000 bit），URLLC 小包（200-300 bit）
    s_u = np.random.randint(9000, 10001) if uf in {1, 2, 3} else np.random.randint(200, 301)

    o_u = np.random.randint(5000, 15001)   # CPU 需求（cycle/bit）

    # 计算结果：eMBB 大（5000-6000 bit），URLLC 小（100-150 bit）
    v_u = np.random.randint(5000, 6001) if uf in {1, 2, 3} else np.random.randint(100, 151)

    # 时延约束：eMBB 宽松（30-100 ms），URLLC 严格（5-10 ms）
    l_u = np.random.randint(30, 100) if uf in {1, 2, 3} else np.random.randint(5, 11)

    return uf, pu, s_u, o_u, v_u, l_u
