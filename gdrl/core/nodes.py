"""
gdrl/core/nodes.py — 四层异构网络节点状态生成
============================================
LEO 卫星：VM 150-200，轨道高度 300-500 km
UAV（无人机）：VM 80-120，高度 100m-5km（替代原 HAPS）
gNB（地面基站）：VM 200-500，高算力低时延固定节点
MEC Server（大型中心站）：VM 800-1500，光纤互联汇聚节点
"""
import numpy as np


def LEO_status():
    """随机生成单颗 LEO 卫星的初始状态。"""
    C_l = np.random.randint(150, 200)
    LEO_place = np.random.randint(300e3, 500e3 + 1)
    return C_l, LEO_place


def all_LEO_status(L):
    """批量生成 L 颗 LEO 卫星的状态向量。"""
    C_l, LEO, C_l_ori = [], [], []
    for l in range(L):
        cl, leo = LEO_status()
        C_l.append(cl)
        LEO.append(leo)
        C_l_ori.append(cl)
    return np.array(C_l), np.array(LEO), np.array(C_l_ori)


def UAV_status():
    """无人机（UAV）初始状态。中等算力、可移动、低空。"""
    C_v = np.random.randint(80, 120)
    UAV_place = np.random.randint(100, 5000)
    return C_v, UAV_place


def all_UAV_status(V):
    """批量生成 V 个 UAV 状态。"""
    C_v, UAV_place, C_v_ori = [], [], []
    for _ in range(V):
        cv, up = UAV_status()
        C_v.append(cv)
        UAV_place.append(up)
        C_v_ori.append(cv)
    return np.array(C_v), np.array(UAV_place), np.array(C_v_ori)


def gNB_status():
    """地面基站（gNB）初始状态。高算力、低时延、固定位置。"""
    C_g = np.random.randint(200, 500)
    gNB_place = 10
    return C_g, gNB_place


def all_gNB_status(G):
    """批量生成 G 个 gNB 的状态。"""
    C_g, gNB_place, C_g_ori = [], [], []
    for _ in range(G):
        cg, gp = gNB_status()
        C_g.append(cg)
        gNB_place.append(gp)
        C_g_ori.append(cg)
    return np.array(C_g), np.array(gNB_place), np.array(C_g_ori)


def MEC_status():
    """大型地面中心站（MEC）。超高算力，通过光纤连接 gNB。"""
    C_m = np.random.randint(800, 1500)
    MEC_place = 5
    return C_m, MEC_place


def all_MEC_status(M):
    """批量生成 M 个 MEC 中心站状态。"""
    C_m, MEC_place, C_m_ori = [], [], []
    for _ in range(M):
        cm, mp = MEC_status()
        C_m.append(cm)
        MEC_place.append(mp)
        C_m_ori.append(cm)
    return np.array(C_m), np.array(MEC_place), np.array(C_m_ori)


# ── 向后兼容别名（旧代码仍可用 HAPS_status 等名称） ────────────
HAPS_status = UAV_status
all_HAPS_status = all_UAV_status
