"""
gdrl/core/env_init.py — SAGIN 环境初始化（每个 episode 重置）
=============================================================
ResetFunction 在每个 episode 开始时调用，生成所有节点的初始状态：
  LEO 状态、HAPS 状态、资源矩阵、用户状态、子信道数。
"""
import numpy as np
import torch

from gdrl.core.nodes import all_HAPS_status, all_LEO_status
from gdrl.core.user import all_user_status


def ResetFunction(N, L, T, U):
    """
    初始化环境状态，返回所有节点的状态字典。

    参数：
        N : HAPS 平台数
        L : LEO 卫星数
        T : 时隙总数（episode 长度）
        U : 用户数

    返回：
        LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
        user_status, A_u, A_u_ori, A_resource
    """
    # 生成 LEO 卫星初始状态（L 颗）
    C_l, LEO_place, C_l_ori = all_LEO_status(L)
    LEO_status = {'C_l': C_l, 'LEO_place': LEO_place, 'C_l_ori': C_l_ori}

    # 生成 HAPS 平台初始状态（N 个）
    C_n, HAPS_place, C_n_ori = all_HAPS_status(N)
    HAPS_status = {'C_n': C_n, 'HAPS_place': HAPS_place, 'C_n_ori': C_n_ori}

    # 资源时间矩阵（追踪每个时隙的资源归还）
    LEO_resource_status = {'C_resource_l': np.zeros((L, T))}
    HAPS_resource_status = {'C_resource_n': np.zeros((N, T))}
    A_resource = np.zeros((T))                # 子信道资源归还时序

    # 生成用户状态
    C_u_ori, U_place = all_user_status(U)
    user_status = {'C_u_ori': C_u_ori, 'U_place': U_place}

    # 初始子信道数（论文设定 30 个子信道）
    A_u = 30
    A_u_ori = 30

    return (LEO_status, HAPS_status, LEO_resource_status, HAPS_resource_status,
            user_status, A_u, A_u_ori, A_resource)
