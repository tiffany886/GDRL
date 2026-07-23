"""
gdrl/envs/rate.py — 无线速率计算（Shannon / URLLC 两种模式）
============================================================
对应论文 Section III-B，支持：
  - eMBB / eMTC（uf=1,2,3）：Shannon 容量公式
  - URLLC（uf=4,5,6）：有限块长近似（URLLC rate）
"""
import numpy as np
from scipy.special import erfcinv


def rate_calculation(pu, hu, uf):
    """
    计算用户 u 在时隙 k 的数据速率。

    参数：
        pu : 用户发射功率（线性，W）
        hu : 信道响应向量（复数 ndarray）
        uf : 用户业务类型（1-6，决定速率公式）

    返回：
        float：速率（bit/s）
    """
    B = 15e3          # 子载波带宽（Hz）
    Tf = 1e-3         # mini-slot 时长（s）
    epsilon_D = 1e-3  # URLLC 解码错误概率
    sigma = 1e-12     # 加性高斯白噪声方差

    snr = pu * np.real(np.conj(hu).T @ hu)[0][0] / sigma ** 2

    if uf in (1, 2, 3):
        # eMBB/eMTC：Shannon 容量
        Ru_k = B * np.log2(1 + snr)
    else:
        # URLLC：有限块长近似（论文 Eq.(7)）
        Ru_k = (B / np.log(2) * (np.log(1 + snr / B)
                - np.sqrt(1 / (Tf * B)) * erfcinv(2 * epsilon_D)))

    return Ru_k
