"""
gdrl/envs/channel.py — GDRL 信道模型
==================================
实现论文 IEEE JSAC 2025 Section III-A 中的三段式信道模型：
  - 地面到 HAPS（毫米波，15 GHz）
  - 地面到 LEO（Sub-6G，2.9 GHz）
  - 两者均采用均匀平面阵列（UPA）天线模型

支持三种运行模式（通过环境变量 GDRL_CHANNEL_MODE 控制）：
  - paper_approx：Python 近似实现，适合快速复现（默认）
  - matlab_p681：调用 MATLAB groundtospace.m，与论文完全一致
  - simple：最简 Rician 近似，仅供调试

随机种子通过环境变量 GDRL_CHANNEL_SEED 控制（默认 73），
保证每次运行信道增益可复现。
"""

import os
import numpy as np
from scipy.constants import speed_of_light as c

try:
    import matlab.engine
except ModuleNotFoundError:
    matlab = None


# ── 全局状态 ─────────────────────────────────────────────────
_MATLAB_ENGINE = None        # MATLAB 引擎单例，延迟初始化
_WARNED = False              # 避免重复打印同一条警告
_CHANNEL_RNG = np.random.default_rng(int(os.environ.get("GDRL_CHANNEL_SEED", "73")))
# 信道模式：paper_approx / matlab_p681 / simple / auto
CHANNEL_MODE = os.environ.get("GDRL_CHANNEL_MODE", "paper_approx").lower()


def _warn_once(message):
    """只打印一次警告，避免每个时隙都刷日志。"""
    global _WARNED
    if not _WARNED:
        print(message)
        _WARNED = True


def _ground_to_space(fk, vertheta):
    """
    计算地面到空间节点（LEO/HAPS）的复数信道增益。

    参数：
        fk       : 载波频率（Hz），LEO=2.9GHz，HAPS=15GHz
        vertheta : 仰角（弧度），论文取 π/6（30°）
    """
    global _MATLAB_ENGINE

    if CHANNEL_MODE == "matlab_p681" and matlab is None:
        raise RuntimeError(
            "GDRL_CHANNEL_MODE=matlab_p681 requires matlab.engine. "
            "Install a MATLAB release whose Python engine supports this Python version."
        )

    # 调用 MATLAB 实现（精确版，符合 ITU-R P.681 建议书）
    if CHANNEL_MODE == "matlab_p681" or (CHANNEL_MODE == "auto" and matlab is not None):
        if _MATLAB_ENGINE is None:
            _MATLAB_ENGINE = matlab.engine.start_matlab()
            _MATLAB_ENGINE.addpath(os.getcwd(), nargout=0)
        return complex(_MATLAB_ENGINE.groundtospace(float(fk), float(vertheta)))

    # 最简 Rician 模型（仅用于调试，不反映真实物理特性）
    if CHANNEL_MODE == "simple":
        _warn_once("Warning: using simple Python channel fallback.")
        rng = np.random.default_rng(73 + int(fk // 1e9))
        k_factor = 10 ** (5.5 / 10)
        los = np.exp(1j * vertheta)
        nlos = (rng.normal() + 1j * rng.normal()) / np.sqrt(2)
        return np.sqrt(k_factor / (k_factor + 1)) * los + np.sqrt(1 / (k_factor + 1)) * nlos

    # 默认：Python 版论文近似模型
    _warn_once("Warning: matlab.engine is unavailable; using Python paper-approx channel model.")
    return _paper_approx_gain(fk, vertheta)


def _paper_approx_gain(fk, vertheta):
    """
    论文 Eq.(1) 的 Python 近似实现：h = w * g

    - w：对数正态阴影衰落，均值 -1.5 dB（<10GHz）或 -3.0 dB，σ=3.8 dB
    - g：Rician 快衰落，K 因子 = 5.5 dB
    """
    k_factor = 10 ** (5.5 / 10)
    shadow_sigma_db = 3.8
    shadow_mean_db = -1.5 if fk < 10e9 else -3.0
    shadow_db = _CHANNEL_RNG.normal(shadow_mean_db, shadow_sigma_db)
    w = 10 ** (shadow_db / 20)                    # dB → 线性幅度

    los = np.exp(1j * vertheta)
    nlos = (_CHANNEL_RNG.normal() + 1j * _CHANNEL_RNG.normal()) / np.sqrt(2)
    g = np.sqrt(k_factor / (k_factor + 1)) * los + np.sqrt(1 / (k_factor + 1)) * nlos
    return w * g


def _array_response(x, y, d_star, fk):
    """
    计算均匀线阵（ULA）的导向矢量（steering vector）。
    返回 shape (y, 1) 的归一化复数向量。
    """
    phase = -1j * 2 * np.pi * fk * d_star / c * np.arange(y) * x
    return (1 / np.sqrt(y)) * np.exp(phase).reshape(-1, 1)


def _upa_response(theta_x, theta_y, mx, my, dx, dy, fk):
    """
    计算均匀平面阵列（UPA）的阵列响应矩阵：A_UPA = a_x ⊗ a_y（论文 Eq.(2)）。
    """
    ax = _array_response(np.sin(theta_y) * np.cos(theta_x), mx, dx, fk)
    ay = _array_response(np.cos(theta_y), my, dy, fk)
    return np.kron(ax, ay)


def _channel_response(gain, distance, fk, total_gain, array_response):
    """
    完整 MIMO 信道响应：h = sqrt(G * λ/(4π*d)) * gain * array_response（论文 Eq.(3)）。
    distance 最小截断为 1.0m，防止数值溢出。
    """
    distance = max(float(distance), 1.0)
    path_loss_amplitude = np.sqrt(total_gain * c / (4 * np.pi * distance * fk))
    return path_loss_amplitude * gain * array_response


def ChannelModel(uf, du):
    """
    主信道模型接口：根据用户服务类型（uf）返回对应信道向量。

    参数：
        uf : 用户业务类型（1/2/5 → LEO Sub-6G；其他 → HAPS mmWave）
        du : 用户到节点的距离（米）
    """
    # ── LEO 链路参数（Sub-6G，2.9 GHz）
    fkl = 2.9e9; lamdal = c / fkl; Gul = 10
    Ml_x, Ml_y = 12, 12; dl_x = dl_y = lamdal

    # ── HAPS 链路参数（毫米波，15 GHz）
    fkn = 15e9; lamdan = c / fkn; Gun = 10
    Mn_x, Mn_y = 6, 6; dn_x = dn_y = lamdan

    # ── 空间角度参数（仰角 30°，水平角 45°）
    vertheta = np.pi / 6
    theta_x  = np.pi / 4
    theta_y  = np.arcsin(np.sin(theta_x) * np.sin(vertheta) / np.cos(vertheta))

    gukl = _ground_to_space(fkl, vertheta)
    gukn = _ground_to_space(fkn, vertheta)

    if uf in (1, 2, 5):
        n_u_l = _upa_response(theta_x, theta_y, Ml_x, Ml_y, dl_x, dl_y, fkl)
        hu = _channel_response(gukl, du, fkl, Gul, n_u_l)
    else:
        n_u_n = _upa_response(theta_x, theta_y, Mn_x, Mn_y, dn_x, dn_y, fkn)
        hu = _channel_response(gukn, du, fkn, Gun, n_u_n)
    return hu


def a_mimo(x, y, d_star, fk):
    """对外暴露的 ULA 导向矢量接口（供 Rate_Calculation 调用）。"""
    return _array_response(x, y, d_star, fk)


def fspl(d, lambda_):
    """自由空间路径损耗（dB）：20*log10(d) + 20*log10(4π/λ)。"""
    return 20 * np.log10(d) + 20 * np.log10(4 * np.pi / lambda_)
