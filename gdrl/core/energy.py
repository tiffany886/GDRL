"""
gdrl/core/energy.py — 能耗计算模型
====================================
传输能耗：E_tx = P_u × (Su / R_u)
计算能耗：E_comp = P_node_per_vm × allocated_vms × t_process
本地能耗：E_local = P_local × t_process
能效：EE = Su / (E_tx + E_comp)  [bits/J]

功耗参考值（可通过环境变量覆盖）：
  GDRL_P_LEO   = 50 W/VM  （LEO 卫星边缘节点）
  GDRL_P_HAPS  = 30 W/VM  （HAPS 平台节点）
  GDRL_P_GNB   = 40 W/VM  （地面基站 gNB）
  GDRL_P_UAV   = 30 W/VM  （无人机 UAV）
  GDRL_P_MEC   = 20 W/VM  （边缘数据中心 MEC）
  GDRL_P_LOCAL =  2 W      （用户手机 CPU）
"""
import os

P_LEO   = float(os.environ.get("GDRL_P_LEO",   "50.0"))  # W per VM
P_HAPS  = float(os.environ.get("GDRL_P_HAPS",  "30.0"))  # W per VM
P_GNB   = float(os.environ.get("GDRL_P_GNB",   "40.0"))  # W per VM
P_UAV   = float(os.environ.get("GDRL_P_UAV",   "30.0"))  # W per VM
P_MEC   = float(os.environ.get("GDRL_P_MEC",   "20.0"))  # W per VM
P_LOCAL = float(os.environ.get("GDRL_P_LOCAL",  "2.0"))  # W
MAX_ENERGY_J = 1.0   # 单用户单时隙能耗截断上限（焦耳）


def tx_energy(power_w: float, data_bits: float, rate_bps: float) -> float:
    """传输能耗（J）= 发射功率 × 传输时间。"""
    if rate_bps <= 0:
        return MAX_ENERGY_J
    e = max(float(power_w), 0.0) * (max(float(data_bits), 0.0) / rate_bps)
    return min(e, MAX_ENERGY_J)


def comp_energy(node_type: str, allocated_vms: float, process_time_s: float) -> float:
    """
    计算能耗（J）= 节点单VM功耗 × 分配VM数 × 处理时间。

    参数：
        node_type      : "leo" | "haps" | "local"
        allocated_vms  : 分配给该用户的 VM 数量
        process_time_s : 处理时间（秒）
    """
    if node_type == "leo":
        p_per_vm = P_LEO
    elif node_type in ("haps", "uav"):
        p_per_vm = P_UAV
    elif node_type == "gnb":
        p_per_vm = P_GNB
    elif node_type == "mec":
        p_per_vm = P_MEC
    elif node_type == "local":
        p_per_vm = P_LOCAL
    else:
        raise ValueError(f"Unknown node_type: {node_type!r}. Expected 'leo', 'haps', 'uav', 'gnb', 'mec', or 'local'.")
    e = p_per_vm * max(float(allocated_vms), 0.0) * max(float(process_time_s), 0.0)
    return min(e, MAX_ENERGY_J)


def local_energy(process_time_s: float) -> float:
    """本地执行能耗（J）= 手机CPU功耗 × 本地处理时间。"""
    e = P_LOCAL * max(float(process_time_s), 0.0)
    return min(e, MAX_ENERGY_J)


def energy_efficiency(data_bits: float, total_energy_j: float) -> float:
    """能效（bits/J）= 处理数据量 / 总能耗。能耗为0时返回0。"""
    if total_energy_j <= 0:
        return 0.0
    return float(data_bits) / total_energy_j
