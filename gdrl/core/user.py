"""
gdrl/core/user.py — 地面用户节点状态生成
=========================================
用户位于地面 50-100m，初始 VM 实例数为 0（依赖 LEO/HAPS 卸载）。
"""
import numpy as np
import torch


def user_status():
    """随机生成单个用户的位置和 VM 数。"""
    VM_user = 0                                  # 本地无计算资源
    user_place = np.random.randint(50, 100)      # 距基准点距离（米）
    return VM_user, user_place


def all_user_status(U):
    """批量生成 U 个用户的状态，返回 Tensor 供 PyTorch 直接使用。"""
    VM_user, user_place = [], []
    for u in range(U):
        vm, place = user_status()
        VM_user.append(vm)
        user_place.append(place)
    return torch.Tensor(VM_user), torch.Tensor(user_place)
