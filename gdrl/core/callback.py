"""
gdrl/core/callback.py — SB3 训练回调，记录每步 reward/action/latency
=====================================================================
用法：
    callback = CustomCallback()
    model.learn(..., callback=callback)
    actions, rewards, latencys = callback.get_training_data()
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class CustomCallback(BaseCallback):
    """记录每个时间步的 reward、action 和 latency，供 AMN 训练使用。"""

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []
        self.actions = []
        self.latencys = []
        self.energies = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards']
        action = self.locals['actions']
        info = self.locals['infos'][0]
        latency = info.get('latency', None)
        self.rewards.append(reward)
        self.actions.append(action)
        self.latencys.append(latency)
        energy = self.locals['infos'][0].get('energy', 0.0)
        self.energies.append(float(energy))
        return True

    def get_training_data(self):
        """返回 (actions, rewards, latencys, energies) numpy 数组，供 AMN 训练集构造使用。"""
        return np.array(self.actions), np.array(self.rewards), np.array(self.latencys), np.array(self.energies)
