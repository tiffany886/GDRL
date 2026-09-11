"""No-RL trajectory baselines B1-B3 (spec §8.1) sharing the same lower layer.

All policies only decide the (K, 2) horizontal UAV velocity each slot; the
environment still runs rule association + ScheduleDAG underneath.
"""
from __future__ import annotations

import numpy as np

from ..env.env import DisasterEnv
from ..env.tasks import APP_PENDING


class Hover:
    name = "hover"

    def act(self, env: DisasterEnv) -> np.ndarray:
        return np.zeros((env.cfg.num_uavs, 2))


class RandomWalk:
    name = "random"

    def __init__(self, rng=None):
        self.rng = np.random.default_rng(0) if rng is None else rng

    def act(self, env: DisasterEnv) -> np.ndarray:
        k = env.cfg.num_uavs
        angle = self.rng.uniform(0.0, 2.0 * np.pi, size=k)
        speed = self.rng.uniform(0.0, env.cfg.uav_speed_max_mps, size=k)
        return np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=1)


class Sweep:
    """Deterministic boustrophedon patrol; one horizontal lane per UAV."""

    name = "sweep"

    def __init__(self):
        self.direction = None

    def act(self, env: DisasterEnv) -> np.ndarray:
        cfg = env.cfg
        k = cfg.num_uavs
        area = cfg.area_size_m
        if self.direction is None:
            self.direction = np.ones(k, dtype=float)
        lane_y = area * (2.0 * np.arange(k) + 1.0) / (2.0 * k)
        actions = np.zeros((k, 2))
        for u in range(k):
            if env.uav_pos[u, 0] >= area - 1.0:
                self.direction[u] = -1.0
            elif env.uav_pos[u, 0] <= 1.0:
                self.direction[u] = 1.0
            actions[u, 0] = cfg.uav_speed_max_mps * self.direction[u]
            actions[u, 1] = np.clip((lane_y[u] - env.uav_pos[u, 1]) * 0.5, -5.0, 5.0)
        return actions


class ChaseBacklog:
    """B2: fly toward the backlog-weighted centroid of own associated terminals."""

    name = "chase"

    def act(self, env: DisasterEnv) -> np.ndarray:
        cfg = env.cfg
        k = cfg.num_uavs
        actions = np.zeros((k, 2))
        for u in range(k):
            target = self._pending_centroid(env, u)
            if target is None:
                continue
            delta = target - env.uav_pos[u]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                continue
            speed = min(cfg.uav_speed_max_mps, dist / max(cfg.slot_seconds, 1e-9))
            actions[u] = delta / dist * speed
        return actions

    @staticmethod
    def _pending_centroid(env: DisasterEnv, uav_k: int):
        cfg = env.cfg
        weights = np.zeros(cfg.num_terminals, dtype=float)
        for n in range(cfg.num_terminals):
            if env.assignment[n] != uav_k:
                continue
            for i in env.terminal_apps[n]:
                if env.apps[i].state == APP_PENDING:
                    weights[n] += 1.0
        total = weights.sum()
        if total <= 0.0:
            return None
        return np.average(env.terminals.pos, axis=0, weights=weights)


def make_policy(name: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    if name == "hover":
        return Hover()
    if name == "random":
        return RandomWalk(rng=rng)
    if name == "sweep":
        return Sweep()
    if name == "chase":
        return ChaseBacklog()
    raise ValueError(f"Unknown baseline {name!r}")

