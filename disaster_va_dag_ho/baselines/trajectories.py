"""Non-learning UAV trajectory module (spec v2 §1.5).

v2 headline: the UAV motion is a *fixed* rule trajectory; nothing here is
learned. The main method (Ours) and B1-B3/B6 all pick one of these policies,
and the flat-RL rows B4/B5 keep exactly the same motion (T-patrol) while
learning only the offload decisions. MAPPO trajectory code stays in the repo
but is not run for the v2 main table.
"""
from __future__ import annotations

import numpy as np

from ..env.env import DisasterEnv
from .no_rl import ChaseBacklog, Hover, RandomWalk, Sweep


class TPatrol:
    """T-patrol (headline v2 trajectory): per-UAV corridor strip patrol.

    The corridor is split into ``K`` x-bands of width ``area/K``; UAV ``u``
    keeps its y on lane ``area*(2u+1)/(2K)`` and oscillates back and forth
    inside its own band at ``uav_speed_max_mps``, flipping at the band edges.
    Deterministic and stateless across episodes.
    """

    name = "patrol"

    def __init__(self):
        self._dir = None

    def act(self, env: DisasterEnv) -> np.ndarray:
        cfg = env.cfg
        k = cfg.num_uavs
        area = float(cfg.area_size_m)
        speed = float(cfg.uav_speed_max_mps)
        if self._dir is None or len(self._dir) != k:
            self._dir = np.ones(k, dtype=float)
        lane_y = area * (2.0 * np.arange(k) + 1.0) / (2.0 * k)
        band_w = area / float(k)
        actions = np.zeros((k, 2))
        for u in range(k):
            lo = u * band_w
            hi = min(area, (u + 1) * band_w)
            x = float(env.uav_pos[u, 0])
            if self._dir[u] > 0.0 and x + speed > hi - 1e-9:
                self._dir[u] = -1.0
            elif self._dir[u] < 0.0 and x - speed < lo + 1e-9:
                self._dir[u] = 1.0
            actions[u, 0] = speed * self._dir[u]
            actions[u, 1] = np.clip((lane_y[u] - env.uav_pos[u, 1]) * 0.5, -5.0, 5.0)
        return actions


def make_trajectory(name: str, seed: int = 0):
    """Return the fixed-trajectory policy named by ``name`` (act(env)->(K,2))."""
    rng = np.random.default_rng(seed)
    if name == "patrol":
        return TPatrol()
    if name == "hover":
        return Hover()
    if name == "sweep":
        return Sweep()
    if name == "chase":
        return ChaseBacklog()
    if name == "random":
        return RandomWalk(rng=rng)
    raise ValueError(f"Unknown trajectory {name!r}")
