"""Terminal placement and slow drift inside the disaster corridor."""
from __future__ import annotations

import numpy as np

from .config import VAConfig


class Terminals:
    """N terminals, half static / half slow-moving with elastic bounces."""

    def __init__(self, cfg: VAConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        area = cfg.area_size_m
        n = cfg.num_terminals
        self.pos = rng.uniform(0.0, area, size=(n, 2))
        static = rng.random(n) < cfg.static_fraction
        speeds = rng.uniform(0.0, cfg.user_speed_max_mps, size=n)
        speeds = np.where(static, 0.0, speeds)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
        self.vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

    def step(self):
        dt = self.cfg.slot_seconds
        area = self.cfg.area_size_m
        new_pos = self.pos + self.vel * dt
        for axis in range(2):
            lo = new_pos[:, axis] < 0.0
            hi = new_pos[:, axis] > area
            new_pos[lo, axis] = -new_pos[lo, axis]
            new_pos[hi, axis] = 2.0 * area - new_pos[hi, axis]
            self.vel[lo, axis] = np.abs(self.vel[lo, axis])
            self.vel[hi, axis] = -np.abs(self.vel[hi, axis])
        self.pos = np.clip(new_pos, 0.0, area)

