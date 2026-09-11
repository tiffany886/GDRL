"""Task arrival process (spec §3.1).

Unit of work = one application request. The concrete DAG templates and the
graph object live in ``scheduler/dag.py`` (W2); until then ``dag`` stays None
and the arrival/deadline/concurrency bookkeeping below is already real.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import VAConfig

APP_PENDING = "pending"
APP_DONE = "done"
APP_FAILED = "failed"


@dataclass
class AppRequest:
    app_id: int
    terminal: int
    arrival_s: float
    deadline_s: float
    template_id: int = 0
    dag: Optional[object] = None  # scheduler.dag.DAG (filled from W2)
    state: str = APP_PENDING
    completion_s: Optional[float] = None
    suffered_uncovered: bool = False  # v2 rule A: pending while out of UAV coverage


class ArrivalProcess:
    """Non-homogeneous Poisson arrivals, one rate per terminal per second."""

    def __init__(self, cfg: VAConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def rate_per_s(self, t_s: float) -> float:
        if self.cfg.burst_start_s <= t_s < self.cfg.burst_end_s:
            return self.cfg.lambda_high_per_s
        return self.cfg.lambda_low_per_s

    def _hotspot_centers(self) -> np.ndarray:
        """(M, 2) hotspot centers; deterministic defaults when not configured."""
        cfg = self.cfg
        if not cfg.hotspot_arrivals:
            return np.zeros((0, 2), dtype=float)
        area = float(cfg.area_size_m)
        if cfg.hotspot_centers is not None:
            flat = np.asarray(cfg.hotspot_centers, dtype=float).ravel()
            if flat.size != 2 * cfg.num_hotspots:
                raise ValueError(
                    f"hotspot_centers needs {2 * cfg.num_hotspots} coords for "
                    f"{cfg.num_hotspots} hotspots, got {flat.size}"
                )
            return flat.reshape(cfg.num_hotspots, 2)
        m = int(cfg.num_hotspots)
        if m == 2:
            return np.array([[0.35, 0.35], [0.65, 0.65]], dtype=float) * area
        fracs = np.linspace(0.25, 0.75, m)
        return np.stack([fracs * area, np.full(m, 0.5 * area)], axis=1)

    def in_burst(self, t_s: float) -> bool:
        return self.cfg.burst_start_s <= t_s < self.cfg.burst_end_s

    def rate_per_terminal(self, t_s: float, terminal_pos) -> np.ndarray:
        """Per-terminal arrival rate (spatial inhomogeneous, spec v2 §1.4).

        With ``hotspot_arrivals=True``: inside any hotspot the rate is
        ``lambda_high_per_s``, outside it is ``lambda_low_per_s``; the burst
        window [burst_start, burst_end) acts as an optional global multiplier.
        With ``hotspot_arrivals=False`` the v1 uniform burst semantics is kept.
        """
        cfg = self.cfg
        pos = np.asarray(terminal_pos, dtype=float)
        n = pos.shape[0]
        if not cfg.hotspot_arrivals:
            lam = np.full(n, self.rate_per_s(t_s), dtype=float)
        else:
            centers = self._hotspot_centers()
            inside = np.zeros(n, dtype=bool)
            for c in centers:
                d2 = (pos[:, 0] - c[0]) ** 2 + (pos[:, 1] - c[1]) ** 2
                inside |= d2 <= float(cfg.hotspot_radius_m) ** 2
            lam = np.where(inside, cfg.lambda_high_per_s, cfg.lambda_low_per_s).astype(float)
        if self.in_burst(t_s):
            lam = lam * float(cfg.burst_multiplier)
        return lam

    def sample_terminals(self, t_s: float, pending_by_terminal: np.ndarray,
                         terminal_pos=None) -> list:
        """Return terminal indices with a new arrival at time ``t_s``.

        Arrivals are suppressed for terminals already at the concurrency cap.
        """
        if terminal_pos is not None:
            lam = self.rate_per_terminal(t_s, terminal_pos)
        else:
            lam = np.full(self.cfg.num_terminals, self.rate_per_s(t_s), dtype=float)
        prob = np.minimum(lam * self.cfg.slot_seconds, 1.0)
        hits = []
        for n in range(self.cfg.num_terminals):
            if pending_by_terminal[n] >= self.cfg.terminal_concurrency_max:
                continue
            if self.rng.random() < prob[n]:
                hits.append(n)
        return hits
