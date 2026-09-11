"""Simplified deterministic LEO visibility windows (spec §4.2).

Full orbit dynamics are out of scope: each LEO gets a deterministic, staggered
list of visible intervals ``[t_in, t_out]`` derived from a period/duty/phase
cycle. Outside any interval the UAV-LEO capacity is a hard zero.
"""
from __future__ import annotations

from .config import VAConfig


class Ephemeris:
    def __init__(self, cfg: VAConfig):
        self.cfg = cfg
        self.horizon_s = cfg.horizon * cfg.slot_seconds
        periods = cfg.leo_window_period_s
        duties = cfg.leo_window_duty
        phases = cfg.leo_window_phase_s
        if not (len(periods) == len(duties) == len(phases) == cfg.num_leos):
            raise ValueError("LEO window arrays must have one entry per LEO")
        self.windows: list[list[tuple[float, float]]] = []
        for l in range(cfg.num_leos):
            period = float(periods[l])
            duty = float(duties[l])
            phase = float(phases[l])
            if period <= 0 or not 0.0 < duty <= 1.0:
                raise ValueError(f"Bad window cycle for LEO {l}: period={period}, duty={duty}")
            win = []
            n = 0
            while True:
                t_in = phase + n * period
                if t_in >= self.horizon_s:
                    break
                t_out = min(t_in + duty * period, self.horizon_s)
                win.append((t_in, t_out))
                n += 1
            self.windows.append(win)

    def visible(self, leo: int, t_s: float) -> bool:
        """True when time ``t_s`` lies inside one of LEO ``leo``'s windows."""
        return any(t_in <= t_s < t_out for t_in, t_out in self.windows[leo])

    def remaining_s(self, leo: int, t_s: float) -> float:
        """Remaining visibility time tau_l (0 when not visible)."""
        best = None
        for t_in, t_out in self.windows[leo]:
            if t_in <= t_s < t_out:
                best = t_out - t_s if best is None else min(best, t_out - t_s)
        return 0.0 if best is None else float(best)

    def any_visible(self, t_s: float) -> bool:
        return any(self.visible(l, t_s) for l in range(self.cfg.num_leos))

