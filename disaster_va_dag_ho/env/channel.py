"""Channel and energy helpers.

Reuses the single source of truth in ``uav_leo_experiment/physics.py``
(LoS/NLoS + Shannon rates and the flight-energy model). The UAV-LEO capacity
is additionally gated by the visibility window and capped by ``C_l``
(spec §4.1: 窗外 C=0 是硬约束).
"""
from __future__ import annotations

import numpy as np
from uav_leo_experiment.physics import (
    flight_energy as _flight_energy,
    rate_uav_leo_vec,
    rate_user_uav_vec,
)

from .config import VAConfig


def terminal_uav_rate_matrix(cfg: VAConfig, terminal_pos, uav_pos) -> np.ndarray:
    """(N, K) terminal->UAV Shannon rates (LoS/NLoS averaged)."""
    terminal_pos = np.asarray(terminal_pos, dtype=float)
    uav_pos = np.asarray(uav_pos, dtype=float)
    return rate_user_uav_vec(cfg, terminal_pos[:, None, :], uav_pos[None, :, :])


def terminal_uav_cover_mask(cfg: VAConfig, terminal_pos, uav_pos) -> np.ndarray:
    """(N, K) bool hard-cover mask on horizontal distance (spec v2 §1.2).

    Terminal n can be served by UAV k only when ||x_n - p_k|| <= R. With
    ``uav_cover_radius_m`` being None or <= 0 (v1 / w/o hard-cover ablation)
    covers every pair so the soft-connection semantics is kept.
    """
    terminal_pos = np.asarray(terminal_pos, dtype=float)
    uav_pos = np.asarray(uav_pos, dtype=float)
    n, k = terminal_pos.shape[0], uav_pos.shape[0]
    mask = np.ones((n, k), dtype=bool)
    if cfg.uav_cover_radius_m is None or float(cfg.uav_cover_radius_m) <= 0.0:
        return mask
    dx = terminal_pos[:, None, 0] - uav_pos[None, :, 0]
    dy = terminal_pos[:, None, 1] - uav_pos[None, :, 1]
    return (dx * dx + dy * dy) <= float(cfg.uav_cover_radius_m) ** 2


def hard_cover_rates(cfg: VAConfig, rates_nk: np.ndarray,
                     terminal_pos, uav_pos) -> np.ndarray:
    """Zero the (n, k) rates that fall outside the hard-cover radius.

    Returns a copy; outside-coverage entries become 0 so they are excluded
    from association candidates and from scheduler uplink feasibility.
    """
    rates = np.asarray(rates_nk, dtype=float).copy()
    if cfg.uav_cover_radius_m is None or float(cfg.uav_cover_radius_m) <= 0.0:
        return rates
    rates[~terminal_uav_cover_mask(cfg, terminal_pos, uav_pos)] = 0.0
    return rates


def uav_leo_channel_rate(cfg: VAConfig, uav_pos, leo_sub_pos) -> np.ndarray:
    """(K, L) raw UAV->LEO channel rates (before window gating / C_l cap)."""
    uav_pos = np.asarray(uav_pos, dtype=float)
    leo_sub_pos = np.asarray(leo_sub_pos, dtype=float)
    return rate_uav_leo_vec(cfg, uav_pos[:, None, :], leo_sub_pos[None, :, :])


def uav_leo_capacity_matrix(cfg: VAConfig, uav_pos, leo_sub_pos, visible) -> np.ndarray:
    """(K, L) usable capacity = min(r_k,l, C_l) inside a window, else hard 0."""
    ch = uav_leo_channel_rate(cfg, uav_pos, leo_sub_pos)
    cap = np.minimum(ch, float(cfg.leo_capacity_bps))
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != cap.shape:
        visible = np.broadcast_to(visible, cap.shape)
    cap[~visible] = 0.0
    return cap


def flight_energy_joules(cfg: VAConfig, moved_m: float) -> float:
    """Propulsion energy (hover + speed term) for one slot of movement."""
    return float(_flight_energy(cfg, float(moved_m)))
