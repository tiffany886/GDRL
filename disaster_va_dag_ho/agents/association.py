"""Rule-based terminal->UAV association (spec §6: 关联不由 RL 输出).

Association is recomputed every slot *after* the UAV move, sorting terminals
by best achievable rate and assigning greedily to the best UAV with a free
capacity slot. Each UAV serves at most ``ceil(N/K)`` terminals.
"""
from __future__ import annotations

from math import ceil

import numpy as np


def associate_terminals(rates_nk: np.ndarray, num_uavs: int) -> np.ndarray:
    """Return length-N assignment array of terminal->UAV indices.

    Only pairs with a strictly positive rate are candidates (spec v2 §1.2:
    hard coverage zeroes rates outside R). A terminal with no feasible UAV
    keeps ``assignment = -1`` instead of being force-assigned.
    """
    rates_nk = np.asarray(rates_nk, dtype=float)
    n_terminals = rates_nk.shape[0]
    cap = ceil(n_terminals / num_uavs)
    counts = np.zeros(num_uavs, dtype=int)
    assignment = np.full(n_terminals, -1, dtype=int)
    order = np.argsort(-rates_nk.max(axis=1), kind="stable")
    for n in order:
        feasible = np.flatnonzero(rates_nk[n] > 0.0)
        if feasible.size == 0:
            continue  # uncovered this slot: assignment stays -1
        feasible = feasible[np.argsort(-rates_nk[n, feasible], kind="stable")]
        for kk in feasible:
            if counts[kk] < cap:
                assignment[n] = int(kk)
                counts[kk] += 1
                break
    return assignment
