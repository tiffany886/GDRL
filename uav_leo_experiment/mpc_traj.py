"""Receding-horizon (MPC) trajectory planner for the UAV-LEO offloading problem.

Motivated by the MEC / vehicular-edge literature that separates the UAV
trajectory problem from the offloading problem and solves the trajectory with
model predictive control / receding-horizon optimization.  At every slot the
planner:

1. builds a small set of candidate first moves (toward the K-step-ahead
   hotspot, the predicted workload centroid, a blend, the top workload user,
   hover) at two speed levels;
2. for each candidate it simulates ``horizon`` slots forward: users and the
   hotspot move according to their observed velocities (the hotspot bounces on
   its road), and at each future position the per-slot cost is computed with
   the *same* exact offloading solver used by GDRL (and the same physics as
   the environment).  Future task arrivals are sampled from the environment's
   hotspot-aware arrival model so that being near the *future* hotspot is
   valued correctly;
3. it picks the candidate with the lowest sum of latency + energy + drop
   penalty and executes only the first move (receding horizon).

``MPCTrajPolicy`` pairs the MPC trajectory with the exact offloading solver
(``offload="exact"``) or with the queue-balanced TEA heuristic
(``offload="tea"``), so the trajectory value can be isolated from the
offloading value.

Reference [8]:
  Y. Zhang, Z. Kuang, Y. Feng, and F. Hou, "Task Offloading and Trajectory
  Optimization for Secure Communications in Dynamic User Multi-UAV MEC
  Systems," IEEE Transactions on Mobile Computing, vol. 23, no. 12,
  pp. 14427-14440, 2024. DOI: 10.1109/TMC.2024.3442909.
  -- receding-horizon / MPC trajectory planning separated from offloading.
"""
import numpy as np

from .baselines import BasePolicy
from .physics import flight_energy, simulate_task

_RATIO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


def predict_hotspot(obs, config, k):
    """Predicted hotspot position k slots ahead (road bounce preserved)."""
    if obs.get("hotspot_pos") is None:
        return None
    vel = np.asarray(obs["hotspot_vel"], dtype=float)
    speed = float(np.linalg.norm(vel))
    if speed <= 1.0e-9:
        return np.asarray(obs["hotspot_pos"], dtype=float).copy()
    dir_vec = vel / speed
    center = np.array([0.5 * config.area_size, 0.5 * config.area_size])
    along = float(np.dot(np.asarray(obs["hotspot_pos"], dtype=float) - center, dir_vec))
    signed_speed = float(np.dot(vel, dir_vec))  # +/- speed along the road
    along_k = along + signed_speed * config.slot_seconds * k
    half = float(config.road_half_len)
    span = 4.0 * half
    x = (along_k + half) % span
    if x > 2.0 * half:
        x = span - x
    along_k = x - half
    return center + dir_vec * along_k


def predict_users(obs, config, k):
    """Predicted user positions k slots ahead (linear mobility, clipped)."""
    vel = np.asarray(obs.get("user_vel", np.zeros((len(obs["user_pos"]), 2))), dtype=float)
    pos = np.asarray(obs["user_pos"], dtype=float) + vel * config.slot_seconds * k
    return np.clip(pos, 0.0, config.area_size)


def _sample_future_tasks(obs, config, pusers, p_hot, rng):
    """Sample one task instantiation at the predicted future slot, matching
    the environment's hotspot-aware arrival / size model."""
    U = len(pusers)
    bits = np.zeros(U, dtype=float)
    cpb = np.zeros(U, dtype=float)
    if config.hotspot_motion and p_hot is not None:
        d2 = np.sum((pusers - p_hot) ** 2, axis=1)
        proximity = np.exp(-d2 / (2.0 * config.hotspot_radius ** 2))
        p_arr = config.base_arrival_prob + (config.hotspot_arrival_prob - config.base_arrival_prob) * proximity
        arrival = rng.uniform(size=U) < p_arr
        size_boost = 1.0 + config.hotspot_size_boost * proximity
    else:
        arrival = rng.uniform(size=U) < config.task_arrival_prob
        size_boost = np.ones(U)
    if arrival.any():
        bits = rng.uniform(config.task_bits_min, config.task_bits_max, size=U) * size_boost
        cpb = rng.uniform(config.cycles_per_bit_min, config.cycles_per_bit_max, size=U) * size_boost
    bits *= arrival
    cpb *= arrival
    return bits, cpb


def _post_move_pos(obs, config, move):
    move = np.asarray(move, dtype=float)
    if config.fixed_uav:
        move = np.zeros(2)
    if config.uav_battery_per_slot > 0.0 and obs.get("uav_battery", 1.0) <= 0.0:
        move = np.zeros(2)
    norm = float(np.linalg.norm(move))
    if norm > config.uav_speed_max and norm > 1.0e-9:
        move = move / norm * config.uav_speed_max
    return np.clip(np.asarray(obs["uav_pos"], dtype=float) + move * config.slot_seconds,
                   0.0, config.area_size)


def _exact_cost_at(obs, config, pos, task_bits=None, cycles_per_bit=None):
    """Per-slot cost (latency + energy_weight*energy + drop_penalty*drops) of
    the exact offloading argmin at the given UAV position."""
    U = config.users
    if U == 0:
        return 0.0
    if task_bits is None:
        task_bits = obs["task_bits"]
    if cycles_per_bit is None:
        cycles_per_bit = obs["cycles_per_bit"]
    d_to_leo = np.linalg.norm(np.asarray(obs["user_pos"], dtype=float)[:, None, :]
                              - np.asarray(obs["leo_pos"], dtype=float)[None, :, :], axis=-1)
    leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
    uav_backlog = float(obs.get("uav_backlog", 0.0))
    leo_backlog = float(obs.get("leo_backlog", 0.0))
    total = 0.0
    for u in range(U):
        bits = float(task_bits[u])
        if bits <= 1.0e-6:
            continue
        best = float("inf")
        for target in (0, 1, 2):
            grid = _RATIO_GRID if target > 0 else np.array([0.0])
            for r in grid:
                res = simulate_task(config, obs["user_pos"][u], pos, leo_for_user[u],
                                    bits, cycles_per_bit[u], target, r,
                                    uav_backlog, leo_backlog)
                cost = (res["latency"] + config.energy_weight * res["energy"]
                        + config.drop_penalty * int(res["dropped"]))
                if cost < best - 1.0e-9:
                    best = cost
        total += best
    return total


class MPCTrajPolicy(BasePolicy):
    """Receding-horizon trajectory planner; offloading via exact solver or TEA."""

    def __init__(self, horizon=3, offload="exact", name=None, n_samples=2, seed=12345):
        self.horizon = int(horizon)
        self.offload = offload
        self.name = name or f"mpc_traj_h{horizon}"
        self.n_samples = int(n_samples)
        self._rng = np.random.default_rng(seed)
        self._tea = None
        if offload == "tea":
            from .baselines import PredictTeaPolicy
            self._tea = PredictTeaPolicy()

    def _candidate_moves(self, obs, config):
        w = np.asarray(obs["task_bits"], dtype=float) * np.asarray(obs["cycles_per_bit"], dtype=float)
        users = np.asarray(obs["user_pos"], dtype=float)
        if w.sum() > 1.0e-9:
            weights = w
        else:
            weights = np.ones(len(users))
        h3 = predict_hotspot(obs, config, 3)
        h2 = predict_hotspot(obs, config, 2)
        h1 = predict_hotspot(obs, config, 1)
        u3 = predict_users(obs, config, 3)
        u1 = predict_users(obs, config, 1)
        c3 = np.average(u3, axis=0, weights=weights)
        c1 = np.average(u1, axis=0, weights=weights)
        top3 = u3[int(np.argmax(w))] if w.max() > 1.0e-9 else c3
        targets = [h3, h2, h1, c3, c1, top3]
        if h1 is not None:
            targets.append(0.5 * c3 + 0.5 * h3)
        moves = [np.zeros(2)]
        for t in targets:
            if t is None:
                continue
            diff = np.asarray(t, dtype=float) - np.asarray(obs["uav_pos"], dtype=float)
            norm = float(np.linalg.norm(diff))
            if norm < 1.0e-9:
                continue
            for speed_frac in (1.0, 0.5):
                limit = config.uav_speed_max * speed_frac
                if norm > limit:
                    moves.append(diff / norm * limit)
                else:
                    moves.append(diff)
        return moves

    def _plan(self, obs, config):
        K = self.horizon
        moves = self._candidate_moves(obs, config)
        best_move = np.zeros(2)
        best_cost = float("inf")
        uav0 = np.asarray(obs["uav_pos"], dtype=float)
        for move in moves:
            pos = uav0.copy()
            total = 0.0
            for k in range(K):
                nxt = _post_move_pos(obs, config, move) if k == 0 else np.clip(
                    pos + move * config.slot_seconds, 0.0, config.area_size)
                moved = float(np.linalg.norm(nxt - pos))
                pos = nxt
                pusers = predict_users(obs, config, k)
                p_hot = predict_hotspot(obs, config, k)
                pobs = dict(obs)
                pobs["user_pos"] = pusers
                if p_hot is not None:
                    pobs["hotspot_pos"] = p_hot
                if k == 0:
                    cost = _exact_cost_at(pobs, config, pos)
                else:
                    c_acc = 0.0
                    for _ in range(self.n_samples):
                        bits, cpb = _sample_future_tasks(obs, config, pusers, p_hot, self._rng)
                        c_acc += _exact_cost_at(pobs, config, pos, bits, cpb)
                    cost = c_acc / self.n_samples
                total += cost
                if not config.no_flight_energy:
                    total += config.energy_weight * flight_energy(config, moved)
            if total < best_cost - 1.0e-9:
                best_cost = total
                best_move = move
        return best_move

    def act(self, obs, rng, config):
        move = self._plan(obs, config)
        pos = _post_move_pos(obs, config, move)
        if self.offload == "tea" and self._tea is not None:
            targets, ratios, _ = self._tea._choose_targets(obs, config, pos)
        else:
            from .traj_drl import _exact_offload
            targets, ratios = _exact_offload(obs, config, pos)
        return {"move": move, "targets": targets, "ratios": ratios}
