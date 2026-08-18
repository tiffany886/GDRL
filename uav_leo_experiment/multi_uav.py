"""Multi-UAV policies for the scaled hotspot line (PMEO-M and baselines).

Extends the single-UAV PMEO logic (traj_drl._predict_tea_move / _exact_offload)
to K UAVs:

- association: each user is assigned to one UAV (best user->UAV rate,
  optionally load-balanced so each UAV serves about ceil(U/K) users);
- trajectory: each UAV follows a demand-predictive expert move toward the
  workload-weighted centroid of its associated users, blended with the nearest
  hotspot's one-slot-ahead position;
- offloading: post-move exact search over {local, UAV_1..UAV_K, LEO} x ratio
  grid, using the environment's true per-slot cost (decision-order argument:
  association AND offloading must be solved at the post-move positions).
"""
import numpy as np

from .baselines import BasePolicy
from .physics import flight_energy, rate_uav_leo_vec, rate_user_uav_vec, simulate_task_multi

_RATIO_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


def uav_pos_array(obs, config):
    p = np.asarray(obs["uav_pos"], dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    return p


def uav_backlog_array(obs, config):
    b = obs["uav_backlog"]
    if np.isscalar(b):
        return np.full(int(config.uavs), float(b))
    return np.asarray(b, dtype=float)


def hotspot_arrays(obs, config):
    hp = obs.get("hotspot_pos")
    hv = obs.get("hotspot_vel")
    if hp is None:
        return None, None
    hp = np.asarray(hp, dtype=float)
    hv = np.asarray(hv, dtype=float)
    if hp.ndim == 1:
        hp = hp[None, :]
        hv = hv[None, :]
    return hp, hv


def associate_users(obs, config, uav_pos=None, balanced=True):
    """Per-user UAV index (0..K-1) by best user->UAV rate.

    With balanced=True each UAV serves at most ceil(U/K) users; high-rate users
    are assigned first so the best UAVs are not wasted on low-demand users.
    """
    K = int(config.uavs)
    U = len(obs["task_bits"])
    p = uav_pos_array(obs, config) if uav_pos is None else np.asarray(uav_pos, dtype=float)
    rate = rate_user_uav_vec(config, np.asarray(obs['user_pos'], dtype=float)[:, None, :], p[None, :, :])  # (U,K)
    if K == 1 or not balanced:
        return np.argmax(rate, axis=1)
    cap = int(np.ceil(U / K))
    best = rate.argmax(axis=1)
    order = np.argsort(-rate[np.arange(U), best])
    assoc = np.full(U, -1, dtype=int)
    counts = np.zeros(K, dtype=int)
    for u in order:
        ranked = np.argsort(-rate[u])
        chosen = -1
        for k in ranked:
            if counts[k] < cap:
                chosen = int(k)
                break
        if chosen < 0:
            chosen = int(best[u])
        assoc[u] = chosen
        counts[chosen] += 1
    return assoc


def predict_tea_move_multi(obs, config, assoc=None):
    """Per-UAV demand-predictive expert move (trajectory layer of PMEO-M)."""
    K = int(config.uavs)
    p = uav_pos_array(obs, config)
    if assoc is None:
        assoc = associate_users(obs, config)
    workload = np.asarray(obs["task_bits"], dtype=float) * np.asarray(obs["cycles_per_bit"], dtype=float)
    user_pos = np.asarray(obs["user_pos"], dtype=float)
    hp, hv = hotspot_arrays(obs, config)
    moves = np.zeros((K, 2), dtype=float)
    for k in range(K):
        idx = np.where(assoc == k)[0]
        if len(idx) > 0:
            wsum = float(workload[idx].sum())
            if wsum > 1.0e-9:
                centroid = np.average(user_pos[idx], axis=0, weights=workload[idx])
            else:
                centroid = user_pos[idx].mean(axis=0)
        else:
            centroid = p[k]
        if hp is not None:
            d2 = np.sum((hp - p[k]) ** 2, axis=1)
            m = int(np.argmin(d2))
            hot_next = hp[m] + hv[m] * config.slot_seconds
            target = 0.5 * centroid + 0.5 * hot_next
        else:
            target = centroid
        mv = target - p[k]
        norm = float(np.linalg.norm(mv))
        if norm > config.uav_speed_max and norm > 1.0e-9:
            mv = mv / norm * config.uav_speed_max
        moves[k] = mv
    return moves


def _offload_cost_tensor(config, user_pos, uav_pos, leo_pos, bits, cpb,
                         uav_backlogs, leo_backlog):
    """Vectorized per-slot cost tensor (U, K+2, R).

    targets: 0 = local, 1..K = UAV k, K+1 = LEO; R = len(_RATIO_GRID).
    The LEO route uses the best UAV relay rate. Matches the env's per-slot
    cost definition used in :func:`simulate_task_multi`.
    """
    K = len(uav_pos)
    U = len(bits)
    R = len(_RATIO_GRID)
    ratio = _RATIO_GRID[None, :]                                        # (1,R)
    bits = np.asarray(bits, dtype=float)[:, None]                        # (U,1)
    cycles = bits * np.asarray(cpb, dtype=float)[:, None]                # (U,1)
    local_latency = (1.0 - ratio) * cycles / config.user_cpu_cycles_per_s          # (U,R)
    full_local_latency = cycles / config.user_cpu_cycles_per_s
    if config.power_based_energy:
        local_energy = config.user_device_power_watt * local_latency
        full_local_energy = config.user_device_power_watt * full_local_latency
    else:
        local_energy = config.compute_energy_coeff * ((1.0 - ratio) * cycles) * (config.user_cpu_cycles_per_s ** 2)
        full_local_energy = config.compute_energy_coeff * cycles * (config.user_cpu_cycles_per_s ** 2)
    user_pos = np.asarray(user_pos, dtype=float)
    uav_pos = np.asarray(uav_pos, dtype=float)
    leo_pos = np.asarray(leo_pos, dtype=float)
    rate_uu = rate_user_uav_vec(config, user_pos[:, None, :], uav_pos[None, :, :])       # (U,K)
    relay = np.minimum(rate_uu, rate_uav_leo_vec(config, uav_pos[None, :, :], leo_pos[:, None, :]))  # (U,K)
    best_relay = relay.max(axis=1)                                                        # (U,)
    cost = np.zeros((U, K + 2, R), dtype=float)
    # local (target 0): full local execution, ratio irrelevant
    dropped_local = full_local_latency > config.success_deadline_s
    cost[:, 0, :] = (full_local_latency + config.energy_weight * full_local_energy
                     + config.drop_penalty * dropped_local.astype(float))
    # UAV targets 1..K
    for k in range(K):
        rate = rate_uu[:, k][:, None]                                                      # (U,1)
        backlog = float(uav_backlogs[k])
        tx = ratio * bits / np.maximum(rate, 1.0)
        wait = np.maximum(backlog - config.uav_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.uav_cpu_cycles_per_s
        service = ratio * cycles / config.uav_cpu_cycles_per_s
        lat = np.maximum(local_latency, tx + wait + service)
        dropped = lat > config.success_deadline_s
        tx_energy = config.user_tx_power_watt * tx
        if config.power_based_energy:
            srv_eng = config.uav_power_watt * (ratio * cycles / config.uav_cpu_cycles_per_s)
        else:
            srv_eng = config.compute_energy_coeff * (ratio * cycles) * (config.uav_cpu_cycles_per_s ** 2)
        eng = np.where(dropped, local_energy, local_energy + tx_energy)
        lat = np.where(dropped, config.success_deadline_s, lat)
        cost[:, k + 1, :] = (lat + config.energy_weight * eng
                             + config.drop_penalty * dropped.astype(float))
    # LEO target K+1 (best relay UAV)
    rate = best_relay[:, None]
    backlog = float(leo_backlog)
    tx = ratio * bits / np.maximum(rate, 1.0)
    wait = np.maximum(backlog - config.leo_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.leo_cpu_cycles_per_s
    service = ratio * cycles / config.leo_cpu_cycles_per_s
    lat = np.maximum(local_latency, tx + wait + service)
    dropped = lat > config.success_deadline_s
    tx_energy = config.user_tx_power_watt * tx
    if config.power_based_energy:
        srv_eng = config.leo_power_watt * (ratio * cycles / config.leo_cpu_cycles_per_s)
    else:
        srv_eng = config.compute_energy_coeff * (ratio * cycles) * (config.leo_cpu_cycles_per_s ** 2)
    eng = np.where(dropped, local_energy, local_energy + tx_energy)
    lat = np.where(dropped, config.success_deadline_s, lat)
    cost[:, K + 1, :] = (lat + config.energy_weight * eng
                         + config.drop_penalty * dropped.astype(float))
    return cost


def exact_offload_multi(obs, config, uav_pos):
    """Post-move exact offloading: per-user argmin over
    {local, UAV_1..UAV_K, LEO} x ratio grid using the env's per-slot cost.
    With fixed slot-start backlogs the per-slot cost is additive across users,
    so the per-user search is the global optimum for the slot."""
    K = int(config.uavs)
    U = len(obs["task_bits"])
    p = np.asarray(uav_pos, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    backlogs = uav_backlog_array(obs, config)
    leo_backlog = float(obs.get("leo_backlog", 0.0))
    d_to_leo = np.linalg.norm(
        np.asarray(obs["user_pos"], dtype=float)[:, None, :]
        - np.asarray(obs["leo_pos"], dtype=float)[None, :, :],
        axis=-1,
    )
    leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
    active = np.asarray(obs["task_bits"], dtype=float) > 1.0e-6
    cost = _offload_cost_tensor(config, obs["user_pos"], p, leo_for_user,
                                obs["task_bits"], obs["cycles_per_bit"],
                                backlogs, leo_backlog)
    flat = cost.reshape(U, -1)
    idx = np.argmin(flat, axis=1)
    tgt = idx // len(_RATIO_GRID)
    ridx = idx % len(_RATIO_GRID)
    targets = tgt.astype(int)
    ratios = _RATIO_GRID[ridx].copy()
    targets[~active] = 0
    ratios[~active] = 0.0
    return targets, ratios




class PmeoMPolicy(BasePolicy):
    """PMEO-M: association + per-UAV expert move + post-move exact offload."""
    name = "pmeo_m"

    def __init__(self, balanced=True):
        self.balanced = balanced

    def act(self, obs, rng, config):
        K = int(config.uavs)
        assoc = associate_users(obs, config, balanced=self.balanced)
        moves = predict_tea_move_multi(obs, config, assoc)
        p_next = np.clip(uav_pos_array(obs, config) + moves * config.slot_seconds,
                         0.0, config.area_size)
        targets, ratios = exact_offload_multi(obs, config, p_next)
        return {"move": moves, "targets": targets, "ratios": ratios}


class CurrentExactMPolicy(BasePolicy):
    """Same expert moves as PMEO-M but offloading solved at the current
    (pre-move) UAV positions: decision-order ablation."""
    name = "current_exact_m"

    def act(self, obs, rng, config):
        assoc = associate_users(obs, config)
        moves = predict_tea_move_multi(obs, config, assoc)
        targets, ratios = exact_offload_multi(obs, config, uav_pos_array(obs, config))
        return {"move": moves, "targets": targets, "ratios": ratios}


class FollowTeaMPolicy(BasePolicy):
    """Move toward the centroid of each UAV's associated users (no hotspot
    prediction, no post-move offloading)."""
    name = "follow_tea_m"

    def act(self, obs, rng, config):
        K = int(config.uavs)
        p = uav_pos_array(obs, config)
        assoc = associate_users(obs, config)
        user_pos = np.asarray(obs["user_pos"], dtype=float)
        moves = np.zeros((K, 2), dtype=float)
        for k in range(K):
            idx = np.where(assoc == k)[0]
            centroid = user_pos[idx].mean(axis=0) if len(idx) > 0 else p[k]
            mv = centroid - p[k]
            norm = float(np.linalg.norm(mv))
            if norm > config.uav_speed_max and norm > 1.0e-9:
                mv = mv / norm * config.uav_speed_max
            moves[k] = mv
        targets, ratios = exact_offload_multi(obs, config, p)
        return {"move": moves, "targets": targets, "ratios": ratios}


class RandomMPolicy(BasePolicy):
    name = "random_m"

    def act(self, obs, rng, config):
        K = int(config.uavs)
        U = len(obs["task_bits"])
        moves = rng.uniform(-1.0, 1.0, size=(K, 2)) * config.uav_speed_max
        targets = rng.integers(0, K + 2, size=U)
        ratios = rng.choice(_RATIO_GRID, size=U)
        return {"move": moves, "targets": targets, "ratios": ratios}


class LocalOnlyMPolicy(BasePolicy):
    name = "local_only_m"

    def act(self, obs, rng, config):
        K = int(config.uavs)
        U = len(obs["task_bits"])
        return {"move": np.zeros((K, 2)), "targets": np.zeros(U, dtype=int),
                "ratios": np.zeros(U)}


class LeoOnlyMPolicy(BasePolicy):
    name = "leo_only_m"

    def act(self, obs, rng, config):
        K = int(config.uavs)
        U = len(obs["task_bits"])
        moves = predict_tea_move_multi(obs, config)
        return {"move": moves, "targets": np.full(U, K + 1, dtype=int),
                "ratios": np.ones(U)}


class UavOnlyMPolicy(BasePolicy):
    name = "uav_only_m"

    def act(self, obs, rng, config):
        K = int(config.uavs)
        U = len(obs["task_bits"])
        assoc = associate_users(obs, config)
        moves = predict_tea_move_multi(obs, config, assoc)
        return {"move": moves, "targets": assoc + 1, "ratios": np.ones(U)}


class PmeoMEcoPolicy(BasePolicy):
    """PMEO-M with flight restraint and H-step lookahead (PMEO-M-Eco).

    Same load-balanced association and post-move exact offloading as PMEO-M,
    but each UAV picks the candidate first move (zero move, expert full/half
    speed, or pure workload centroid) with the lowest H-step cumulative
    post-move exact-offload cost plus flight energy.  The H-step lookahead
    predicts user/hotspot motion, so a UAV stays put only when staying put is
    truly no worse over the horizon (it does not sacrifice future drops to
    save one slot of flight).  Decoupled and deterministic: no sampling, and
    the per-UAV search is over a fixed small candidate set, so the policy
    remains far cheaper than the sampling-based MPC baseline.
    """
    name = "pmeo_m_eco"

    def __init__(self, balanced=True, horizon=3, include_future=False,
                 include_leo=False, name=None, cand_mode="full"):
        self.balanced = balanced
        self.horizon = int(horizon)
        self.include_future = include_future
        self.include_leo = include_leo
        self.cand_mode = cand_mode
        if name is not None:
            self.name = name

    def _candidate_cost(self, obs, config, k, pos_all, mv, leo_for_user,
                        backlogs, leo_backlog):
        """H-step cumulative exact-offload cost for UAV k flying mv each slot."""
        dt = config.slot_seconds
        user_pos = np.asarray(obs["user_pos"], dtype=float)
        user_vel = np.asarray(obs["user_vel"], dtype=float)
        hp = obs.get("hotspot_pos")
        hv = obs.get("hotspot_vel")
        total = 0.0
        for h in range(self.horizon):
            cost = _offload_cost_tensor(
                config, user_pos, pos_all, leo_for_user,
                obs["task_bits"], obs["cycles_per_bit"], backlogs, leo_backlog)
            total += float(cost.min(axis=(1, 2)).sum())
            if h < self.horizon - 1:
                user_pos = np.clip(user_pos + user_vel * dt, 0.0, config.area_size)
                if hp is not None:
                    hp = np.asarray(hp, dtype=float) + np.asarray(hv, dtype=float) * dt
                pos_all = pos_all.copy()
                pos_all[k] = np.clip(pos_all[k] + mv * dt, 0.0, config.area_size)
        dist = float(np.linalg.norm(mv)) * dt
        total += config.energy_weight * self.horizon * flight_energy(config, dist)
        return total

    def act(self, obs, rng, config):
        K = int(config.uavs)
        dt = config.slot_seconds
        p = uav_pos_array(obs, config)
        assoc = associate_users(obs, config, balanced=self.balanced)
        expert = predict_tea_move_multi(obs, config, assoc)
        user_pos = np.asarray(obs["user_pos"], dtype=float)
        user_vel = np.asarray(obs["user_vel"], dtype=float)
        user_pos_next = np.clip(user_pos + user_vel * dt, 0.0, config.area_size)
        workload = (np.asarray(obs["task_bits"], dtype=float)
                    * np.asarray(obs["cycles_per_bit"], dtype=float))
        backlogs = uav_backlog_array(obs, config)
        leo_backlog = float(obs.get("leo_backlog", 0.0))
        d_to_leo = np.linalg.norm(
            user_pos[:, None, :] - np.asarray(obs["leo_pos"], dtype=float)[None, :, :],
            axis=-1)
        leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
        moves = np.zeros((K, 2), dtype=float)
        for k in range(K):
            idx = np.where(assoc == k)[0]
            centroid = p[k]
            if len(idx) > 0:
                wsum = float(workload[idx].sum())
                if wsum > 1.0e-9:
                    centroid = np.average(user_pos[idx], axis=0, weights=workload[idx])
                else:
                    centroid = user_pos[idx].mean(axis=0)
            cand = [
                np.zeros(2),
                _clip_move(expert[k], config.uav_speed_max),
                0.5 * _clip_move(expert[k], config.uav_speed_max),
                _clip_move(centroid - p[k], config.uav_speed_max),
            ]
            if self.cand_mode == "centroid":
                cand = [
                    np.zeros(2),
                    _clip_move(centroid - p[k], config.uav_speed_max),
                ]
            if self.include_future:
                fcentroid = p[k]
                if len(idx) > 0:
                    wsum = float(workload[idx].sum())
                    if wsum > 1.0e-9:
                        fcentroid = np.average(user_pos_next[idx], axis=0, weights=workload[idx])
                    else:
                        fcentroid = user_pos_next[idx].mean(axis=0)
                cand.append(_clip_move(fcentroid - p[k], config.uav_speed_max))
                hp, hv = hotspot_arrays(obs, config)
                if hp is not None and len(idx) > 0:
                    d2 = np.sum((hp - p[k]) ** 2, axis=1)
                    m = int(np.argmin(d2))
                    hot_next = hp[m] + hv[m] * dt
                    f_expert = 0.5 * fcentroid + 0.5 * hot_next
                    cand.append(_clip_move(f_expert - p[k], config.uav_speed_max))
            if self.include_leo:
                leo_xy = np.asarray(obs["leo_pos"], dtype=float)[:, :2]
                if len(idx) > 0:
                    d_avg = np.mean(
                        np.linalg.norm(user_pos[idx, None, :] - leo_xy[None, :, :], axis=-1),
                        axis=0)
                    m_leo = int(np.argmin(d_avg))
                else:
                    m_leo = int(np.argmin(np.linalg.norm(p[k][None, :] - leo_xy, axis=-1)))
                cand.append(_clip_move(leo_xy[m_leo] - p[k], config.uav_speed_max))
            best_mv = cand[0]
            best_cost = float("inf")
            for mv in cand:
                pos_all = p.copy()
                pos_all[k] = np.clip(p[k] + mv * dt, 0.0, config.area_size)
                total = self._candidate_cost(obs, config, k, pos_all, mv,
                                             leo_for_user, backlogs, leo_backlog)
                if total < best_cost:
                    best_cost = total
                    best_mv = mv
            moves[k] = best_mv
        p_next = np.clip(p + moves * dt, 0.0, config.area_size)
        targets, ratios = exact_offload_multi(obs, config, p_next)
        return {"move": moves, "targets": targets, "ratios": ratios}


def _clip_move(mv, speed_max):
    mv = np.asarray(mv, dtype=float)
    norm = float(np.linalg.norm(mv))
    if norm > speed_max and norm > 1.0e-9:
        mv = mv / norm * speed_max
    return mv


class MpcMPolicy(BasePolicy):
    """Decoupled receding-horizon trajectory planner (MPC-M).

    Each UAV plans H-step lookahead over its load-balanced association subset
    (users are fixed to UAV k during planning); candidate first moves are the
    zero move, demand centroid, hotspot-predictive move and random
    perturbations.  The H-step cumulative cost uses the same vectorized cost
    tensor as PMEO-M over {local, UAV k, LEO} x ratio grid.  The executed
    offloading is post-move exact over all UAVs (same as PMEO-M).
    """
    name = "mpc_m_h3"

    def __init__(self, horizon=3, n_samples=4, seed=12345):
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.rng_seed = int(seed)

    def _sub_cost(self, obs, config, k, idx, uav_pos_k, leo_for_user):
        """One-slot exact cost of subset users against a single candidate UAV k
        (targets restricted to {local, UAV k, LEO})."""
        # single-UAV tensor: targets {local=0, UAV k=1, LEO=2}
        cost = _offload_cost_tensor(config, obs["user_pos"], uav_pos_k[None, :],
                                    leo_for_user, obs["task_bits"],
                                    obs["cycles_per_bit"], uav_backlog_array(obs, config),
                                    float(obs.get("leo_backlog", 0.0)))
        cost = cost[idx, :, :].reshape(len(idx), -1)
        return float(cost.min(axis=1).sum())

    def _plan_uav(self, obs, config, k, idx, rng):
        user_pos = np.asarray(obs["user_pos"], dtype=float)
        p_all = uav_pos_array(obs, config)
        p_k = p_all[k]
        speed = config.uav_speed_max
        dt = config.slot_seconds
        workload = np.asarray(obs["task_bits"], dtype=float) * np.asarray(obs["cycles_per_bit"], dtype=float)
        centroid = user_pos[idx].mean(axis=0) if len(idx) > 0 else p_k
        if len(idx) > 0 and float(workload[idx].sum()) > 1.0e-9:
            centroid = np.average(user_pos[idx], axis=0, weights=workload[idx])
        expert = predict_tea_move_multi(obs, config)[k]
        cand = [np.zeros(2), _clip_move(centroid - p_k, speed), expert]
        hp, hv = hotspot_arrays(obs, config)
        if hp is not None:
            d2 = np.sum((hp - p_k) ** 2, axis=1)
            m = int(np.argmin(d2))
            target = 0.5 * centroid + 0.5 * (hp[m] + hv[m] * dt)
            cand.append(_clip_move(target - p_k, speed))
        for _ in range(self.n_samples):
            cand.append(_clip_move(expert + rng.normal(0.0, 0.4 * speed, size=2), speed))
        d_to_leo = np.linalg.norm(
            np.asarray(obs["user_pos"], dtype=float)[:, None, :]
            - np.asarray(obs["leo_pos"], dtype=float)[None, :, :], axis=-1)
        leo_for_user = np.asarray(obs["leo_pos"], dtype=float)[np.argmin(d_to_leo, axis=1)]
        best = np.zeros(2)
        best_cost = float("inf")
        for mv in cand:
            pos = np.clip(p_k + mv * dt, 0.0, config.area_size)
            cost = 0.0
            pobs = obs
            for h in range(self.horizon):
                cost += self._sub_cost(pobs, config, k, idx, pos, leo_for_user)
                if h < self.horizon - 1:
                    nxt = np.clip(pobs["user_pos"] + pobs["user_vel"] * dt, 0.0, config.area_size)
                    pobs = dict(pobs)
                    pobs["user_pos"] = nxt
                    if pobs.get("hotspot_pos") is not None:
                        hp2 = np.asarray(pobs["hotspot_pos"], dtype=float)
                        hv2 = np.asarray(pobs["hotspot_vel"], dtype=float)
                        pobs["hotspot_pos"] = hp2 + hv2 * dt
                pos = np.clip(pos + mv * dt, 0.0, config.area_size)
            cost += config.energy_weight * self.horizon * flight_energy(config, float(np.linalg.norm(mv)))
            if cost < best_cost:
                best_cost = cost
                best = mv
        return best

    def act(self, obs, rng, config):
        K = int(config.uavs)
        assoc = associate_users(obs, config)
        rng_loc = np.random.default_rng(self.rng_seed)
        moves = np.zeros((K, 2), dtype=float)
        for k in range(K):
            idx = np.where(assoc == k)[0]
            moves[k] = self._plan_uav(obs, config, k, idx, rng_loc)
        p_next = np.clip(uav_pos_array(obs, config) + moves * config.slot_seconds,
                         0.0, config.area_size)
        targets, ratios = exact_offload_multi(obs, config, p_next)
        return {"move": moves, "targets": targets, "ratios": ratios}


def multi_uav_policies():
    return [
        PmeoMPolicy(),
        CurrentExactMPolicy(),
        FollowTeaMPolicy(),
        UavOnlyMPolicy(),
        LeoOnlyMPolicy(),
        LocalOnlyMPolicy(),
        RandomMPolicy(),
        PmeoMEcoPolicy(),
        PmeoMEcoPolicy(horizon=5, include_future=True, name="pmeo_m_eco_h5"),
        PmeoMEcoPolicy(balanced=False, name="pmeo_m_eco_nb"),
        PmeoMEcoPolicy(cand_mode="centroid", name="pmeo_m_eco_cc"),
        MpcMPolicy(),
    ]
