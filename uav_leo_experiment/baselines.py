import math
import numpy as np
from .physics import nearest_leo_index, rate_uav_leo, rate_user_uav, simulate_task

class BasePolicy:
    name = "base"
    def act(self, obs, rng, config):
        raise NotImplementedError

class RandomPolicy(BasePolicy):
    name = "random"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        return {
            "move": rng.uniform(-1.0, 1.0, size=2) * config.uav_speed_max,
            "targets": rng.integers(0, 3, size=users),
            "ratios": rng.uniform(0.0, 1.0, size=users),
        }

class LocalOnlyPolicy(BasePolicy):
    name = "local_only"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        return {"move": np.zeros(2), "targets": np.zeros(users, dtype=int), "ratios": np.zeros(users)}

class UavOnlyPolicy(BasePolicy):
    name = "uav_only"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        centroid = obs["user_pos"].mean(axis=0)
        move = _limited_move(centroid - obs["uav_pos"], config.uav_speed_max)
        return {"move": move, "targets": np.ones(users, dtype=int), "ratios": np.ones(users)}

class LeoOnlyPolicy(BasePolicy):
    name = "leo_only"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        centroid = obs["user_pos"].mean(axis=0)
        move = _limited_move(centroid - obs["uav_pos"], config.uav_speed_max)
        return {"move": move, "targets": np.full(users, 2, dtype=int), "ratios": np.ones(users)}

class GreedyPartialPolicy(BasePolicy):
    name = "greedy_partial"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        centroid = obs["user_pos"].mean(axis=0)
        move = _limited_move(centroid - obs["uav_pos"], config.uav_speed_max)
        task_score = obs["task_bits"] * obs["cycles_per_bit"]
        threshold = float(np.median(task_score))
        targets = np.where(task_score >= threshold, 2, 1)
        ratios = np.where(task_score >= threshold, 0.75, 0.45)
        return {"move": move, "targets": targets, "ratios": ratios}

class RateAwareRelayPolicy(BasePolicy):
    """Choose the remote target with better predicted link rate, then full offload."""
    name = "rate_aware"
    def act(self, obs, rng, config):
        users = len(obs["task_bits"])
        centroid = obs["user_pos"].mean(axis=0)
        move = _limited_move(centroid - obs["uav_pos"], config.uav_speed_max)
        next_pos = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
        targets = np.ones(users, dtype=int)
        for user_idx in range(users):
            uav_rate = rate_user_uav(config, obs["user_pos"][user_idx], next_pos)
            leo_idx = nearest_leo_index(obs["leo_pos"], obs["user_pos"][user_idx])
            leo_rate = min(uav_rate, rate_uav_leo(config, next_pos, obs["leo_pos"][leo_idx]))
            if leo_rate > 1.2 * uav_rate:
                targets[user_idx] = 2
        return {"move": move, "targets": targets, "ratios": np.ones(users)}

class OptimizedPartialPolicy(BasePolicy):
    """Grid-search partial offloading with lightweight trajectory search.
    All look-ahead cost estimates are computed with the same physics used by
    the environment (:func:`physics.simulate_task`), including shared UAV/LEO
    compute queues when ``config.use_queues`` is enabled.
    """
    name = "optimized_partial"
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    deadline_penalty = 0.0
    energy_scale = 1.0
    flight_scale = 1.0
    prefer_deadline_users = False
    def act(self, obs, rng, config):
        _, _, gains = self._choose_targets(obs, config, obs["uav_pos"])
        target = self._trajectory_target(obs, config, gains)
        base_move = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
        centroid_move = _limited_move(obs["user_pos"].mean(axis=0) - obs["uav_pos"], config.uav_speed_max)
        candidates = [
            np.zeros(2),
            0.5 * base_move,
            base_move,
            0.5 * centroid_move,
            centroid_move,
        ]
        best = None
        for move in candidates:
            predicted_pos = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
            targets, ratios, _ = self._choose_targets(obs, config, predicted_pos)
            decision_cost = sum(
                self._cost(obs, config, predicted_pos, user_idx, int(targets[user_idx]), float(ratios[user_idx]))
                for user_idx in range(len(targets))
            )
            moved = float(np.linalg.norm(predicted_pos - obs["uav_pos"]))
            flight_energy = self._flight_energy(config, moved)
            total_cost = decision_cost + self.flight_scale * config.energy_weight * flight_energy
            if best is None or total_cost < best[0]:
                best = (total_cost, move, targets, ratios)
        return {"move": best[1], "targets": best[2], "ratios": best[3]}
    def _flight_energy(self, config, moved):
        if config.no_flight_energy:
            return 0.0
        speed = moved / max(config.slot_seconds, 1.0e-9)
        if config.speed_cubed_energy:
            return config.hover_power_watt * config.slot_seconds + config.drag_coeff_watt_per_m3s3 * (speed ** 3) * config.slot_seconds
        return config.hover_power_watt * config.slot_seconds + config.move_energy_coeff * moved
    def _trajectory_target(self, obs, config, gains):
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        weights = gains * workload / max(float(workload.max()), 1.0)
        if self.prefer_deadline_users:
            weights = weights * (1.0 + 1.0 / (0.1 + obs["uav_backlog_norm"] + 1.0))
        if weights.sum() <= 0.0:
            return obs["user_pos"].mean(axis=0)
        return np.average(obs["user_pos"], axis=0, weights=weights)
    def _choose_targets(self, obs, config, uav_pos):
        users = len(obs["task_bits"])
        if users * (1 + 2 * (len(self.ratio_grid) - 1)) > 200:
            return self._choose_targets_scalar(obs, config, uav_pos)
        return self._choose_targets_vec(obs, config, uav_pos)
    def _choose_targets_vec(self, obs, config, uav_pos):
        """Vectorized per-user target/ratio grid search (shared physics)."""
        from .physics import rate_uav_leo_vec, rate_user_uav_vec
        U = len(obs["task_bits"])
        grid = self.ratio_grid
        ratios = grid[1:]
        n_targets = 2
        n_combos = 1 + n_targets * len(ratios)
        combo_target = np.zeros(n_combos, dtype=int)
        combo_ratio = np.zeros(n_combos, dtype=float)
        combo_target[1:1 + len(ratios)] = 1
        combo_ratio[1:1 + len(ratios)] = ratios
        combo_target[1 + len(ratios):] = 2
        combo_ratio[1 + len(ratios):] = ratios
        bits = np.asarray(obs["task_bits"], dtype=float)
        cpb = np.asarray(obs["cycles_per_bit"], dtype=float)
        cycles = bits * cpb
        ratio = np.tile(combo_ratio, (U, 1))          # (U, C)
        target = np.tile(combo_target, (U, 1))
        is_local = (target == 0) | (ratio <= 1.0e-6)
        is_uav = ~is_local & (target == 1)
        is_leo = ~is_local & (target == 2)
        leo_idx = np.array([nearest_leo_index(obs["leo_pos"], obs["user_pos"][u]) for u in range(U)])
        leo_pos = np.asarray(obs["leo_pos"], dtype=float)[leo_idx]
        rate_uu = rate_user_uav_vec(config, np.asarray(obs["user_pos"], dtype=float), np.asarray(uav_pos, dtype=float))
        rate_ul = np.minimum(rate_uu, rate_uav_leo_vec(config, np.asarray(uav_pos, dtype=float), leo_pos))
        tx = np.where(is_uav, ratio * bits[:, None] / np.maximum(rate_uu[:, None], 1.0), 0.0)
        tx += np.where(is_leo, ratio * bits[:, None] / np.maximum(rate_ul[:, None], 1.0), 0.0)
        wait = np.where(is_uav, np.maximum(obs["uav_backlog"] - config.uav_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.uav_cpu_cycles_per_s, 0.0)
        wait += np.where(is_leo, np.maximum(obs["leo_backlog"] - config.leo_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.leo_cpu_cycles_per_s, 0.0)
        service = np.where(is_uav, ratio * cycles[:, None] / config.uav_cpu_cycles_per_s, 0.0)
        service += np.where(is_leo, ratio * cycles[:, None] / config.leo_cpu_cycles_per_s, 0.0)
        local_lat = (1.0 - ratio) * cycles[:, None] / config.user_cpu_cycles_per_s
        full_local = cycles[:, None] / config.user_cpu_cycles_per_s
        latency = np.where(is_local, full_local, np.maximum(local_lat, tx + wait + service))
        dropped = latency > config.success_deadline_s
        if config.power_based_energy:
            local_energy = config.user_device_power_watt * local_lat
            full_local_energy = config.user_device_power_watt * full_local
        else:
            local_energy = config.compute_energy_coeff * ((1.0 - ratio) * cycles[:, None]) * (config.user_cpu_cycles_per_s ** 2)
            full_local_energy = config.compute_energy_coeff * cycles[:, None] * (config.user_cpu_cycles_per_s ** 2)
        remote_cycles = ratio * cycles[:, None]
        tx_energy = config.user_tx_power_watt * tx
        if config.power_based_energy:
            uav_energy = config.uav_power_watt * (remote_cycles / config.uav_cpu_cycles_per_s)
            leo_energy = config.leo_power_watt * (remote_cycles / config.leo_cpu_cycles_per_s)
            energy = np.where(is_local, full_local_energy, local_energy + tx_energy)
        else:
            uav_energy = config.compute_energy_coeff * remote_cycles * (config.uav_cpu_cycles_per_s ** 2)
            leo_energy = config.compute_energy_coeff * remote_cycles * (config.leo_cpu_cycles_per_s ** 2)
            energy = np.where(is_local, full_local_energy, local_energy + tx_energy + np.where(is_uav, uav_energy, leo_energy))
        energy = np.where(dropped, np.where(is_local, full_local_energy, local_energy), energy)
        latency = np.where(dropped & ~is_local, config.success_deadline_s, latency)
        violation = np.maximum(latency - config.success_deadline_s, 0.0)
        cost = (config.latency_weight * latency
                + self.energy_scale * config.energy_weight * energy
                + self.deadline_penalty * violation * violation)
        if self.deadline_penalty > 0.0:
            cost = cost + self.deadline_penalty * dropped * config.drop_penalty * 10.0
        best_idx = np.argmin(cost, axis=1)
        targets = combo_target[best_idx].copy()
        ratios_out = combo_ratio[best_idx].copy()
        local_cost = cost[:, 0]
        best_cost = cost[np.arange(U), best_idx]
        gains = np.maximum(local_cost - best_cost, 0.0)
        return targets.astype(int), ratios_out.astype(float), gains.astype(float)
    def _choose_targets_scalar(self, obs, config, uav_pos):
        users = len(obs["task_bits"])
        targets = []
        ratios = []
        gains = []
        for user_idx in range(users):
            local = simulate_task(
                config, obs["user_pos"][user_idx], uav_pos,
                self._leo_for(obs, user_idx), obs["task_bits"][user_idx],
                obs["cycles_per_bit"][user_idx], 0, 0.0,
                obs["uav_backlog"], obs["leo_backlog"],
            )
            local_cost = self._cost_of(config, local)
            best_cost = local_cost
            best_target = 0
            best_ratio = 0.0
            for target in (1, 2):
                for ratio in self.ratio_grid[1:]:
                    result = simulate_task(
                        config, obs["user_pos"][user_idx], uav_pos,
                        self._leo_for(obs, user_idx), obs["task_bits"][user_idx],
                        obs["cycles_per_bit"][user_idx], target, float(ratio),
                        obs["uav_backlog"], obs["leo_backlog"],
                    )
                    cost = self._cost_of(config, result)
                    if cost < best_cost:
                        best_cost = cost
                        best_target = target
                        best_ratio = float(ratio)
            targets.append(best_target)
            ratios.append(best_ratio)
            gains.append(max(0.0, local_cost - best_cost))
        return np.asarray(targets, dtype=int), np.asarray(ratios, dtype=float), np.asarray(gains, dtype=float)
    def _leo_for(self, obs, user_idx):
        leo_idx = nearest_leo_index(obs["leo_pos"], obs["user_pos"][user_idx])
        return obs["leo_pos"][leo_idx]
    def _cost(self, obs, config, uav_pos, user_idx, target, ratio):
        result = simulate_task(
            config, obs["user_pos"][user_idx], uav_pos,
            self._leo_for(obs, user_idx), obs["task_bits"][user_idx],
            obs["cycles_per_bit"][user_idx], target, ratio,
            obs["uav_backlog"], obs["leo_backlog"],
        )
        return self._cost_of(config, result)
    def _cost_of(self, config, result):
        latency = result["latency"]
        deadline_violation = max(0.0, latency - config.success_deadline_s)
        penalty = 0.0
        if result["dropped"]:
            penalty = self.deadline_penalty * config.drop_penalty * 10.0
        return (
            config.latency_weight * latency
            + self.energy_scale * config.energy_weight * result["energy"]
            + self.deadline_penalty * deadline_violation * deadline_violation
            + penalty
        )

class QueueAwareTeaPolicy(OptimizedPartialPolicy):
    """TEA decisions plus queue-aware per-slot load balancing.

    Per-user TEA grid search only sees the backlog at the start of the slot, so
    it can over-commit the shared UAV/LEO queues when many users pick the same
    server in the same slot.  This variant adds a water-filling step: while a
    server's projected per-slot load exceeds its drain capacity, the user with
    the smallest offloading gain on that server is demoted (ratio reduced, or
    moved to the less-loaded server / local execution).  The UAV still tracks
    the task-weighted centroid so the access link stays strong.
    """
    name = "tea_q"
    deadline_penalty = 8.0
    energy_scale = 0.8
    flight_scale = 0.8
    prefer_deadline_users = True

    def act(self, obs, rng, config):
        targets, ratios, gains = self._choose_targets_balanced(obs, config, obs["uav_pos"])
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        if workload.sum() > 1.0e-9:
            target = np.average(obs["user_pos"], axis=0, weights=workload)
        else:
            target = obs["user_pos"].mean(axis=0)
        move = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
        return {"move": move, "targets": targets, "ratios": ratios}

    def _choose_targets_balanced(self, obs, config, uav_pos):
        targets, ratios, gains = self._choose_targets(obs, config, uav_pos)
        if not config.use_queues:
            return targets, ratios, gains
        bits = np.asarray(obs["task_bits"], dtype=float)
        cpb = np.asarray(obs["cycles_per_bit"], dtype=float)
        cycles = bits * cpb
        active = bits > 1.0e-6
        if not active.any():
            return targets, ratios, gains
        dt = max(config.slot_seconds, 1.0e-9)
        uav_cap = config.uav_cpu_cycles_per_s * dt
        leo_cap = config.leo_cpu_cycles_per_s * dt
        ratio_grid = self.ratio_grid
        targets = list(targets)
        ratios = list(ratios)
        gains = list(gains)

        def load_of(server):
            total = 0.0
            for u in range(len(targets)):
                if active[u] and int(targets[u]) == server:
                    total += float(ratios[u]) * float(cycles[u])
            return total

        def demote_one():
            """Return True if a demotion was applied."""
            uav_load = load_of(1)
            leo_load = load_of(2)
            overloaded = []
            if uav_load > uav_cap * 1.02:
                overloaded.append(1)
            if leo_load > leo_cap * 1.02:
                overloaded.append(2)
            if not overloaded:
                return False
            # Demote from the most overloaded server first.
            server = max(overloaded, key=lambda s: (load_of(s) - (uav_cap if s == 1 else leo_cap)) / max(uav_cap if s == 1 else leo_cap, 1.0))
            other = 1 if server == 2 else 2
            other_load = load_of(other)
            other_cap = leo_cap if other == 2 else uav_cap
            # Users on the overloaded server ordered by smallest offload gain first.
            cands = [u for u in range(len(targets)) if active[u] and int(targets[u]) == server]
            if not cands:
                return False
            cands.sort(key=lambda u: (float(gains[u]) / max(float(cycles[u]), 1.0), float(cycles[u])))
            for u in cands:
                cur_ratio = float(ratios[u])
                grid_idx = int(np.argmin(np.abs(ratio_grid - cur_ratio)))
                # Try to move to the other server if it has spare capacity and a
                # lower (or equal) projected cost for the same ratio.
                if other_load + cur_ratio * float(cycles[u]) <= other_cap * 1.02:
                    new_cost = self._cost(obs, config, uav_pos, u, other, cur_ratio)
                    old_cost = self._cost(obs, config, uav_pos, u, server, cur_ratio)
                    if new_cost <= old_cost:
                        targets[u] = other
                        return True
                # Otherwise reduce the ratio (0.75 -> 0.5 -> 0.25 -> local).
                for j in range(grid_idx - 1, -1, -1):
                    new_ratio = float(ratio_grid[j])
                    if new_ratio <= 1.0e-6:
                        targets[u] = 0
                        ratios[u] = 0.0
                        return True
                    new_cost = self._cost(obs, config, uav_pos, u, server, new_ratio)
                    old_cost = self._cost(obs, config, uav_pos, u, server, cur_ratio)
                    if new_cost <= old_cost:
                        ratios[u] = new_ratio
                        return True
            return False

        for _ in range(4 * len(targets)):
            if not demote_one():
                break
        return np.asarray(targets, dtype=int), np.asarray(ratios, dtype=float), np.asarray(gains, dtype=float)


class PredictTeaPolicy(QueueAwareTeaPolicy):
    """Demand-predictive TEA expert: queue-balanced target/ratio decisions
    plus a trajectory that blends the task-weighted user centroid with the
    one-step-ahead predicted hotspot position (hotspot_pos + vel*dt).  The
    hotspot carries the heavy tasks, so anticipating its motion keeps the
    UAV ahead of the workload instead of chasing the current centroid."""
    name = "predict_tea"

    def act(self, obs, rng, config):
        targets, ratios, _ = self._choose_targets_balanced(obs, config, obs["uav_pos"])
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        if workload.sum() > 1.0e-9:
            centroid = np.average(obs["user_pos"], axis=0, weights=workload)
        else:
            centroid = obs["user_pos"].mean(axis=0)
        if obs.get("hotspot_pos") is not None:
            hotspot_next = obs["hotspot_pos"] + obs.get("hotspot_vel", np.zeros(2)) * config.slot_seconds
            target = 0.5 * centroid + 0.5 * hotspot_next
        else:
            target = centroid
        move = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
        return {"move": move, "targets": targets, "ratios": ratios}


class FollowCentroidTeaPolicy(OptimizedPartialPolicy):
    """TEA decisions plus a demand-aware trajectory: always move toward the
    task-weighted user centroid (hotspot tracking). This is the expert used to
    bootstrap DRL agents: it keeps the UAV close to the moving hotspot, which
    greedy one-step TEA policies underestimate because flight energy is charged
    per slot while the rate benefit only materializes one slot later.
    """
    name = "follow_tea"
    deadline_penalty = 8.0
    energy_scale = 0.8
    flight_scale = 0.8
    prefer_deadline_users = True

    def act(self, obs, rng, config):
        targets, ratios, _ = self._choose_targets(obs, config, obs["uav_pos"])
        workload = obs["task_bits"] * obs["cycles_per_bit"]
        if workload.sum() > 1.0e-9:
            target = np.average(obs["user_pos"], axis=0, weights=workload)
        else:
            target = obs["user_pos"].mean(axis=0)
        move = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
        return {"move": move, "targets": targets, "ratios": ratios}


class TrajectoryEnergyAwarePolicy(OptimizedPartialPolicy):
    """Trajectory-energy-aware partial offloading heuristic."""
    name = "tea_partial"
    deadline_penalty = 0.0
    energy_scale = 1.0
    flight_scale = 1.0

class DeadlineAwarePartialPolicy(OptimizedPartialPolicy):
    """TEA variant that emphasizes deadline satisfaction under hard tasks."""
    name = "deadline_tea"
    deadline_penalty = 8.0
    energy_scale = 0.8
    flight_scale = 0.8
    prefer_deadline_users = True

class EnergyGuardedPartialPolicy(OptimizedPartialPolicy):
    """TEA variant with stronger energy and flight movement regularization."""
    name = "energy_guarded_tea"
    deadline_penalty = 2.0
    energy_scale = 2.5
    flight_scale = 2.0

class FullOffloadTeaPolicy(OptimizedPartialPolicy):
    """Ablation-like policy: optimized target and trajectory, but no partial ratios."""
    name = "full_offload_tea"
    ratio_grid = np.array([0.0, 1.0], dtype=float)
    deadline_penalty = 2.0
    energy_scale = 1.0
    flight_scale = 1.0

def estimate_latency_energy(obs, config, uav_pos, user_idx, target, ratio):
    """Backward-compatible wrapper around the shared physics simulator."""
    leo_idx = nearest_leo_index(obs["leo_pos"], obs["user_pos"][user_idx])
    result = simulate_task(
        config, obs["user_pos"][user_idx], uav_pos, obs["leo_pos"][leo_idx],
        obs["task_bits"][user_idx], obs["cycles_per_bit"][user_idx],
        target, ratio, obs.get("uav_backlog", 0.0), obs.get("leo_backlog", 0.0),
    )
    return result["latency"], result["energy"]

def _limited_move(move, speed_max):
    move = np.asarray(move, dtype=float)
    norm = float(np.linalg.norm(move))
    if norm <= speed_max or norm <= 1.0e-9:
        return move
    return move / norm * speed_max


class ExpertExactOffloadPolicy(BasePolicy):
    """Demand-predictive expert trajectory + exact per-slot offloading.

    The offloading decision is evaluated either at the post-move UAV position
    (as GDRL does) or at the current position; the pair isolates the value of
    aligning the offloading decision with the position that actually serves
    the slot.
    """
    name = "expert_exact"

    def __init__(self, offload_at="post_move"):
        self.offload_at = offload_at
        self.name = "postmove_exact" if offload_at == "post_move" else "current_exact"

    def act(self, obs, rng, config):
        from .traj_drl import _exact_offload, _predict_tea_move
        move = _predict_tea_move(obs, config)
        if self.offload_at == "post_move":
            pos = np.clip(obs["uav_pos"] + move * config.slot_seconds, 0.0, config.area_size)
        else:
            pos = np.asarray(obs["uav_pos"], dtype=float)
        targets, ratios = _exact_offload(obs, config, pos)
        return {"move": move, "targets": targets, "ratios": ratios}


class EnergyGatedExpertExactPolicy(ExpertExactOffloadPolicy):
    """PMEO with an energy gate: the demand-predictive move is executed only
    when the offloading gain of the post-move position exceeds the flight
    energy cost of the move (scaled by ``energy_weight``).  This keeps the
    trajectory energy-aware without any lookahead, matching the behaviour of
    MPC when energy is priced."""

    name = "pmeo_e"

    def __init__(self):
        super().__init__(offload_at="post_move")
        self.name = "pmeo_e"

    def act(self, obs, rng, config):
        from .mpc_traj import _exact_cost_at, _post_move_pos
        from .physics import flight_energy
        from .traj_drl import _exact_offload, _predict_tea_move
        move = _predict_tea_move(obs, config)
        pos_new = _post_move_pos(obs, config, move)
        if not config.no_flight_energy:
            moved = float(np.linalg.norm(pos_new - np.asarray(obs["uav_pos"], dtype=float)))
            flight = flight_energy(config, moved)
            cur_pos = np.asarray(obs["uav_pos"], dtype=float)
            gain = _exact_cost_at(obs, config, cur_pos) - _exact_cost_at(obs, config, pos_new)
            if gain <= config.energy_weight * flight:
                move = np.zeros(2, dtype=float)
                pos_new = cur_pos
        targets, ratios = _exact_offload(obs, config, pos_new)
        return {"move": move, "targets": targets, "ratios": ratios}


def default_policies():
    return [
        RandomPolicy(),
        LocalOnlyPolicy(),
        UavOnlyPolicy(),
        LeoOnlyPolicy(),
        GreedyPartialPolicy(),
        RateAwareRelayPolicy(),
        FullOffloadTeaPolicy(),
        EnergyGuardedPartialPolicy(),
        DeadlineAwarePartialPolicy(),
        TrajectoryEnergyAwarePolicy(),
        FollowCentroidTeaPolicy(),
        PredictTeaPolicy(),
        ExpertExactOffloadPolicy(offload_at="post_move"),
        ExpertExactOffloadPolicy(offload_at="current"),
        EnergyGatedExpertExactPolicy(),
    ]
