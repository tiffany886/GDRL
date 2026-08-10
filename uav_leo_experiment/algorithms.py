"""Literature-style optimization baselines for the UAV-LEO offloading problem.
Policies implemented here are standard benchmarks found in MEC / V2X papers:
- :class:`LyapunovPolicy`        — drift-plus-penalty online algorithm using
                                   queue backlogs as Lyapunov drift.
- :class:`GeneticPolicy`         — per-slot genetic algorithm optimizer.
- :class:`ExhaustiveOptimalPolicy` — per-slot exhaustive search (near-optimal
                                   upper bound for small user counts, coordinate
                                   descent fallback for larger counts).
All look-ahead estimates use the same vectorized physics as the environment so
benchmark results are directly comparable.
"""
import numpy as np
from .baselines import BasePolicy, OptimizedPartialPolicy, _limited_move
from .physics import flight_energy, nearest_leo_index, simulate_task

# ---------------------------------------------------------------------------
# Vectorized physics (used by search-based policies)
# ---------------------------------------------------------------------------
def _rate_user_uav_vec(config, user_pos, uav_pos):
    """Vectorized user->UAV rate. Inputs broadcastable to common shape. (..., 2)."""
    d3_sq = np.sum((user_pos - uav_pos) ** 2, axis=-1) + config.uav_altitude ** 2
    if config.channel_model == "los_nlos":
        d3 = np.sqrt(np.maximum(d3_sq, 1.0))
        hdist = np.sqrt(np.maximum(d3_sq - config.uav_altitude ** 2, 0.0))
        theta = np.degrees(np.arctan2(config.uav_altitude, np.maximum(hdist, 1.0)))
        plos = 1.0 / (1.0 + config.los_param_a * np.exp(-config.los_param_b * (theta - config.los_param_a)))
        fspl = 20.0 * np.log10(d3) + 20.0 * np.log10(config.carrier_hz) - 147.55
        exp = getattr(config, "path_loss_exp", 2.0)
        if exp != 2.0:
            fspl = fspl + 10.0 * (exp - 2.0) * np.log10(d3)
        g_los = 10.0 ** (-(fspl + config.path_loss_los) / 10.0)
        g_nlos = 10.0 ** (-(fspl + config.path_loss_nlos) / 10.0)
        gain = (plos * g_los + (1.0 - plos) * g_nlos) * (10.0 ** (config.user_uav_gain_db / 10.0))
        snr = config.user_tx_power_watt * gain / config.noise_watt
    else:
        snr = config.user_tx_power_watt * (1.0 / np.maximum(d3_sq, 1.0)) / config.noise_watt
    return np.maximum(config.bandwidth_hz * np.log2(1.0 + snr), 1.0)

def _rate_uav_leo_vec(config, uav_pos, leo_pos):
    d3_sq = np.sum((uav_pos - leo_pos) ** 2, axis=-1) + config.leo_altitude ** 2
    if config.channel_model == "los_nlos":
        d3 = np.sqrt(np.maximum(d3_sq, 1.0))
        hdist = np.sqrt(np.maximum(d3_sq - config.leo_altitude ** 2, 0.0))
        theta = np.degrees(np.arctan2(config.leo_altitude, np.maximum(hdist, 1.0)))
        plos = 1.0 / (1.0 + config.los_param_a * np.exp(-config.los_param_b * (theta - config.los_param_a)))
        fspl = 20.0 * np.log10(d3) + 20.0 * np.log10(config.carrier_hz) - 147.55
        exp = getattr(config, "path_loss_exp", 2.0)
        if exp != 2.0:
            fspl = fspl + 10.0 * (exp - 2.0) * np.log10(d3)
        g_los = 10.0 ** (-(fspl + config.path_loss_los) / 10.0)
        g_nlos = 10.0 ** (-(fspl + config.path_loss_nlos) / 10.0)
        gain = (plos * g_los + (1.0 - plos) * g_nlos) * (10.0 ** (config.uav_leo_gain_db / 10.0))
        snr = config.uav_tx_power_watt * gain / config.noise_watt
    else:
        snr = config.uav_tx_power_watt * (1.0 / np.maximum(d3_sq, 1.0)) / config.noise_watt
    return np.maximum(config.bandwidth_hz * np.log2(1.0 + snr), 1.0)

def evaluate_actions_vec(config, obs, moves, targets, ratios):
    """Vectorized look-ahead cost of a batch of candidate actions.
    Parameters
    ----------
    moves   : (P, 2) or (2,) array of UAV moves.
    targets : (P, U) integer array of targets.
    ratios  : (P, U) float array of offload ratios.
    Returns
    -------
    total_cost : (P,) sum over users of latency + energy_weight * energy
                 + drop_penalty * dropped, plus flight energy per move.
    dropped    : (P, U) drop indicators.
    latencies  : (P, U)
    """
    moves = np.asarray(moves, dtype=float)
    if moves.ndim == 1:
        moves = moves[None, :]
    targets = np.asarray(targets)
    ratios = np.asarray(ratios, dtype=float)
    if moves.shape[0] == 1 and targets.shape[0] > 1:
        moves = np.repeat(moves, targets.shape[0], axis=0)
    P = moves.shape[0]
    U = len(obs["task_bits"])
    uav_next = np.clip(obs["uav_pos"] + moves * config.slot_seconds, 0.0, config.area_size)
    uav_next = uav_next[:, None, :]  # (P,1,2)
    user_pos = np.asarray(obs["user_pos"], dtype=float)[None, :, :]  # (1,U,2)
    bits = np.asarray(obs["task_bits"], dtype=float)[None, :]
    cpb = np.asarray(obs["cycles_per_bit"], dtype=float)[None, :]
    cycles = bits * cpb
    ratio = np.clip(np.asarray(ratios, dtype=float), 0.0, 1.0)
    target = np.asarray(targets, dtype=int)
    local_cycles = (1.0 - ratio) * cycles
    local_latency = local_cycles / config.user_cpu_cycles_per_s
    if config.power_based_energy:
        local_energy = config.user_device_power_watt * local_latency
        full_local_energy = config.user_device_power_watt * (cycles / config.user_cpu_cycles_per_s)
    else:
        local_energy = config.compute_energy_coeff * local_cycles * (config.user_cpu_cycles_per_s ** 2)
        full_local_energy = config.compute_energy_coeff * cycles * (config.user_cpu_cycles_per_s ** 2)
    full_local_latency = cycles / config.user_cpu_cycles_per_s
    # per-user nearest LEO position (1, U, 2)
    leo_idx = np.array([nearest_leo_index(obs["leo_pos"], obs["user_pos"][u]) for u in range(U)])
    leo_pos = np.asarray(obs["leo_pos"], dtype=float)[leo_idx][None, :, :]
    rate_uu = _rate_user_uav_vec(config, user_pos, uav_next)              # (P,U)
    rate_ul = _rate_uav_leo_vec(config, uav_next, leo_pos)                # (P,U)
    rate_leo = np.minimum(rate_uu, rate_ul)
    is_local = (target == 0) | (ratio <= 1.0e-6)
    is_uav = ~is_local & (target == 1)
    is_leo = ~is_local & (target == 2)
    tx_latency = np.zeros((P, U))
    wait_latency = np.zeros((P, U))
    service_latency = np.zeros((P, U))
    tx_latency += np.where(is_uav, ratio * bits / np.maximum(rate_uu, 1.0), 0.0)
    tx_latency += np.where(is_leo, ratio * bits / np.maximum(rate_leo, 1.0), 0.0)
    wait_latency += np.where(
        is_uav,
        np.maximum(obs["uav_backlog"] - config.uav_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.uav_cpu_cycles_per_s,
        0.0,
    )
    wait_latency += np.where(
        is_leo,
        np.maximum(obs["leo_backlog"] - config.leo_cpu_cycles_per_s * config.slot_seconds, 0.0) / config.leo_cpu_cycles_per_s,
        0.0,
    )
    service_latency += np.where(is_uav, ratio * cycles / config.uav_cpu_cycles_per_s, 0.0)
    service_latency += np.where(is_leo, ratio * cycles / config.leo_cpu_cycles_per_s, 0.0)
    latency = np.where(is_local, full_local_latency, np.maximum(local_latency, tx_latency + wait_latency + service_latency))
    dropped = latency > config.success_deadline_s
    remote_cycles = ratio * cycles
    tx_energy = config.user_tx_power_watt * tx_latency
    if config.power_based_energy:
        uav_energy = config.uav_power_watt * (remote_cycles / config.uav_cpu_cycles_per_s)
        leo_energy = config.leo_power_watt * (remote_cycles / config.leo_cpu_cycles_per_s)
        energy = np.where(is_local, full_local_energy, local_energy + tx_energy)
    else:
        uav_energy = config.compute_energy_coeff * remote_cycles * (config.uav_cpu_cycles_per_s ** 2)
        leo_energy = config.compute_energy_coeff * remote_cycles * (config.leo_cpu_cycles_per_s ** 2)
        energy = np.where(is_local, full_local_energy, local_energy + tx_energy + np.where(is_uav, uav_energy, leo_energy))
    energy = np.where(dropped, np.where(is_local, full_local_energy, local_energy), energy)
    latency = np.where(dropped, config.success_deadline_s, latency)
    total = (
        config.latency_weight * latency.sum(axis=1)
        + config.energy_weight * energy.sum(axis=1)
        + config.drop_penalty * dropped.sum(axis=1)
    )
    if not config.no_flight_energy:
        norms = np.linalg.norm(moves, axis=1)
        speed = norms / max(config.slot_seconds, 1.0e-9)
        if config.speed_cubed_energy:
            flight_vec = config.hover_power_watt * config.slot_seconds + config.drag_coeff_watt_per_m3s3 * (speed ** 3) * config.slot_seconds
        else:
            flight_vec = config.hover_power_watt * config.slot_seconds + config.move_energy_coeff * norms
        total += config.energy_weight * flight_vec
    return total, dropped, latency, energy

def _flight(config, moved):
    if config.no_flight_energy:
        return 0.0
    return flight_energy(config, moved)

# ---------------------------------------------------------------------------
# Lyapunov drift-plus-penalty policy
# ---------------------------------------------------------------------------
class LyapunovPolicy(OptimizedPartialPolicy):
    """Drift-plus-penalty: minimize V*cost + backlog * added_cycles."""
    name = "lyapunov"
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    lyapunov_v = 1.0
    def __init__(self, lyapunov_v=None):
        if lyapunov_v is not None:
            self.lyapunov_v = lyapunov_v
    def _choose_targets(self, obs, config, uav_pos):
        users = len(obs["task_bits"])
        targets = []
        ratios = []
        gains = []
        cap_uav = config.uav_cpu_cycles_per_s * config.slot_seconds
        cap_leo = config.leo_cpu_cycles_per_s * config.slot_seconds
        q_uav = max(float(obs["uav_backlog"]), 0.0) / cap_uav
        q_leo = max(float(obs["leo_backlog"]), 0.0) / cap_leo
        for user_idx in range(users):
            leo = self._leo_for(obs, user_idx)
            best_cost = None
            best_target = 0
            best_ratio = 0.0
            for target in (0, 1, 2):
                grid = self.ratio_grid if target != 0 else np.array([0.0])
                for ratio in grid:
                    result = simulate_task(
                        config, obs["user_pos"][user_idx], uav_pos, leo,
                        obs["task_bits"][user_idx], obs["cycles_per_bit"][user_idx],
                        target, float(ratio), obs["uav_backlog"], obs["leo_backlog"],
                    )
                    drift = q_uav * (result["added_uav"] / cap_uav) + q_leo * (result["added_leo"] / cap_leo)
                    cost = self.lyapunov_v * self._cost_of(config, result) + drift
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_target = target
                        best_ratio = float(ratio)
            targets.append(best_target)
            ratios.append(best_ratio)
            local_result = simulate_task(
                config, obs["user_pos"][user_idx], uav_pos, leo,
                obs["task_bits"][user_idx], obs["cycles_per_bit"][user_idx],
                0, 0.0, obs["uav_backlog"], obs["leo_backlog"],
            )
            gains.append(max(0.0, self._cost_of(config, local_result) - best_cost))
        return np.asarray(targets, dtype=int), np.asarray(ratios, dtype=float), np.asarray(gains, dtype=float)

# ---------------------------------------------------------------------------
# Genetic algorithm policy
# ---------------------------------------------------------------------------
class GeneticPolicy(BasePolicy):
    """Per-slot genetic algorithm over (move, targets, ratios)."""
    name = "genetic"
    pop_size = 60
    generations = 24
    def __init__(self, pop_size=None, generations=None):
        if pop_size is not None:
            self.pop_size = pop_size
        if generations is not None:
            self.generations = generations
    elite = 5
    mutation_rate = 0.25
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    def act(self, obs, rng, config):
        U = len(obs["task_bits"])
        pop = self._init_pop(obs, config, rng, U)
        fitness = self._fitness(obs, config, pop)
        for _ in range(self.generations):
            pop, fitness = self._evolve(obs, config, pop, fitness, rng, U)
        best = pop[int(np.argmin(fitness))]
        return {
            "move": best[0],
            "targets": best[1],
            "ratios": best[2],
        }
    def _init_pop(self, obs, config, rng, U):
        seeds = []
        tea = OptimizedPartialPolicy()
        tea_action = tea.act(obs, rng, config)
        seeds.append((tea_action["move"], tea_action["targets"].astype(int), tea_action["ratios"].astype(float)))
        centroid = _limited_move(obs["user_pos"].mean(axis=0) - obs["uav_pos"], config.uav_speed_max)
        seeds.append((centroid, np.ones(U, dtype=int), np.ones(U)))
        seeds.append((np.zeros(2), np.ones(U, dtype=int), np.ones(U)))
        seeds.append((np.zeros(2), np.zeros(U, dtype=int), np.zeros(U)))
        pop = []
        for seed in seeds:
            pop.append((np.asarray(seed[0], dtype=float).copy(), seed[1].copy(), seed[2].copy()))
        while len(pop) < self.pop_size:
            move = rng.uniform(-1.0, 1.0, size=2) * config.uav_speed_max
            targets = rng.integers(0, 3, size=U)
            ratios = rng.choice(self.ratio_grid, size=U)
            pop.append((move, targets.astype(int), ratios.astype(float)))
        return pop[: self.pop_size]
    def _fitness(self, obs, config, pop):
        P = len(pop)
        moves = np.stack([p[0] for p in pop])
        targets = np.stack([p[1] for p in pop])
        ratios = np.stack([p[2] for p in pop])
        total, _, _, _ = evaluate_actions_vec(config, obs, moves, targets, ratios)
        return total
    def _evolve(self, obs, config, pop, fitness, rng, U):
        order = np.argsort(fitness)
        new_pop = [pop[i] for i in order[: self.elite]]
        while len(new_pop) < self.pop_size:
            i1, i2 = rng.integers(0, len(pop), size=2)
            if fitness[i1] > fitness[i2]:
                i1, i2 = i2, i1
            parent_a, parent_b = pop[i1], pop[i2]
            move = np.where(rng.uniform(size=2) < 0.5, parent_a[0], parent_b[0])
            targets = np.where(rng.uniform(size=U) < 0.5, parent_a[1], parent_b[1]).astype(int)
            ratios = np.where(rng.uniform(size=U) < 0.5, parent_a[2], parent_b[2]).astype(float)
            if rng.uniform() < self.mutation_rate:
                move = move + rng.normal(0.0, 0.2 * config.uav_speed_max, size=2)
            target_mut = rng.uniform(size=U) < 0.1
            targets[target_mut] = rng.integers(0, 3, size=int(target_mut.sum()))
            ratio_mut = rng.uniform(size=U) < 0.15
            ratios[ratio_mut] = rng.choice(self.ratio_grid, size=int(ratio_mut.sum()))
            new_pop.append((move, targets, ratios))
        return new_pop, self._fitness(obs, config, new_pop)

# ---------------------------------------------------------------------------
# Particle swarm optimization policy (per-slot)
# ---------------------------------------------------------------------------
class PSOPolicy(BasePolicy):
    """Particle swarm optimization over the joint (move, target, ratio) action.

    Each particle encodes the UAV move, per-user target scores and per-user
    offload ratio. The fitness is the same vectorized look-ahead cost used by
    the genetic and exhaustive policies, so results are directly comparable.
    """

    name = "pso"

    def __init__(self, n_particles=40, generations=10, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _encode(self, config, move, targets, ratios):
        U = len(targets)
        scores = np.zeros((U, 3), dtype=float)
        for u in range(U):
            scores[u, int(targets[u])] = 1.0
        return np.concatenate([
            np.asarray(move, dtype=float) / max(config.uav_speed_max, 1.0),
            scores.reshape(-1),
            np.clip(np.asarray(ratios, dtype=float), 0.0, 1.0),
        ])

    def _decode(self, config, x):
        U = self.U
        move = np.clip(x[:2], -1.0, 1.0) * config.uav_speed_max
        scores = x[2:2 + 3 * U].reshape(U, 3)
        targets = np.argmax(scores, axis=1).astype(int)
        ratios = np.clip(x[2 + 3 * U:], 0.0, 1.0)
        return move, targets, ratios

    def act(self, obs, rng, config):
        U = self.U = len(obs["task_bits"])
        tea = OptimizedPartialPolicy()
        tea_action = tea.act(obs, rng, config)
        dim = 2 + 4 * U
        lb = np.concatenate([np.full(2, -1.0), np.zeros(3 * U), np.zeros(U)])
        ub = np.concatenate([np.full(2, 1.0), np.ones(3 * U), np.ones(U)])
        x = np.zeros((self.n_particles, dim))
        v = rng.uniform(-0.1, 0.1, size=(self.n_particles, dim))
        x[0] = self._encode(config, tea_action["move"], tea_action["targets"], tea_action["ratios"])
        x[1:] = rng.uniform(lb, ub, size=(self.n_particles - 1, dim))
        pbest = x.copy()
        moves, targets, ratios = self._decode_all(config, x)
        total, _, _, _ = evaluate_actions_vec(config, obs, moves, targets, ratios)
        pbest_cost = total.copy()
        gbest = int(np.argmin(total))
        for _ in range(self.generations):
            r1 = rng.uniform(0.0, 1.0, size=(self.n_particles, dim))
            r2 = rng.uniform(0.0, 1.0, size=(self.n_particles, dim))
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (x[gbest] - x)
            x = np.clip(x + v, lb, ub)
            moves, targets, ratios = self._decode_all(config, x)
            total, _, _, _ = evaluate_actions_vec(config, obs, moves, targets, ratios)
            improved = total < pbest_cost
            pbest[improved] = x[improved]
            pbest_cost[improved] = total[improved]
            gbest = int(np.argmin(pbest_cost))
        move, targets, ratios = self._decode(config, pbest[gbest])
        return {"move": move, "targets": targets.astype(int), "ratios": ratios.astype(float)}

    def _decode_all(self, config, x):
        U = self.U
        moves = np.clip(x[:, :2], -1.0, 1.0) * config.uav_speed_max
        scores = x[:, 2:2 + 3 * U].reshape(-1, U, 3)
        targets = np.argmax(scores, axis=2)
        ratios = np.clip(x[:, 2 + 3 * U:], 0.0, 1.0)
        return moves, targets, ratios


# ---------------------------------------------------------------------------
# Simulated annealing policy (per-slot)
# ---------------------------------------------------------------------------
class SimulatedAnnealingPolicy(BasePolicy):
    """Per-slot simulated annealing over the joint (move, target, ratio) action.

    Standard MEC baseline: a candidate solution is seeded from the TEA policy
    and iteratively perturbed; worse neighbours are accepted with the Metropolis
    probability ``exp(-dE / T)`` while the temperature anneals geometrically.
    """
    name = "sa"
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    def __init__(self, iters=600, t0=40.0, t_min=0.05, alpha=0.985):
        self.iters = iters
        self.t0 = t0
        self.t_min = t_min
        self.alpha = alpha

    def _candidate_moves(self, obs, config):
        tea = OptimizedPartialPolicy()
        _, _, gains = tea._choose_targets(obs, config, obs["uav_pos"])
        base = _limited_move(tea._trajectory_target(obs, config, gains) - obs["uav_pos"], config.uav_speed_max)
        centroid = _limited_move(obs["user_pos"].mean(axis=0) - obs["uav_pos"], config.uav_speed_max)
        return [np.zeros(2), 0.5 * centroid, centroid, 0.5 * base, base]

    def act(self, obs, rng, config):
        U = len(obs["task_bits"])
        tea = OptimizedPartialPolicy()
        seed = tea.act(obs, rng, config)
        move = np.asarray(seed["move"], dtype=float).copy()
        targets = np.asarray(seed["targets"], dtype=int).copy()
        ratios = np.asarray(seed["ratios"], dtype=float).copy()
        move_pool = self._candidate_moves(obs, config)
        cur_cost = float(evaluate_actions_vec(config, obs, move[None, :],
                                              targets[None, :], ratios[None, :])[0][0])
        best_move, best_targets, best_ratios = move.copy(), targets.copy(), ratios.copy()
        best_cost = cur_cost
        T = float(self.t0)
        speed = config.uav_speed_max
        for _ in range(self.iters):
            cand_move = move.copy()
            if rng.uniform() < 0.35:
                cand_move = move_pool[int(rng.integers(0, len(move_pool)))].copy()
            else:
                cand_move = _limited_move(cand_move + rng.normal(0.0, 0.15 * speed, size=2), speed)
            cand_targets = targets.copy()
            cand_ratios = ratios.copy()
            n_perturb = int(rng.integers(1, 4))
            for _ in range(n_perturb):
                u = int(rng.integers(0, U))
                if rng.uniform() < 0.5:
                    cand_targets[u] = int(rng.integers(0, 3))
                else:
                    cand_ratios[u] = float(rng.choice(self.ratio_grid))
            cost = float(evaluate_actions_vec(config, obs, cand_move[None, :],
                                              cand_targets[None, :], cand_ratios[None, :])[0][0])
            if cost <= cur_cost or rng.uniform() < np.exp((cur_cost - cost) / max(T, 1.0e-6)):
                move, targets, ratios, cur_cost = cand_move, cand_targets, cand_ratios, cost
            if cur_cost < best_cost:
                best_move, best_targets, best_ratios = move.copy(), targets.copy(), ratios.copy()
                best_cost = cur_cost
            T = max(T * self.alpha, self.t_min)
        return {"move": best_move, "targets": best_targets.astype(int),
                "ratios": best_ratios.astype(float)}


# ---------------------------------------------------------------------------
# Ant colony optimization policy (per-slot)
# ---------------------------------------------------------------------------
class AntColonyPolicy(BasePolicy):
    """Per-slot ant colony optimization for task offloading.

    Each ant builds a full (move, per-user target, per-user ratio) assignment.
    Choice probabilities follow the classic pheromone x heuristic rule; the
    pheromone trails are updated with an elitist deposit from the best ant.
    Standard MEC / V2X metaheuristic baseline.
    """
    name = "aco"
    ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    def __init__(self, n_ants=30, iterations=6, rho=0.35, alpha=1.0, beta=2.0, q0=0.9):
        self.n_ants = n_ants
        self.iterations = iterations
        self.rho = rho
        self.alpha_p = alpha
        self.beta = beta
        self.q0 = q0

    def _action_codes(self):
        # (target, ratio) combinations: local(r=0), uav(4 ratios), leo(4 ratios)
        codes = [(0, 0.0)]
        for tgt in (1, 2):
            for r in (0.25, 0.5, 0.75, 1.0):
                codes.append((tgt, r))
        return codes

    def _candidate_moves(self, obs, config):
        tea = OptimizedPartialPolicy()
        _, _, gains = tea._choose_targets(obs, config, obs["uav_pos"])
        base = _limited_move(tea._trajectory_target(obs, config, gains) - obs["uav_pos"], config.uav_speed_max)
        centroid = _limited_move(obs["user_pos"].mean(axis=0) - obs["uav_pos"], config.uav_speed_max)
        return [np.zeros(2), 0.5 * centroid, centroid, 0.5 * base, base]

    def act(self, obs, rng, config):
        U = len(obs["task_bits"])
        codes = self._action_codes()
        n_codes = len(codes)
        moves_pool = self._candidate_moves(obs, config)
        # heuristic desirability: inverse per-user look-ahead cost
        eta = np.zeros((U, n_codes), dtype=float)
        leo_idx = [nearest_leo_index(obs["leo_pos"], obs["user_pos"][u]) for u in range(U)]
        for u in range(U):
            if obs["task_bits"][u] <= 1.0e-6:
                eta[u, 0] = 1.0
                continue
            for a, (tgt, rat) in enumerate(codes):
                if tgt == 0:
                    res = simulate_task(config, obs["user_pos"][u], obs["uav_pos"],
                                        obs["leo_pos"][leo_idx[u]], obs["task_bits"][u],
                                        obs["cycles_per_bit"][u], 0, 0.0,
                                        obs["uav_backlog"], obs["leo_backlog"])
                else:
                    res = simulate_task(config, obs["user_pos"][u], obs["uav_pos"],
                                        obs["leo_pos"][leo_idx[u]], obs["task_bits"][u],
                                        obs["cycles_per_bit"][u], tgt, rat,
                                        obs["uav_backlog"], obs["leo_backlog"])
                cost = (config.latency_weight * res["latency"]
                        + config.energy_weight * res["energy"]
                        + (config.drop_penalty * 10.0 if res["dropped"] else 0.0))
                eta[u, a] = 1.0 / (1.0 + cost)
        tau = np.ones((U, n_codes), dtype=float)
        best_action = None
        best_cost = np.inf
        for it in range(self.iterations):
            ants_moves = []
            ants_targets = np.zeros((self.n_ants, U), dtype=int)
            ants_ratios = np.zeros((self.n_ants, U), dtype=float)
            for a in range(self.n_ants):
                move = moves_pool[int(rng.integers(0, len(moves_pool)))]
                targets = np.zeros(U, dtype=int)
                ratios = np.zeros(U, dtype=float)
                for u in range(U):
                    if obs["task_bits"][u] <= 1.0e-6:
                        continue
                    scores = (tau[u] ** self.alpha_p) * (eta[u] ** self.beta)
                    if scores.sum() <= 0.0:
                        choice = 0
                    elif rng.uniform() < self.q0:
                        choice = int(np.argmax(scores))
                    else:
                        probs = scores / scores.sum()
                        choice = int(rng.choice(n_codes, p=probs))
                    targets[u] = codes[choice][0]
                    ratios[u] = codes[choice][1]
                ants_moves.append(move)
                ants_targets[a] = targets
                ants_ratios[a] = ratios
            costs = evaluate_actions_vec(config, obs, np.stack(ants_moves),
                                         ants_targets, ants_ratios)[0]
            best_a = int(np.argmin(costs))
            if costs[best_a] < best_cost:
                best_cost = float(costs[best_a])
                best_action = (ants_moves[best_a].copy(), ants_targets[best_a].copy(),
                               ants_ratios[best_a].copy())
            # pheromone update: evaporate then elitist deposit
            tau *= (1.0 - self.rho)
            deposit = 1.0 / max(best_cost, 1.0e-9)
            for u in range(U):
                if obs["task_bits"][u] <= 1.0e-6:
                    continue
                tgt = int(best_action[1][u])
                rat = float(best_action[2][u])
                for a, (ct, cr) in enumerate(codes):
                    if ct == tgt and abs(cr - rat) < 1.0e-9:
                        tau[u, a] += self.rho * deposit
        return {"move": best_action[0], "targets": best_action[1].astype(int),
                "ratios": best_action[2].astype(float)}


# ---------------------------------------------------------------------------
# Exhaustive / near-optimal search policy
# ---------------------------------------------------------------------------
class ExhaustiveOptimalPolicy(BasePolicy):
    """Per-slot exhaustive search (joint action grid), near-optimal upper bound.
    For U > ``max_joint_users`` it falls back to coordinate descent over the
    per-user action grid, which is still a strong near-optimal benchmark.
    """
    name = "exhaustive_optimal"
    max_joint_users = 4
    ratio_grid = np.array([0.0, 0.5, 1.0], dtype=float)
    def act(self, obs, rng, config):
        U = len(obs["task_bits"])
        moves = self._candidate_moves(obs, config)
        if U <= self.max_joint_users:
            best = self._joint_search(obs, config, moves)
        else:
            best = self._coordinate_descent(obs, config, moves)
        return {"move": best[0], "targets": best[1], "ratios": best[2]}
    def _candidate_moves(self, obs, config):
        tea = OptimizedPartialPolicy()
        _, _, gains = tea._choose_targets(obs, config, obs["uav_pos"])
        target = tea._trajectory_target(obs, config, gains)
        base = _limited_move(target - obs["uav_pos"], config.uav_speed_max)
        centroid = _limited_move(obs["user_pos"].mean(axis=0) - obs["uav_pos"], config.uav_speed_max)
        return [
            np.zeros(2),
            0.5 * base, base,
            0.5 * centroid,
        ]
    def _joint_search(self, obs, config, moves):
        U = len(obs["task_bits"])
        combos = np.array(np.meshgrid(*[np.arange(3 * len(self.ratio_grid)) for _ in range(U)])).reshape(U, -1).T
        target_grid = np.tile(np.repeat(np.arange(3), len(self.ratio_grid)), (combos.shape[0], 1))
        ratio_grid = np.tile(np.tile(self.ratio_grid, 3), (combos.shape[0], 1))
        targets = np.zeros_like(combos)
        ratios = np.zeros_like(combos, dtype=float)
        for col in range(U):
            targets[:, col] = combos[:, col] // len(self.ratio_grid)
            ratios[:, col] = self.ratio_grid[combos[:, col] % len(self.ratio_grid)]
        best = None
        for move in moves:
            total, _, _, _ = evaluate_actions_vec(config, obs, np.asarray(move)[None, :], targets, ratios)
            idx = int(np.argmin(total))
            if best is None or total[idx] < best[0]:
                best = (float(total[idx]), np.asarray(move, dtype=float), targets[idx].copy(), ratios[idx].copy())
        return best[1:]
    def _coordinate_descent(self, obs, config, moves):
        U = len(obs["task_bits"])
        tea = OptimizedPartialPolicy()
        tea_action = tea.act(obs, rng=np.random.default_rng(0), config=config)
        targets = tea_action["targets"].astype(int)
        ratios = tea_action["ratios"].astype(float)
        action_ids = np.arange(3 * len(self.ratio_grid))
        a_targets = (action_ids // len(self.ratio_grid)).astype(int)
        a_ratios = self.ratio_grid[action_ids % len(self.ratio_grid)]
        best_move = np.zeros(2)
        best_total = np.inf
        for move in moves:
            for _ in range(3):
                improved = False
                for u in range(U):
                    T = np.tile(targets, (len(action_ids), 1))
                    R = np.tile(ratios, (len(action_ids), 1))
                    T[:, u] = a_targets
                    R[:, u] = a_ratios
                    total, _, _, _ = evaluate_actions_vec(config, obs, np.asarray(move)[None, :], T, R)
                    best_a = int(np.argmin(total))
                    if T[best_a, u] != targets[u] or R[best_a, u] != ratios[u]:
                        targets[u] = T[best_a, u]
                        ratios[u] = R[best_a, u]
                        improved = True
                if not improved:
                    break
            total, _, _, _ = evaluate_actions_vec(config, obs, np.asarray(move)[None, :], targets[None, :], ratios[None, :])
            if total[0] < best_total:
                best_total = float(total[0])
                best_move = np.asarray(move, dtype=float)
        return best_move, targets.copy(), ratios.copy()
