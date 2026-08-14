"""Shared physics for the UAV-LEO offloading experiment.
Single source of truth used by both the simulator (env.py) and every policy
(baselines.py / algorithms.py / learning.py). Keeping one implementation
guarantees that look-ahead cost estimates used by heuristics and the actual
environment rollout are identical.
Two channel models are supported via ``config.channel_model``:
- ``free_space`` : legacy ``1/d^2`` gain used by the original experiment line.
- ``los_nlos``   : 3GPP-style elevation-dependent LoS probability plus
                   LoS/NLoS path loss (carrier-frequency dependent) and
                   antenna gains. Used by the v2x experiment line.
``config.use_queues`` enables shared UAV/LEO compute queues: offloaded cycles
are added to a backlog that drains at the server CPU rate, so per-slot
decisions interact across time slots. When disabled, the model reduces to the
legacy per-slot independent execution.
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Channel model
# ---------------------------------------------------------------------------
def los_probability(config, elevation_deg):
    """Elevation-dependent LoS probability (3GPP TR 36.777 style)."""
    a = config.los_param_a
    b = config.los_param_b
    return 1.0 / (1.0 + a * math.exp(-b * (elevation_deg - a)))

def path_loss_db(config, distance, los):
    d = max(float(distance), 1.0)
    fspl = 20.0 * math.log10(d) + 20.0 * math.log10(config.carrier_hz) - 147.55
    if getattr(config, "path_loss_exp", 2.0) != 2.0:
        fspl = fspl + 10.0 * (config.path_loss_exp - 2.0) * math.log10(d)
    excess = config.path_loss_los if los else config.path_loss_nlos
    return fspl + excess

def channel_gain_linear(config, distance, los):
    return 10.0 ** (-path_loss_db(config, distance, los) / 10.0)

def _elevation_deg(horizontal_dist, altitude):
    return math.degrees(math.atan2(altitude, max(horizontal_dist, 1.0)))

def _expected_gain_los_nlos(config, distance_3d, altitude, extra_gain_db):
    hdist = math.sqrt(max(distance_3d ** 2 - altitude ** 2, 0.0))
    theta = _elevation_deg(hdist, altitude)
    plos = los_probability(config, theta)
    g_los = channel_gain_linear(config, distance_3d, True)
    g_nlos = channel_gain_linear(config, distance_3d, False)
    gain = plos * g_los + (1.0 - plos) * g_nlos
    return gain * (10.0 ** (extra_gain_db / 10.0))

def shannon_rate(config, distance_square, tx_power, extra_gain_db=0.0):
    """Shannon rate given squared 3D distance (legacy free-space model)."""
    channel_gain = 1.0 / max(distance_square, 1.0)
    snr = tx_power * channel_gain / config.noise_watt
    return max(config.bandwidth_hz * math.log2(1.0 + snr), 1.0)

def rate_user_uav(config, user_pos, uav_pos):
    d3_square = float(np.sum((user_pos - uav_pos) ** 2)) + config.uav_altitude ** 2
    if config.channel_model == "los_nlos":
        d3 = math.sqrt(d3_square)
        gain = _expected_gain_los_nlos(config, d3, config.uav_altitude, config.user_uav_gain_db)
        snr = config.user_tx_power_watt * gain / config.noise_watt
    else:
        exp = getattr(config, "path_loss_exp", 2.0)
        snr = config.user_tx_power_watt * (1.0 / max(d3_square ** (exp / 2.0), 1.0)) / config.noise_watt
    return max(config.bandwidth_hz * math.log2(1.0 + snr), 1.0)

def rate_uav_leo(config, uav_pos, leo_pos):
    if getattr(config, "backhaul_mbps", None) is not None:
        return max(config.backhaul_mbps * 1.0e6, 1.0)
    d3_square = float(np.sum((uav_pos - leo_pos) ** 2)) + config.leo_altitude ** 2
    if config.channel_model == "los_nlos":
        d3 = math.sqrt(d3_square)
        gain = _expected_gain_los_nlos(config, d3, config.leo_altitude, config.uav_leo_gain_db)
        snr = config.uav_tx_power_watt * gain / config.noise_watt
    else:
        exp = getattr(config, "path_loss_exp", 2.0)
        snr = config.uav_tx_power_watt * (1.0 / max(d3_square ** (exp / 2.0), 1.0)) / config.noise_watt
    return max(config.bandwidth_hz * math.log2(1.0 + snr), 1.0)

# ---------------------------------------------------------------------------
# Task execution model
# ---------------------------------------------------------------------------
def simulate_task(
    config,
    user_pos,
    uav_pos,
    leo_pos,
    task_bits,
    cycles_per_bit,
    target,
    ratio,
    uav_backlog,
    leo_backlog,
    user_idx=0,
):
    """Return latency/energy/queue delta for one (user, target, ratio) choice.
    Returns a dict with keys: latency, energy, added_uav, added_leo, dropped.
    ``dropped`` is True when the remote part cannot finish within the deadline
    (queue + service + transmission exceed the deadline); the task then counts
    as a failure and no cycles are added to the server queue.
    """
    dt = config.slot_seconds
    bits = float(task_bits)
    cycles = bits * float(cycles_per_bit)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    deadline = float(config.success_deadline_s)
    local_latency = (1.0 - ratio) * cycles / config.user_cpu_cycles_per_s
    if config.power_based_energy:
        local_energy = config.user_device_power_watt * local_latency
    else:
        local_energy = (
            config.compute_energy_coeff
            * ((1.0 - ratio) * cycles)
            * (config.user_cpu_cycles_per_s ** 2)
        )
    if target == 0 or ratio <= 1.0e-6:
        latency = cycles / config.user_cpu_cycles_per_s
        if config.power_based_energy:
            energy = config.user_device_power_watt * latency
        else:
            energy = (
                config.compute_energy_coeff * cycles * (config.user_cpu_cycles_per_s ** 2)
            )
        return {
            "latency": latency,
            "energy": energy,
            "server_energy": 0.0,
            "added_uav": 0.0,
            "added_leo": 0.0,
            "dropped": latency > deadline,
            "target": "local",
        }
    if target == 1:
        rate = rate_user_uav(config, user_pos, uav_pos)
        server_cpu = config.uav_cpu_cycles_per_s
        backlog = float(uav_backlog)
        added_key = "added_uav"
        target_name = "uav"
    else:
        rate = min(
            rate_user_uav(config, user_pos, uav_pos),
            rate_uav_leo(config, uav_pos, leo_pos),
        )
        server_cpu = config.leo_cpu_cycles_per_s
        backlog = float(leo_backlog)
        added_key = "added_leo"
        target_name = "leo"
    tx_latency = ratio * bits / max(rate, 1.0)
    wait_latency = max(backlog - server_cpu * dt, 0.0) / server_cpu
    service_latency = ratio * cycles / server_cpu
    latency = max(local_latency, tx_latency + wait_latency + service_latency)
    dropped = latency > deadline
    tx_energy = config.user_tx_power_watt * tx_latency
    if config.power_based_energy:
        server_power = config.leo_power_watt if target == 2 else config.uav_power_watt
        server_energy = server_power * (ratio * cycles / server_cpu)
        remote_energy = 0.0
    else:
        server_energy = (
            config.compute_energy_coeff * (ratio * cycles) * (server_cpu ** 2)
        )
        remote_energy = server_energy
    if dropped:
        # The remote part is abandoned; local part still executed.
        latency = float(deadline)
        energy = local_energy
        return {
            "latency": latency,
            "energy": energy,
            "server_energy": 0.0,
            "added_uav": 0.0,
            "added_leo": 0.0,
            "dropped": True,
            "target": target_name,
        }
    if config.power_based_energy:
        energy = local_energy + tx_energy
    else:
        energy = local_energy + tx_energy + remote_energy
    return {
        "latency": latency,
        "energy": energy,
        "server_energy": server_energy,
        added_key: ratio * cycles,
        "added_uav": ratio * cycles if target == 1 else 0.0,
        "added_leo": ratio * cycles if target == 2 else 0.0,
        "dropped": False,
        "target": target_name,
    }

def simulate_task_multi(
    config,
    user_pos,
    uav_pos_array,
    leo_pos,
    task_bits,
    cycles_per_bit,
    target,
    ratio,
    uav_backlog_array,
    leo_backlog,
    user_idx=0,
):
    """Multi-UAV version of :func:`simulate_task`.

    ``target`` codes: 0 = local, 1..K = UAV k, K+1 = LEO. The LEO route uses the
    UAV relay with the best user->UAV->LEO combined rate. Returns the same dict
    keys as :func:`simulate_task`, but ``added_uav`` is a length-K array so the
    environment can credit the right UAV queue."""
    dt = config.slot_seconds
    bits = float(task_bits)
    cycles = bits * float(cycles_per_bit)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    deadline = float(config.success_deadline_s)
    K = len(uav_pos_array)
    uav_pos_array = np.asarray(uav_pos_array, dtype=float)
    local_latency = (1.0 - ratio) * cycles / config.user_cpu_cycles_per_s
    if config.power_based_energy:
        local_energy = config.user_device_power_watt * local_latency
    else:
        local_energy = (
            config.compute_energy_coeff
            * ((1.0 - ratio) * cycles)
            * (config.user_cpu_cycles_per_s ** 2)
        )
    if target == 0 or ratio <= 1.0e-6:
        latency = cycles / config.user_cpu_cycles_per_s
        if config.power_based_energy:
            energy = config.user_device_power_watt * latency
        else:
            energy = config.compute_energy_coeff * cycles * (config.user_cpu_cycles_per_s ** 2)
        return {
            "latency": latency,
            "energy": energy,
            "server_energy": 0.0,
            "added_uav": np.zeros(K, dtype=float),
            "added_leo": 0.0,
            "dropped": latency > deadline,
            "target": "local",
        }
    if 1 <= target <= K:
        k = int(target) - 1
        rate = rate_user_uav(config, user_pos, uav_pos_array[k])
        server_cpu = config.uav_cpu_cycles_per_s
        backlog = float(uav_backlog_array[k])
        added_uav = np.zeros(K, dtype=float)
        added_uav[k] = ratio * cycles
        target_name = "uav%d" % k
    else:
        relays = np.array([
            min(rate_user_uav(config, user_pos, uav_pos_array[k]),
                rate_uav_leo(config, uav_pos_array[k], leo_pos))
            for k in range(K)
        ])
        k = int(np.argmax(relays))
        rate = float(relays[k])
        server_cpu = config.leo_cpu_cycles_per_s
        backlog = float(leo_backlog)
        added_uav = np.zeros(K, dtype=float)
        added_leo = ratio * cycles
        target_name = "leo"
    tx_latency = ratio * bits / max(rate, 1.0)
    wait_latency = max(backlog - server_cpu * dt, 0.0) / server_cpu
    service_latency = ratio * cycles / server_cpu
    latency = max(local_latency, tx_latency + wait_latency + service_latency)
    dropped = latency > deadline
    tx_energy = config.user_tx_power_watt * tx_latency
    if config.power_based_energy:
        server_power = config.leo_power_watt if target == K + 1 else config.uav_power_watt
        server_energy = server_power * (ratio * cycles / server_cpu)
        remote_energy = 0.0
    else:
        server_energy = config.compute_energy_coeff * (ratio * cycles) * (server_cpu ** 2)
        remote_energy = server_energy
    if dropped:
        latency = float(deadline)
        energy = local_energy
        return {
            "latency": latency,
            "energy": energy,
            "server_energy": 0.0,
            "added_uav": np.zeros(K, dtype=float),
            "added_leo": 0.0,
            "dropped": True,
            "target": target_name,
        }
    if config.power_based_energy:
        energy = local_energy + tx_energy
    else:
        energy = local_energy + tx_energy + remote_energy
    return {
        "latency": latency,
        "energy": energy,
        "server_energy": server_energy,
        "added_uav": added_uav,
        "added_leo": added_leo if target == K + 1 else 0.0,
        "dropped": False,
        "target": target_name,
    }
def flight_energy(config, moved):
    """Propulsion energy for one slot: hover + speed-dependent terms."""
    speed = moved / max(config.slot_seconds, 1.0e-9)
    if config.speed_cubed_energy:
        return config.hover_power_watt * config.slot_seconds + (
            config.drag_coeff_watt_per_m3s3 * (speed ** 3) * config.slot_seconds
        )
    return config.hover_power_watt * config.slot_seconds + config.move_energy_coeff * moved

def nearest_leo_index(leo_pos, user_pos):
    d = np.linalg.norm(leo_pos - user_pos, axis=1)
    return int(np.argmin(d))

# ---------------------------------------------------------------------------
# Vectorized channel model (for environment stepping and search policies)
# ---------------------------------------------------------------------------
def rate_user_uav_vec(config, user_pos, uav_pos):
    """Vectorized user->UAV rate; inputs broadcast to (..., 2)."""
    uav_pos = np.asarray(uav_pos, dtype=float)
    d3_sq = np.sum((np.asarray(user_pos, dtype=float) - uav_pos) ** 2, axis=-1) + config.uav_altitude ** 2
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
        exp = getattr(config, "path_loss_exp", 2.0)
        snr = config.user_tx_power_watt * (1.0 / np.maximum(d3_sq ** (exp / 2.0), 1.0)) / config.noise_watt
    return np.maximum(config.bandwidth_hz * np.log2(1.0 + snr), 1.0)

def rate_uav_leo_vec(config, uav_pos, leo_pos):
    """Vectorized UAV->LEO rate; inputs broadcast to (..., 2)."""
    if getattr(config, "backhaul_mbps", None) is not None:
        return np.full(np.broadcast(np.asarray(uav_pos, dtype=float)[..., 0],
                                    np.asarray(leo_pos, dtype=float)[..., 0]).shape,
                       max(config.backhaul_mbps * 1.0e6, 1.0), dtype=float)
    uav_pos = np.asarray(uav_pos, dtype=float)
    d3_sq = np.sum((np.asarray(leo_pos, dtype=float) - uav_pos) ** 2, axis=-1) + config.leo_altitude ** 2
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
        exp = getattr(config, "path_loss_exp", 2.0)
        snr = config.uav_tx_power_watt * (1.0 / np.maximum(d3_sq ** (exp / 2.0), 1.0)) / config.noise_watt
    return np.maximum(config.bandwidth_hz * np.log2(1.0 + snr), 1.0)
