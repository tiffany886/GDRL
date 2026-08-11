from dataclasses import dataclass, replace


@dataclass
class UavLeoConfig:
    seed: int = 73
    users: int = 5
    leos: int = 4
    horizon: int = 100
    episodes: int = 20
    difficulty: str = "easy"
    ablation: str = "none"
    area_size: float = 1000.0
    slot_seconds: float = 1.0
    uav_speed_max: float = 35.0
    uav_altitude: float = 120.0
    leo_altitude: float = 500000.0
    bandwidth_hz: float = 1.0e6
    noise_watt: float = 1.0e-13
    user_tx_power_watt: float = 0.2
    uav_tx_power_watt: float = 0.5
    uav_cpu_cycles_per_s: float = 12.0e9
    leo_cpu_cycles_per_s: float = 80.0e9
    user_cpu_cycles_per_s: float = 1.5e9
    task_bits_min: float = 3.0e5
    task_bits_max: float = 1.2e6
    cycles_per_bit_min: float = 500.0
    cycles_per_bit_max: float = 1200.0
    hover_power_watt: float = 80.0
    move_energy_coeff: float = 4.0
    compute_energy_coeff: float = 1.0e-27
    latency_weight: float = 1.0
    energy_weight: float = 0.001
    success_deadline_s: float = 0.8
    # --- v2x (vehicular mobility + LoS/NLoS + shared compute queues) ---
    mobility: bool = False
    user_speed_max: float = 10.0
    channel_model: str = "free_space"          # free_space | los_nlos
    carrier_hz: float = 2.4e9
    los_param_a: float = 9.61
    los_param_b: float = 0.16
    path_loss_los: float = 1.6
    path_loss_nlos: float = 23.0
    path_loss_exp: float = 2.0          # log-distance exponent (2.0 = FSPL)
    backhaul_mbps: float = None        # fixed UAV->LEO backhaul capacity (None = channel model)
    user_uav_gain_db: float = 6.0
    uav_leo_gain_db: float = 50.0
    use_queues: bool = False
    drop_penalty: float = 0.0
    speed_cubed_energy: bool = False
    drag_coeff_watt_per_m3s3: float = 0.02
    task_arrival_prob: float = 1.0
    uav_battery_per_slot: float = 0.0
    power_based_energy: bool = False
    user_device_power_watt: float = 1.5
    uav_power_watt: float = 100.0
    leo_power_watt: float = 1000.0
    hotspot_motion: bool = False
    hotspot_speed: float = 18.0
    hotspot_radius: float = 220.0
    hotspot_arrival_prob: float = 0.95
    base_arrival_prob: float = 0.3
    hotspot_size_boost: float = 0.0
    road_network: bool = False
    n_roads: int = 3
    road_half_len: float = 350.0
    fixed_uav: bool = False
    no_flight_energy: bool = False
    full_offload_only: bool = False
    output_root: str = "experiments/uav_leo"


DIFFICULTY_PRESETS = {
    "easy": {
        "task_bits_min": 3.0e5,
        "task_bits_max": 1.2e6,
        "cycles_per_bit_min": 500.0,
        "cycles_per_bit_max": 1200.0,
        "bandwidth_hz": 1.0e6,
        "success_deadline_s": 0.8,
        "energy_weight": 0.001,
    },
    "medium": {
        "task_bits_min": 1.0e6,
        "task_bits_max": 3.0e6,
        "cycles_per_bit_min": 700.0,
        "cycles_per_bit_max": 1500.0,
        "bandwidth_hz": 1.0e6,
        "success_deadline_s": 0.5,
        "energy_weight": 0.001,
    },
    "hard": {
        "task_bits_min": 2.0e6,
        "task_bits_max": 6.0e6,
        "cycles_per_bit_min": 900.0,
        "cycles_per_bit_max": 1800.0,
        "bandwidth_hz": 8.0e5,
        "success_deadline_s": 0.35,
        "energy_weight": 0.0008,
    },
    "stress": {
        "task_bits_min": 4.0e6,
        "task_bits_max": 1.0e7,
        "cycles_per_bit_min": 1000.0,
        "cycles_per_bit_max": 2200.0,
        "bandwidth_hz": 5.0e5,
        "success_deadline_s": 0.25,
        "energy_weight": 0.0006,
    },
    # --- v2x line: vehicular users, LoS/NLoS channel, shared queues ---
    "v2x_easy": {
        "task_bits_min": 4.0e5,
        "task_bits_max": 1.2e6,
        "cycles_per_bit_min": 500.0,
        "cycles_per_bit_max": 1000.0,
        "bandwidth_hz": 6.0e6,
        "success_deadline_s": 2.5,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "uav_cpu_cycles_per_s": 12.0e9,
        "leo_cpu_cycles_per_s": 80.0e9,
        "mobility": True,
        "user_speed_max": 12.0,
        "channel_model": "los_nlos",
        "use_queues": True,
        "drop_penalty": 4.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 0.65,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
    },
    "v2x_medium": {
        "task_bits_min": 1.0e6,
        "task_bits_max": 2.6e6,
        "cycles_per_bit_min": 700.0,
        "cycles_per_bit_max": 1400.0,
        "bandwidth_hz": 4.0e6,
        "success_deadline_s": 2.0,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "uav_cpu_cycles_per_s": 12.0e9,
        "leo_cpu_cycles_per_s": 80.0e9,
        "mobility": True,
        "user_speed_max": 14.0,
        "channel_model": "los_nlos",
        "use_queues": True,
        "drop_penalty": 4.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 0.65,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
    },
    "v2x_hard": {
        "task_bits_min": 1.6e6,
        "task_bits_max": 4.0e6,
        "cycles_per_bit_min": 900.0,
        "cycles_per_bit_max": 1700.0,
        "bandwidth_hz": 2.5e6,
        "success_deadline_s": 1.5,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "uav_cpu_cycles_per_s": 12.0e9,
        "leo_cpu_cycles_per_s": 80.0e9,
        "mobility": True,
        "user_speed_max": 16.0,
        "channel_model": "los_nlos",
        "use_queues": True,
        "drop_penalty": 4.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 0.65,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
    },
    "v2x_stress": {
        "task_bits_min": 2.5e6,
        "task_bits_max": 6.0e6,
        "cycles_per_bit_min": 1000.0,
        "cycles_per_bit_max": 1900.0,
        "bandwidth_hz": 2.0e6,
        "success_deadline_s": 1.2,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "uav_cpu_cycles_per_s": 12.0e9,
        "leo_cpu_cycles_per_s": 80.0e9,
        "mobility": True,
        "user_speed_max": 18.0,
        "channel_model": "los_nlos",
        "use_queues": True,
        "drop_penalty": 4.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 0.65,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
    },
    # --- hotspot line: v2x mechanics plus a moving traffic hotspot. Workload
    # is spatially concentrated and moves along a road, so the UAV must budget
    # its flight energy across the horizon (battery-limited) while tracking the
    # hotspot and splitting load between UAV and LEO queues. ---
    "v2x_hotspot_hard": {
        "task_bits_min": 1.2e6,
        "task_bits_max": 3.0e6,
        "cycles_per_bit_min": 900.0,
        "cycles_per_bit_max": 1600.0,
        "bandwidth_hz": 1.8e6,
        "success_deadline_s": 0.65,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "user_cpu_cycles_per_s": 1.5e9,
        "uav_cpu_cycles_per_s": 13.0e9,
        "leo_cpu_cycles_per_s": 10.0e9,
        "drag_coeff_watt_per_m3s3": 0.004,
        "mobility": True,
        "user_speed_max": 16.0,
        "channel_model": "los_nlos",
        "path_loss_exp": 3.0,
        "backhaul_mbps": 7.0,
        "use_queues": True,
        "drop_penalty": 6.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 1.0,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
        "hotspot_motion": True,
        "hotspot_speed": 18.0,
        "hotspot_radius": 130.0,
        "hotspot_arrival_prob": 1.0,
        "base_arrival_prob": 0.05,
        "hotspot_size_boost": 0.2,
    },
    "v2x_hotspot_stress": {
        "task_bits_min": 1.2e6,
        "task_bits_max": 3.2e6,
        "cycles_per_bit_min": 900.0,
        "cycles_per_bit_max": 1700.0,
        "bandwidth_hz": 2.0e6,
        "success_deadline_s": 0.6,
        "energy_weight": 0.001,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.2,
        "uav_tx_power_watt": 0.5,
        "user_cpu_cycles_per_s": 1.5e9,
        "uav_cpu_cycles_per_s": 15.0e9,
        "leo_cpu_cycles_per_s": 11.0e9,
        "drag_coeff_watt_per_m3s3": 0.004,
        "mobility": True,
        "user_speed_max": 18.0,
        "channel_model": "los_nlos",
        "path_loss_exp": 3.0,
        "backhaul_mbps": 6.5,
        "use_queues": True,
        "drop_penalty": 6.0,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "uav_speed_max": 25.0,
        "task_arrival_prob": 1.0,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
        "hotspot_motion": True,
        "hotspot_speed": 20.0,
        "hotspot_radius": 120.0,
        "hotspot_arrival_prob": 1.0,
        "base_arrival_prob": 0.05,
        "hotspot_size_boost": 0.25,
    },
}


ABLATION_PRESETS = {
    "none": {},
    "fixed_uav": {"fixed_uav": True},
    "no_flight_energy": {"no_flight_energy": True},
    "full_offload_only": {"full_offload_only": True},
    "no_hotspot": {"hotspot_motion": False, "hotspot_arrival_prob": 0.0,
                   "base_arrival_prob": 1.0, "hotspot_size_boost": 0.0},
}


def make_config(difficulty="easy", ablation="none", **overrides):
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    if ablation not in ABLATION_PRESETS:
        raise ValueError(f"Unknown ablation: {ablation}")
    cfg = UavLeoConfig(difficulty=difficulty, ablation=ablation)
    cfg = replace(cfg, **DIFFICULTY_PRESETS[difficulty])
    cfg = replace(cfg, **ABLATION_PRESETS[ablation])
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    if clean_overrides:
        cfg = replace(cfg, **clean_overrides)
    return cfg


def small_config():
    return make_config("easy")

