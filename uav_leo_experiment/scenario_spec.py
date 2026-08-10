"""Authoritative scenario specifications for the UAV-LEO hotspot line.

These dicts are applied as explicit config overrides by train_drl.py,
run_experiment.py and run_sweep.py so that the published results do not
depend on the mutable ``DIFFICULTY_PRESETS`` in config.py (which can be
edited by other experiment scripts). Values were tuned so that BOTH
dimensions of the problem matter on the hotspot scenarios:

- trajectory: hovering (greedy one-step TEA) loses clearly to demand-aware
  centroid tracking (distance-sensitive user-UAV access link, strong LEO
  backhaul); follow_tea beats hover_tea by ~40% in total reward;
- offloading decisions: all-to-LEO / all-to-UAV / all-local lose clearly to
  the joint TEA target+ratio search (mixed local-viable small tasks and
  offload-only large tasks), and follow_tea beats follow_leo by ~55%.

Link tuning: with ``bandwidth_hz=1.0 MHz``, ``path_loss_exp=3.0`` and
``user_tx_power_watt=0.3`` the user->UAV Shannon rate is ~7.8 Mbps at 25 m,
~6.7 Mbps at 100 m, ~4.7 Mbps at 200 m and ~2.6 Mbps at 300 m.  Large tasks
(3 Mb, 1700 cycles/bit) therefore need the UAV to stay close to the hotspot,
while small tasks (0.8 Mb, 900 cycles/bit) are locally viable on the 5 GHz
user CPU.  The LEO provides a fixed-capacity high-gain backhaul
(``backhaul_mbps``) and very large compute capacity (satellite edge server);
the UAV is the distance-sensitive access relay with a 50-60 GHz edge server
whose shared compute queue makes per-user load balancing matter.
"""

MY_SCENARIOS = {
    "v2x_hotspot_hard": {
        "users": 12,
        "leos": 4,
        "horizon": 30,
        "area_size": 1000.0,
        "slot_seconds": 1.0,
        "uav_speed_max": 25.0,
        "uav_altitude": 120.0,
        "leo_altitude": 500000.0,
        "bandwidth_hz": 1.0e6,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.3,
        "uav_tx_power_watt": 0.5,
        "user_cpu_cycles_per_s": 5.0e9,
        "uav_cpu_cycles_per_s": 50.0e9,
        "leo_cpu_cycles_per_s": 1.0e12,
        "task_bits_min": 0.8e6,
        "task_bits_max": 3.0e6,
        "cycles_per_bit_min": 900.0,
        "cycles_per_bit_max": 1700.0,
        "success_deadline_s": 0.65,
        "latency_weight": 1.0,
        "energy_weight": 0.001,
        "drop_penalty": 4.0,
        "mobility": True,
        "user_speed_max": 16.0,
        "channel_model": "los_nlos",
        "path_loss_exp": 3.0,
        "backhaul_mbps": 20.0,
        "use_queues": True,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "drag_coeff_watt_per_m3s3": 0.008,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
        "hotspot_motion": True,
        "hotspot_speed": 18.0,
        "hotspot_radius": 100.0,
        "hotspot_arrival_prob": 1.0,
        "base_arrival_prob": 0.05,
        "hotspot_size_boost": 1.2,
    },
    "v2x_hotspot_stress": {
        "users": 16,
        "leos": 4,
        "horizon": 30,
        "area_size": 1000.0,
        "slot_seconds": 1.0,
        "uav_speed_max": 25.0,
        "uav_altitude": 120.0,
        "leo_altitude": 500000.0,
        "bandwidth_hz": 1.1e6,
        "noise_watt": 2.0e-13,
        "user_tx_power_watt": 0.3,
        "uav_tx_power_watt": 0.5,
        "user_cpu_cycles_per_s": 5.0e9,
        "uav_cpu_cycles_per_s": 60.0e9,
        "leo_cpu_cycles_per_s": 1.0e12,
        "task_bits_min": 0.8e6,
        "task_bits_max": 3.0e6,
        "cycles_per_bit_min": 1000.0,
        "cycles_per_bit_max": 1900.0,
        "success_deadline_s": 0.6,
        "latency_weight": 1.0,
        "energy_weight": 0.001,
        "drop_penalty": 4.0,
        "mobility": True,
        "user_speed_max": 18.0,
        "channel_model": "los_nlos",
        "path_loss_exp": 3.0,
        "backhaul_mbps": 20.0,
        "use_queues": True,
        "power_based_energy": True,
        "speed_cubed_energy": True,
        "drag_coeff_watt_per_m3s3": 0.008,
        "uav_battery_per_slot": 150.0,
        "road_network": True,
        "hotspot_motion": True,
        "hotspot_speed": 20.0,
        "hotspot_radius": 90.0,
        "hotspot_arrival_prob": 1.0,
        "base_arrival_prob": 0.05,
        "hotspot_size_boost": 1.3,
    },
}


# Keys that are always controlled by the CLI / runner and must not collide
# with the scenario dict when passed as **kwargs to make_config().
_CONTROLLED_KEYS = {"users", "leos", "horizon", "episodes", "seed",
                    "difficulty", "ablation", "output_root"}


def scenario_overrides(difficulty):
    """Return the authoritative override dict for a difficulty, if defined."""
    result = dict(MY_SCENARIOS.get(difficulty, {}))
    for key in _CONTROLLED_KEYS:
        result.pop(key, None)
    return result
