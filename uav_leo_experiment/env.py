import numpy as np
from dataclasses import asdict
from .config import UavLeoConfig
from .physics import flight_energy, nearest_leo_index, rate_uav_leo_vec, rate_user_uav_vec

class UavLeoEnv:
    """UAV-LEO offloading simulator with optional vehicular mobility and queues.
    The model is shared with every policy through :mod:`physics`. Two regimes:
    - legacy (defaults): static users, free-space channel, per-slot
      independent execution (no queues).
    - v2x (config presets ``v2x_*``): mobile vehicular users, LoS/NLoS channel,
      shared UAV/LEO compute queues with deadlines and drop penalty.
    """
    def __init__(self, config=None):
        self.config = config or UavLeoConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.t = 0
        self.user_pos = None
        self.user_vel = None
        self.uav_pos = None
        self.leo_pos = None
        self.task_bits = None
        self.cycles_per_bit = None
        self.uav_backlog = 0.0
        self.leo_backlog = 0.0
        self.uav_battery = 0.0
        self.road_dirs = None
        self.user_road = None
        self.user_along = None
        self.hotspot_pos = None
        self.hotspot_vel = None
        self.hotspot_road = None
        self.hotspot_along = None
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.t = 0
        self.uav_backlog = 0.0
        self.leo_backlog = 0.0
        self.uav_battery = c.uav_battery_per_slot * c.horizon
        if c.road_network:
            angles = self.rng.uniform(0.0, np.pi, size=c.n_roads)
            self.road_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            self.user_road = self.rng.integers(0, c.n_roads, size=c.users)
            self.user_along = self.rng.uniform(-c.road_half_len, c.road_half_len, size=c.users)
            center = np.array([0.5 * c.area_size, 0.5 * c.area_size])
            self.user_pos = center + self.user_along[:, None] * self.road_dirs[self.user_road]
            if c.mobility:
                speed = self.rng.uniform(0.0, c.user_speed_max, size=c.users)
                self.user_vel = self.road_dirs[self.user_road] * speed[:, None]
            else:
                self.user_vel = np.zeros((c.users, 2))
        else:
            self.user_pos = self.rng.uniform(0.0, c.area_size, size=(c.users, 2))
            if c.mobility:
                speed = self.rng.uniform(0.0, c.user_speed_max, size=(c.users, 1))
                angle = self.rng.uniform(0.0, 2.0 * np.pi, size=(c.users, 1))
                self.user_vel = np.concatenate([speed * np.cos(angle), speed * np.sin(angle)], axis=1)
            else:
                self.user_vel = np.zeros((c.users, 2))
        self.uav_pos = np.array([0.5 * c.area_size, 0.5 * c.area_size], dtype=float)
        xs = np.linspace(0.15 * c.area_size, 0.85 * c.area_size, c.leos)
        ys = np.full(c.leos, 0.5 * c.area_size)
        self.leo_pos = np.stack([xs, ys], axis=1)
        if c.hotspot_motion:
            self.hotspot_road = 0
            self.hotspot_along = float(self.rng.uniform(-0.5 * c.road_half_len, 0.5 * c.road_half_len))
            direction = c.hotspot_speed if self.rng.uniform() < 0.5 else -c.hotspot_speed
            self.hotspot_vel = self.road_dirs[self.hotspot_road] * direction
            self.hotspot_pos = center + self.hotspot_along * self.road_dirs[self.hotspot_road]
        else:
            self.hotspot_pos = None
            self.hotspot_vel = None
        self._sample_tasks()
        return self.observation()
    def _sample_tasks(self):
        c = self.config
        self.task_bits = self.rng.uniform(c.task_bits_min, c.task_bits_max, size=c.users)
        self.cycles_per_bit = self.rng.uniform(c.cycles_per_bit_min, c.cycles_per_bit_max, size=c.users)
        if c.hotspot_motion and self.hotspot_pos is not None:
            d2 = np.sum((self.user_pos - self.hotspot_pos) ** 2, axis=1)
            proximity = np.exp(-d2 / (2.0 * c.hotspot_radius ** 2))
            p_arr = c.base_arrival_prob + (c.hotspot_arrival_prob - c.base_arrival_prob) * proximity
            arrival = self.rng.uniform(size=c.users) < p_arr
            size_boost = 1.0 + c.hotspot_size_boost * proximity
            self.task_bits = self.task_bits * size_boost
            self.cycles_per_bit = self.cycles_per_bit * size_boost
        elif c.task_arrival_prob < 1.0:
            arrival = self.rng.uniform(size=c.users) < c.task_arrival_prob
        else:
            arrival = np.ones(c.users, dtype=bool)
        self.task_bits = self.task_bits * arrival
        self.cycles_per_bit = self.cycles_per_bit * arrival
    def _move_users(self):
        c = self.config
        if not c.mobility:
            return
        if c.road_network:
            direction = np.sum(self.user_vel * self.road_dirs[self.user_road], axis=1)
            self.user_along = self.user_along + np.sign(direction) * np.linalg.norm(self.user_vel, axis=1) * c.slot_seconds
            low = self.user_along < -c.road_half_len
            high = self.user_along > c.road_half_len
            self.user_along[low] = 2.0 * (-c.road_half_len) - self.user_along[low]
            self.user_along[high] = 2.0 * c.road_half_len - self.user_along[high]
            flip = low | high
            self.user_vel[flip] = -self.user_vel[flip]
            center = np.array([0.5 * c.area_size, 0.5 * c.area_size])
            self.user_pos = center + self.user_along[:, None] * self.road_dirs[self.user_road]
        else:
            self.user_pos = self.user_pos + self.user_vel * c.slot_seconds
            for axis in range(2):
                low = self.user_pos[:, axis] < 0.0
                high = self.user_pos[:, axis] > c.area_size
                self.user_pos[low, axis] = -self.user_pos[low, axis]
                self.user_pos[high, axis] = 2.0 * c.area_size - self.user_pos[high, axis]
                self.user_vel[low | high, axis] = -self.user_vel[low | high, axis]
            self.user_pos = np.clip(self.user_pos, 0.0, c.area_size)

    def _move_hotspot(self):
        c = self.config
        if not c.hotspot_motion or self.hotspot_pos is None:
            return
        self.hotspot_along = self.hotspot_along + c.hotspot_speed * c.slot_seconds
        if self.hotspot_along > c.road_half_len:
            self.hotspot_along = 2.0 * c.road_half_len - self.hotspot_along
            self.hotspot_vel = -self.hotspot_vel
        elif self.hotspot_along < -c.road_half_len:
            self.hotspot_along = 2.0 * (-c.road_half_len) - self.hotspot_along
            self.hotspot_vel = -self.hotspot_vel
        center = np.array([0.5 * c.area_size, 0.5 * c.area_size])
        self.hotspot_pos = center + self.hotspot_along * self.road_dirs[self.hotspot_road]

    def observation(self):
        c = self.config
        uav_backlog_norm = self.uav_backlog / max(c.uav_cpu_cycles_per_s * c.slot_seconds, 1.0)
        leo_backlog_norm = self.leo_backlog / max(c.leo_cpu_cycles_per_s * c.slot_seconds, 1.0)
        return {
            "t": self.t,
            "user_pos": self.user_pos.copy(),
            "user_vel": self.user_vel.copy(),
            "uav_pos": self.uav_pos.copy(),
            "leo_pos": self.leo_pos.copy(),
            "task_bits": self.task_bits.copy(),
            "cycles_per_bit": self.cycles_per_bit.copy(),
            "uav_backlog": float(self.uav_backlog),
            "leo_backlog": float(self.leo_backlog),
            "uav_battery": float(self.uav_battery),
            "uav_backlog_norm": float(uav_backlog_norm),
            "leo_backlog_norm": float(leo_backlog_norm),
            "uav_battery_norm": float(self.uav_battery / max(c.uav_battery_per_slot * c.horizon, 1.0)),
            "hotspot_pos": None if self.hotspot_pos is None else self.hotspot_pos.copy(),
            "hotspot_vel": None if self.hotspot_vel is None else self.hotspot_vel.copy(),
        }
    def step(self, action):
        c = self.config
        move = np.asarray(action.get("move", np.zeros(2)), dtype=float)
        if c.fixed_uav:
            move = np.zeros(2, dtype=float)
        if c.uav_battery_per_slot > 0.0 and self.uav_battery <= 0.0:
            move = np.zeros(2, dtype=float)
        norm = float(np.linalg.norm(move))
        if norm > c.uav_speed_max:
            move = move / max(norm, 1.0e-9) * c.uav_speed_max
            norm = c.uav_speed_max
        old_uav = self.uav_pos.copy()
        self.uav_pos = np.clip(self.uav_pos + move * c.slot_seconds, 0.0, c.area_size)
        moved = float(np.linalg.norm(self.uav_pos - old_uav))
        targets = np.asarray(action.get("targets", np.ones(c.users)), dtype=int)
        ratios = np.asarray(action.get("ratios", np.ones(c.users)), dtype=float)
        ratios = np.clip(ratios, 0.0, 1.0)
        if c.full_offload_only:
            ratios = (ratios > 1.0e-6).astype(float)
        if c.no_flight_energy:
            flight = 0.0
        else:
            flight = flight_energy(c, moved)
        if c.uav_battery_per_slot > 0.0:
            self.uav_battery = max(self.uav_battery - flight, 0.0)
            if self.uav_battery <= 0.0:
                flight = 0.0
        leo_idx = np.array([nearest_leo_index(self.leo_pos, self.user_pos[u]) for u in range(c.users)])
        leo_for_user = self.leo_pos[leo_idx]
        bits = np.asarray(self.task_bits, dtype=float)
        cpb = np.asarray(self.cycles_per_bit, dtype=float)
        cycles = bits * cpb
        ratio = np.clip(np.asarray(ratios, dtype=float), 0.0, 1.0)
        target = np.clip(np.asarray(targets, dtype=int), 0, 2)
        is_local = (target == 0) | (ratio <= 1.0e-6)
        is_uav = ~is_local & (target == 1)
        is_leo = ~is_local & (target == 2)
        local_latency = (1.0 - ratio) * cycles / c.user_cpu_cycles_per_s
        if c.power_based_energy:
            local_energy = c.user_device_power_watt * local_latency
        else:
            local_energy = c.compute_energy_coeff * ((1.0 - ratio) * cycles) * (c.user_cpu_cycles_per_s ** 2)
        full_local_latency = cycles / c.user_cpu_cycles_per_s
        if c.power_based_energy:
            full_local_energy = c.user_device_power_watt * full_local_latency
        else:
            full_local_energy = c.compute_energy_coeff * cycles * (c.user_cpu_cycles_per_s ** 2)
        rate_uu = rate_user_uav_vec(c, self.user_pos, self.uav_pos)
        rate_ul = np.minimum(rate_uu, rate_uav_leo_vec(c, self.uav_pos, leo_for_user))
        tx_latency = np.where(is_uav, ratio * bits / np.maximum(rate_uu, 1.0), 0.0)
        tx_latency += np.where(is_leo, ratio * bits / np.maximum(rate_ul, 1.0), 0.0)
        wait_latency = np.where(
            is_uav,
            np.maximum(self.uav_backlog - c.uav_cpu_cycles_per_s * c.slot_seconds, 0.0) / c.uav_cpu_cycles_per_s,
            0.0,
        )
        wait_latency += np.where(
            is_leo,
            np.maximum(self.leo_backlog - c.leo_cpu_cycles_per_s * c.slot_seconds, 0.0) / c.leo_cpu_cycles_per_s,
            0.0,
        )
        service_latency = np.where(is_uav, ratio * cycles / c.uav_cpu_cycles_per_s, 0.0)
        service_latency += np.where(is_leo, ratio * cycles / c.leo_cpu_cycles_per_s, 0.0)
        latency = np.where(is_local, full_local_latency, np.maximum(local_latency, tx_latency + wait_latency + service_latency))
        dropped_mask = latency > c.success_deadline_s
        remote_cycles = ratio * cycles
        tx_energy = c.user_tx_power_watt * tx_latency
        if c.power_based_energy:
            uav_energy = c.uav_power_watt * (remote_cycles / c.uav_cpu_cycles_per_s)
            leo_energy = c.leo_power_watt * (remote_cycles / c.leo_cpu_cycles_per_s)
            server_energy = np.where(is_uav, uav_energy, 0.0) + np.where(is_leo, leo_energy, 0.0)
            energy = np.where(is_local, full_local_energy, local_energy + tx_energy)
        else:
            uav_energy = c.compute_energy_coeff * remote_cycles * (c.uav_cpu_cycles_per_s ** 2)
            leo_energy = c.compute_energy_coeff * remote_cycles * (c.leo_cpu_cycles_per_s ** 2)
            server_energy = np.where(is_uav, uav_energy, 0.0) + np.where(is_leo, leo_energy, 0.0)
            energy = np.where(is_local, full_local_energy, local_energy + tx_energy + np.where(is_uav, uav_energy, leo_energy))
        server_energy = np.where(dropped_mask, 0.0, server_energy)
        energy = np.where(dropped_mask, np.where(is_local, full_local_energy, local_energy), energy)
        latency = np.where(dropped_mask & ~is_local, c.success_deadline_s, latency)
        added_uav_cycles = np.where(is_uav & ~dropped_mask, ratio * cycles, 0.0)
        added_leo_cycles = np.where(is_leo & ~dropped_mask, ratio * cycles, 0.0)
        total_latency = float(latency.sum())
        task_energy = float(energy.sum())
        server_energy_sum = float(server_energy.sum())
        dropped = int(dropped_mask.sum())
        successes = c.users - dropped
        added_uav = float(added_uav_cycles.sum())
        added_leo = float(added_leo_cycles.sum())
        target_names = np.where(is_local, "local", np.where(is_uav, "uav", "leo")).tolist()
        per_user_latency = latency.tolist()
        # drain server queues by the CPU processed this slot, then add new work
        self.uav_backlog = max(self.uav_backlog - c.uav_cpu_cycles_per_s * c.slot_seconds, 0.0) + added_uav
        self.leo_backlog = max(self.leo_backlog - c.leo_cpu_cycles_per_s * c.slot_seconds, 0.0) + added_leo
        total_energy = task_energy + flight
        reward = (
            -c.latency_weight * total_latency
            - c.energy_weight * total_energy
            - c.drop_penalty * dropped
        )
        self._move_users()
        self._move_hotspot()
        self.t += 1
        done = self.t >= c.horizon
        if not done:
            self._sample_tasks()
        info = {
            "latency": total_latency,
            "latency_mean": total_latency / c.users,
            "task_energy": task_energy,
            "server_energy": server_energy_sum,
            "flight_energy": flight,
            "total_energy": total_energy,
            "success_rate": successes / c.users,
            "drop_rate": dropped / c.users,
            "target_names": target_names,
            "offload_ratio_mean": float(np.mean(ratios)),
            "uav_pos_x": float(self.uav_pos[0]),
            "uav_pos_y": float(self.uav_pos[1]),
            "uav_backlog": float(self.uav_backlog),
            "leo_backlog": float(self.leo_backlog),
            "uav_battery": float(self.uav_battery),
            "per_user_latency": per_user_latency,
        }
        return self.observation(), float(reward), done, info
    def manifest(self):
        return asdict(self.config)
