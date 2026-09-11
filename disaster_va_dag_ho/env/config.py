"""Scenario/energy/channel configuration for the disaster VA-DAG-HO env.

Field names marked "physics reuse" are intentionally identical to the
attribute names read by ``uav_leo_experiment/physics.py`` so the LoS/NLoS +
Shannon rate helpers and the flight-energy model can be reused as-is
(duck typing on the config object).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class VAConfig:
    # --- identity ---
    run_name: str = "main"
    seed: int = 0

    # --- region & entities (spec §2.2 main table) ---
    area_size_m: float = 2000.0
    num_terminals: int = 30
    num_uavs: int = 3
    num_leos: int = 2
    uav_altitude: float = 100.0
    leo_altitude: float = 550_000.0
    slot_seconds: float = 1.0
    horizon: int = 50

    # --- task arrival (spec §3.1) ---
    burst_start_s: float = 15.0
    burst_end_s: float = 35.0
    lambda_low_per_s: float = 0.02
    lambda_high_per_s: float = 0.1
    terminal_concurrency_max: int = 2
    app_deadline_s: float = 12.0
    deadline_margin_s: float = 5.0        # 紧急度排序：剩余时间低于该值视为“截止期紧迫”
    template_weights: List[float] = field(default_factory=lambda: [0.45, 0.30, 0.25])
    dag_cycles_scale: float = 1.0         # 标定：任务计算量乘子（实验期调参）
    dag_bits_scale: float = 1.0           # 标定：任务传输量乘子

    # --- terminal mobility (half static / slow) ---
    user_speed_max_mps: float = 2.0
    static_fraction: float = 0.5

    # --- UAV motion & energy (spec §3.2, no charging) ---
    uav_speed_max_mps: float = 20.0
    uav_min_sep_m: float = 30.0
    soc_init_j: float = 50_000.0
    soc_low_fraction: float = 0.2
    hover_power_watt: float = 80.0
    drag_coeff_watt_per_m3s3: float = 0.004
    speed_cubed_energy: bool = True

    # --- channel (physics reuse keys; spec §4.1) ---
    channel_model: str = "los_nlos"
    bandwidth_hz: float = 2.0e6
    noise_watt: float = 2.0e-13
    carrier_hz: float = 2.4e9
    los_param_a: float = 9.61
    los_param_b: float = 0.16
    path_loss_los: float = 1.6
    path_loss_nlos: float = 23.0
    path_loss_exp: float = 2.0
    user_tx_power_watt: float = 0.2
    uav_tx_power_watt: float = 0.5
    user_uav_gain_db: float = 6.0
    uav_leo_gain_db: float = 50.0
    leo_capacity_bps: float = 10.0e6
    backhaul_mbps: Optional[float] = None

    # --- v2: hard UAV coverage & spatial hotspot arrivals (spec v2 §1.2/§1.4) ---
    uav_cover_radius_m: Optional[float] = None   # None or <=0 = soft-connect (v1 / w/o hard-cover)
    uav_init_xy: Optional[List[float]] = None  # flat [x0,y0,...] for K UAVs; None = default spread
    hotspot_arrivals: bool = False               # False = v1 uniform global lambda
    num_hotspots: int = 2
    hotspot_radius_m: float = 300.0              # inside: lambda_high, outside: lambda_low
    hotspot_centers: Optional[List[float]] = None  # flat [x0,y0,x1,y1,...]; None = default spots
    burst_multiplier: float = 1.0                # global multiplier over [burst_start, burst_end)
    trajectory: str = "patrol"                   # v2 fixed trajectory id (spec §1.5)

    # --- simplified ephemeris: deterministic staggered windows (seconds) ---
    leo_window_period_s: List[float] = field(default_factory=lambda: [22.0, 22.0])
    leo_window_duty: List[float] = field(default_factory=lambda: [0.4, 0.4])
    leo_window_phase_s: List[float] = field(default_factory=lambda: [0.0, 11.0])

    # --- compute capacity / power placeholders (used from W2 ScheduleDAG) ---
    uav_cpu_cycles_per_s: float = 8.0e9
    leo_cpu_cycles_per_s: float = 60.0e9
    uav_power_watt: float = 150.0
    comm_power_watt: float = 20.0

    # --- reward weights (spec §5; fixed for the main table) ---
    alpha_latency: float = 0.02
    alpha_energy: float = 0.01
    alpha_fail: float = 8.0
    low_battery_penalty: float = 4.0
    boundary_penalty: float = 1.0
    collision_penalty: float = 2.0

    # --- observation (tuned in W3) ---
    obs_fail_window: int = 10
    obs_terminal_features: bool = False  # B4 flat: 追加每终端就绪/截止期特征块
    obs_energy: bool = True              # 消融 w/o energy-in-state

    # --- scheduler variants / ablations (W4) ---
    vw_prune: bool = True    # B6: False = 常在线假设（忽略可见窗剪枝）
    dag_order: bool = True   # 消融 w/o DAG-order：False = 依赖任务当独立包

    @property
    def area_km(self) -> float:
        return self.area_size_m / 1000.0


def _coerce(cfg: VAConfig, key: str, value):
    """Best-effort cast of a YAML scalar into the dataclass field type."""
    if value is None:
        return None
    ftype = type(cfg.__dataclass_fields__[key].default)
    if ftype is bool and not isinstance(value, bool):
        if str(value).strip().lower() in ("true", "1", "yes"):
            return True
        if str(value).strip().lower() in ("false", "0", "no"):
            return False
        return bool(value)
    if ftype is float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Config key {key}: cannot parse {value!r} as float")
        return float(value)
    if ftype is int and not isinstance(value, bool):
        return int(value)
    if isinstance(value, list) and ftype in (int, float):
        return [ftype(v) if not isinstance(v, str) else float(v) for v in value]
    return value


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> VAConfig:
    """Build a VAConfig from an optional YAML plus keyword overrides.

    ``path`` defaults to ``disaster_va_dag_ho/configs/main.yaml``.
    """
    cfg = VAConfig()
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = set(cfg.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = dataclasses.replace(
            cfg, **{k: _coerce(cfg, k, v) for k, v in raw.items() if v is not None}
        )
    if overrides:
        known = set(cfg.__dataclass_fields__)
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown config override keys: {sorted(unknown)}")
        cfg = dataclasses.replace(
            cfg, **{k: _coerce(cfg, k, v) for k, v in overrides.items() if v is not None}
        )
    return cfg


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "main.yaml"


def _resolve_yaml(path) -> Path:
    """Resolve a YAML path; bare names are looked up in ``configs/`` first."""
    p = Path(path)
    if not p.is_absolute() and not p.parent.parts:
        cand = DEFAULT_CONFIG_PATH.parent / p
        if cand.exists():
            return cand
    return p


def load_config_variant(delta_path: Optional[str | Path] = None,
                        overrides: Optional[dict] = None,
                        base_path: Optional[str | Path] = None) -> VAConfig:
    """Load a base YAML (default: main.yaml) then overlay delta + overrides."""
    base = base_path if base_path is not None else DEFAULT_CONFIG_PATH
    cfg = load_config(base)
    if delta_path is not None:
        raw = yaml.safe_load(_resolve_yaml(delta_path).read_text(encoding="utf-8")) or {}
        known = set(cfg.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {delta_path}: {sorted(unknown)}")
        cfg = dataclasses.replace(
            cfg, **{k: _coerce(cfg, k, v) for k, v in raw.items() if v is not None}
        )
    if overrides:
        known = set(cfg.__dataclass_fields__)
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown config override keys: {sorted(unknown)}")
        cfg = dataclasses.replace(
            cfg, **{k: _coerce(cfg, k, v) for k, v in overrides.items() if v is not None}
        )
    return cfg
