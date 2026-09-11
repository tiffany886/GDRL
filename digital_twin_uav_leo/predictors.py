"""Digital-twin state extrapolation models.

``pred`` is a dict mirroring the physical obs fields that the twin predicts:
user_pos, user_vel, hotspot_pos (possibly None), hotspot_vel (possibly None).
task_bits / cycles_per_bit are intentionally frozen between syncs (spec 2.2).
"""
import numpy as np


def _extrapolate_hotspot(hotspot_pos, hotspot_vel, dt, cfg):
    """Linear extrapolation with road bounce, matching env._move_hotspot.

    Accepts 1D (single hotspot, M=1 obs convention) or 2D (M,2) arrays.
    """
    if hotspot_pos is None or hotspot_vel is None:
        return hotspot_pos, hotspot_vel
    pos = np.atleast_2d(np.asarray(hotspot_pos, dtype=float))
    vel = np.atleast_2d(np.asarray(hotspot_vel, dtype=float))
    center = np.array([0.5 * cfg.area_size, 0.5 * cfg.area_size])
    half = float(getattr(cfg, "road_half_len", 0.25 * cfg.area_size))
    out_pos = np.zeros_like(pos)
    out_vel = np.zeros_like(vel)
    for m in range(pos.shape[0]):
        v = vel[m]
        speed = float(np.linalg.norm(v))
        if speed <= 1e-9:
            out_pos[m] = pos[m]
            out_vel[m] = v
            continue
        direc = v / speed
        signed_speed = float(np.dot(v, direc))
        along = float(np.dot(pos[m] - center, direc))
        along = along + signed_speed * dt
        span = 4.0 * half
        x = (along + half) % span
        if x > 2.0 * half:
            x = span - x
        along = x - half
        out_pos[m] = center + direc * along
        out_vel[m] = direc * (signed_speed if not (x > 2.0 * half or x < -2.0 * half)
                              else -signed_speed)
    if np.asarray(hotspot_pos).ndim == 1:
        return out_pos[0], out_vel[0]
    return out_pos, out_vel


def _linear(pred, dt, cfg):
    pred["user_pos"] = pred["user_pos"] + pred["user_vel"] * dt
    hp, hv = _extrapolate_hotspot(pred.get("hotspot_pos"), pred.get("hotspot_vel"),
                                  dt, cfg)
    pred["hotspot_pos"] = hp
    pred["hotspot_vel"] = hv
    return pred


def _in_burst_at(cfg, t):
    if t is None or int(getattr(cfg, "burst_cycle", 0)) <= 0:
        return False
    if getattr(cfg, "burst_random", False):
        # 孪生不知道随机突发的相位：按非突发采样
        return False
    phase = t % int(cfg.burst_cycle)
    return int(cfg.burst_offset) <= phase < int(cfg.burst_offset) + int(cfg.burst_on)


def _resample_tasks(pred, cfg, rng, t=None):
    """Hotspot-aware task re-sampling, mirroring env._sample_tasks."""
    c = cfg
    up = np.asarray(pred["user_pos"], dtype=float)
    U = up.shape[0]
    burst = _in_burst_at(cfg, t)
    task_bits = rng.uniform(c.task_bits_min, c.task_bits_max, size=U)
    cpb = rng.uniform(c.cycles_per_bit_min, c.cycles_per_bit_max, size=U)
    if getattr(c, "hotspot_motion", False) and pred.get("hotspot_pos") is not None:
        hp = np.atleast_2d(np.asarray(pred["hotspot_pos"], dtype=float))
        d2 = np.sum((up[:, None, :] - hp[None, :, :]) ** 2, axis=-1)
        proximity = np.max(np.exp(-d2 / (2.0 * c.hotspot_radius ** 2)), axis=1)
        p_arr = c.base_arrival_prob + (c.hotspot_arrival_prob - c.base_arrival_prob) * proximity
        if burst:
            p_arr = np.minimum(p_arr * c.burst_arrival_mult, 1.0)
        arrival = rng.uniform(size=U) < p_arr
        size_boost = 1.0 + c.hotspot_size_boost * proximity
        if burst:
            size_boost = size_boost * c.burst_size_mult
        task_bits = task_bits * size_boost
        cpb = cpb * size_boost
    elif getattr(c, "task_arrival_prob", 1.0) < 1.0:
        p = c.task_arrival_prob * (c.burst_arrival_mult if burst else 1.0)
        arrival = rng.uniform(size=U) < min(p, 1.0)
    else:
        arrival = np.ones(U, dtype=bool)
    pred["task_bits"] = task_bits * arrival
    pred["cycles_per_bit"] = cpb * arrival
    return pred


def predict(name, pred, dt, cfg, rng=None, eps=0.0, t=None):
    """Extrapolate the twin state one slot forward; returns the same dict."""
    pred = dict(pred)
    if name == "freeze":
        return pred
    if name in ("linear", "linear_noise", "resample"):
        out = _linear(pred, dt, cfg)
        if name == "resample":
            return _resample_tasks(out, cfg, rng, t=t)
        if name == "linear_noise":
            scale = float(np.maximum(np.linalg.norm(pred["user_vel"], axis=-1).max(), 1.0))
            noise = rng.normal(0.0, eps * scale, size=pred["user_pos"].shape)
            out["user_pos"] = out["user_pos"] + noise
            if out["hotspot_pos"] is not None:
                hscale = float(np.linalg.norm(pred["hotspot_vel"], axis=-1).max()) if pred["hotspot_vel"] is not None else 0.0
                hnoise = rng.normal(0.0, eps * max(hscale, 1.0),
                                    size=np.asarray(out["hotspot_pos"]).shape)
                out["hotspot_pos"] = np.asarray(out["hotspot_pos"]) + hnoise
        return out
    raise ValueError(f"unknown predictor: {name}")
