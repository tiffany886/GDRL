"""Error estimation and threshold-triggered resync for the digital twin."""
import numpy as np


def estimate_error(obs_pred, obs_true, n_probe, rng, cfg=None,
                   task_mode="per_user"):
    """Max divergence over n_probe sampled users (+ hotspot if present).

    Combines position error (metres) with task-state drift mapped to an
    equivalent position error.  ``task_mode="per_user"`` maps a fully changed
    task to ~hotspot_radius of position error (sensitive to i.i.d. resampling);
    ``task_mode="load"`` maps the *relative aggregate load change* between the
    twin and the physical world (decision-relevant for bursty traffic: only
    load-regime transitions, e.g. burst start/end, exceed the threshold).
    Falls back to position-only when task fields are absent.
    """
    pp = np.asarray(obs_pred["user_pos"], dtype=float)
    pt = np.asarray(obs_true["user_pos"], dtype=float)
    n = pp.shape[0]
    k = min(int(n_probe), n)
    idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
    err = float(np.linalg.norm(pp[idx] - pt[idx], axis=-1).max())
    if obs_pred.get("hotspot_pos") is not None and obs_true.get("hotspot_pos") is not None:
        hp = np.atleast_2d(np.asarray(obs_pred["hotspot_pos"], dtype=float))
        ht = np.atleast_2d(np.asarray(obs_true["hotspot_pos"], dtype=float))
        err = max(err, float(np.linalg.norm(hp - ht, axis=-1).max()))
    if obs_pred.get("task_bits") is not None and obs_true.get("task_bits") is not None:
        tb_p = np.asarray(obs_pred["task_bits"], dtype=float)[idx]
        tb_t = np.asarray(obs_true["task_bits"], dtype=float)[idx]
        ref = float(getattr(cfg, "task_bits_max", 3.0e6))
        radius = float(getattr(cfg, "hotspot_radius", 100.0))
        if task_mode == "load":
            load_p = float(np.asarray(obs_pred["task_bits"], dtype=float).sum())
            load_t = float(np.asarray(obs_true["task_bits"], dtype=float).sum())
            denom = max(load_p, load_t, float(getattr(cfg, "task_bits_min", ref)))
            rel = abs(load_p - load_t) / max(denom, 1e-9)
            err = max(err, rel * radius)
        else:
            err = max(err, float(np.abs(tb_p - tb_t).max()) / max(ref, 1e-9) * radius)
    return err


def should_resync(error, delta):
    return error > delta


def local_ingress_error(received_bits, expected_bits, ref):
    """Locally observable ingress mismatch.

    ``received_bits`` are the task volumes the UAVs *actually received* this
    slot through the offload requests they served -- knowledge available at
    the UAV without any extra signalling.  ``expected_bits`` is the twin's own
    prediction for the users it decided to serve.  No global physical state is
    consulted, so this quantity is realisable in deployment.
    """
    rec = float(np.sum(received_bits))
    exp = float(np.sum(expected_bits))
    denom = max(rec, exp, float(ref))
    return abs(rec - exp) / max(denom, 1e-9)


def change_point_error(ewma, received_bits, ref):
    """Pure-local change-point statistic on the received offload volume."""
    rec = float(np.sum(received_bits))
    base = max(float(ewma), float(ref))
    return abs(rec - float(ewma)) / max(base, 1e-9), rec
