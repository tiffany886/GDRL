"""P0 gate for the VA-DAG-HO revision plan (decision memo 2026-09-09).

Runs deterministic rows only (no RL training):
  * layout: aligned (current frozen), corners, interior hotspot placements
  * methods: Ours(T-patrol), B3-hover (uniform default stations),
             B7-hoverHot (hover stations placed on hotspot centres + cell
             centre: the fair "knowledge of hotspot sites" hover baseline)
  * scene seeds 0..9, frozen centre cell (R=800, lambda_high=0.2).

Purpose / gates (REVISION_PLAN_VA_DAG_HO.md P0-E1/P1-E4):
  1. Ours vs B7 on the aligned layout tells whether the patrol advantage is
     robust to a *stationary* baseline that knows where the hotspots are.
  2. Ours vs B3/B7 across interior/corners layouts tells whether the gap is
     an artifact of hotspot-at-extreme placement.

Outputs: results/p0_gate_raw.csv, results/p0_gate_summary.csv,
results/p0_gate_stats.txt (exact paired Wilcoxon, n=10 per cell).
"""
from __future__ import annotations

import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from ..baselines.trajectories import make_trajectory
from ..env.env import DisasterEnv
from .run_table import BASE_V2, RESULTS, _make_env, _stats

SEEDS = list(range(10))
LAYOUTS = {
    # layout -> flat hotspot centres [x0,y0,x1,y1,...]
    "aligned": [500.0, 150.0, 1500.0, 1850.0],
    "corners": [150.0, 150.0, 1850.0, 1850.0],
    "interior": [700.0, 700.0, 1300.0, 1300.0],
}
CENTER_OVERRIDES = {"uav_cover_radius_m": 800.0,
                    "lambda_high_per_s": 0.2,
                    "lambda_low_per_s": 0.03}


def _run_row(method: str, label: str, seed: int, layout: str):
    centers = LAYOUTS[layout]
    extra = dict(CENTER_OVERRIDES, hotspot_centers=list(centers))
    if method == "hover" and label.startswith("B7"):
        extra["uav_init_xy"] = _hoverhot_stations_impl(centers, 3, 2000.0)
    env = _make_env(None, seed, extra=extra, base=BASE_V2)
    policy = make_trajectory(method, seed=seed)
    env.reset(seed=seed)
    done = False
    while not done:
        _, _, done, _ = env.step(policy.act(env))
    meta = {"method": label, "mover": method, "layout": layout,
            "scene_seed": seed, "cell": "R800-l0.2",
            "hotspot_centers": str(list(centers))}
    return meta, _stats(env)


def _hoverhot_stations_impl(centers_flat, k: int, area: float):
    hs = np.asarray(centers_flat, dtype=float).reshape(-1, 2)
    out = [list(map(float, h)) for h in hs]
    while len(out) < k:
        out.append([float(area) / 2.0, float(area) / 2.0])
    return [v for pair in out[:k] for v in pair]


def eff_latency(st: dict, deadline: float = 12.0) -> float:
    done = int(st["apps_done"])
    failed = int(st["apps_failed"])
    arrived = max(int(st["apps_arrived"]), 1)
    return (float(st["mean_delay_s"]) * done + failed * deadline) / arrived


def wilcox_exact(diffs, alternative="greater"):
    """Exact one/two-sided signed-rank p (zeros dropped; ranks averaged).

    n <= 10 -> enumerate all 2^n sign assignments over averaged ranks.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return None
    order = np.argsort(np.abs(d))
    absd = np.abs(d)[order]
    # average ranks for ties
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_obs = float(np.sum(ranks[d[order] > 0]))
    total = 1 << n
    ge = le = 0
    for mask in range(total):
        w = 0.0
        for b in range(n):
            if (mask >> b) & 1:
                w += ranks[b]
        if w >= w_obs - 1e-9:
            ge += 1
        if w <= w_obs + 1e-9:
            le += 1
    if alternative == "greater":
        return ge / total
    if alternative == "less":
        return le / total
    return 2.0 * min(ge, le) / total


def _fmt_ms(v):
    return f"{float(np.mean(v)):.3f}\u00b1{float(np.std(v)):.3f}"


def main():
    t0 = time.perf_counter()
    rows = []
    for layout in LAYOUTS:
        for seed in SEEDS:
            for method, label in [("patrol", "Ours(T-patrol)"),
                                  ("hover", "B3-hover"),
                                  ("hover", "B7-hoverHot")]:
                meta, st = _run_row(method, label, seed, layout)
                st = {k: st[k] for k in (
                    "success_rate", "mean_delay_s", "energy_j",
                    "leo_invalid_attempts", "leo_placements",
                    "dep_violations", "apps_arrived", "apps_done",
                    "apps_failed", "frac_unassociated", "fail_uncovered")}
                row = dict(meta, **st)
                row["eff_latency_s"] = round(eff_latency(st), 6)
                rows.append(row)

    raw = RESULTS / "p0_gate_raw.csv"
    fields = ["method", "mover", "layout", "cell", "scene_seed",
              "hotspot_centers", "success_rate", "mean_delay_s",
              "eff_latency_s", "energy_j", "leo_invalid_attempts",
              "leo_placements", "dep_violations", "apps_arrived",
              "apps_done", "apps_failed", "frac_unassociated",
              "fail_uncovered"]
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary grouped by (layout, method)
    summ = {}
    for r in rows:
        summ.setdefault((r["layout"], r["method"]), []).append(r)
    with (RESULTS / "p0_gate_summary.csv").open("w", newline="",
                                                encoding="utf-8") as f:
        keys = ["success_rate", "mean_delay_s", "eff_latency_s", "energy_j",
                "leo_invalid_attempts", "apps_done", "apps_failed",
                "frac_unassociated", "fail_uncovered"]
        w = csv.writer(f)
        w.writerow(["layout", "method", "seeds"] + keys)
        for (layout, method) in sorted(summ):
            vals = {k: [r[k] for r in summ[(layout, method)]] for k in keys}
            w.writerow([layout, method, len(vals["success_rate"])] +
                       [_fmt_ms(vals[k]) for k in keys])

    # paired stats
    stats_lines = ["# P0 gate paired stats (exact Wilcoxon, one-sided unless "
                   "noted; n=10, centre cell R800-l0.2)",
                   "# layouts: aligned = frozen; corners/interior = geometry probe", ""]
    per_layout = {}
    for r in rows:
        per_layout.setdefault(r["layout"], {}).setdefault(
            r["method"], []).append(r)
    for layout in LAYOUTS:
        stats_lines.append(f"[{layout}]")
        ours = per_layout[layout]["Ours(T-patrol)"]
        for base_name in ("B3-hover", "B7-hoverHot"):
            base = per_layout[layout][base_name]
            o_s = [r["success_rate"] for r in ours]
            b_s = [r["success_rate"] for r in base]
            o_e = [r["eff_latency_s"] for r in ours]
            b_e = [r["eff_latency_s"] for r in base]
            o_j = [r["energy_j"] for r in ours]
            b_j = [r["energy_j"] for r in base]
            ds = [a - b for a, b in zip(o_s, b_s)]
            de = [b - a for a, b in zip(o_e, b_e)]  # >0 => Ours faster
            dj = [a - b for a, b in zip(o_j, b_j)]
            p_s = wilcox_exact(ds, "greater")
            p_e = wilcox_exact(de, "greater")
            p_j = wilcox_exact(dj, "two-sided")
            stats_lines.append(
                f"  Ours-{base_name}: succ d={np.mean(ds):+.4f} p={p_s:.4f} | "
                f"efflat(base-Ours) d={np.mean(de):+.3f}s p={p_e:.4f} | "
                f"energy d={np.mean(dj):+.1f}J p={p_j:.4f} (2-sided)")
        stats_lines.append("")
    (RESULTS / "p0_gate_stats.txt").write_text("\n".join(stats_lines),
                                               encoding="utf-8")
    print("\n".join(stats_lines))
    print(f"\nwrote {raw} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
