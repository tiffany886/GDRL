"""W3 statistical analysis for P1-lite (paired one-sided Wilcoxon etc.).

Reads per-(scene_seed) paired rows produced by run_sens_v2 (raw CSVs always
carry scene_seed) and, for the frozen-center B6 comparison, the v2 main raw
table (rows are in seed order 0..4 per method -> paired by index).

Report columns per (cell, method): success_rate | mean_delay_s (conditional)
| effective_total_latency = (done_delay_sum + apps_failed*deadline)/arrived,
where done_delay_sum = mean_delay_s * apps_done.

Paired tests (one-sided, Wilcoxon signed-rank, scipy default zero_method):
  H1a Ours_succ > hover_succ ; H1b Ours_succ > B4_succ
  H2   B6_delay  > Ours_delay (frozen main center rows)
Any pair with fewer than 5 usable differences is reported as n<5 (no p).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_raw(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def eff_latency(r: dict, deadline: float) -> float:
    done = int(float(r["apps_done"]))
    failed = int(float(r["apps_failed"]))
    arrived = max(int(float(r["apps_arrived"])), 1)
    delay_sum = num(r["mean_delay_s"]) * done
    return (delay_sum + failed * deadline) / arrived


def wilcox_gt(a: list, b: list, label: str, out):
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out.write(f"  {label:44s} n={n}  mean_diff={np.mean(d):+.4f}")
    if n < 5:
        out.write("  (n<5: 不报 p)\n")
        return None
    try:
        res = stats.wilcoxon(d, alternative="greater")
        out.write(f"  W={res.statistic:.1f} p={res.pvalue:.4f}\n")
        return res.pvalue
    except ValueError as e:  # all-zero diffs
        out.write(f"  (all-zero diff: {e})\n")
        return None


def wilcox_2s(a: list, b: list, label: str, out):
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    out.write(f"  {label:44s} n={n}  mean_diff={np.mean(d):+.4f}")
    if n < 5:
        out.write("  (n<5: 不报 p)\n")
        return None
    try:
        res = stats.wilcoxon(d, alternative="two-sided")
        out.write(f"  W={res.statistic:.1f} p={res.pvalue:.4f}\n")
        return res.pvalue
    except ValueError as e:
        out.write(f"  (all-zero diff: {e})\n")
        return None


def summarize(raw_path, deadline: float, tag: str, out):
    rows = load_raw(raw_path)
    # per (cell, scene_seed) method map
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], {}).setdefault(int(float(r["scene_seed"])), {})[
            r["method"]] = r
    out.write(f"\n=== {tag}: 成立区间 + 配对检验 ===\n")
    for cell in sorted(cells):
        seeds = sorted(cells[cell])
        ours = [cells[cell][s].get("Ours(T-patrol)") for s in seeds]
        hover = [cells[cell][s].get("B3-hover") for s in seeds]
        b4 = [cells[cell][s].get("B4-flat") for s in seeds]
        out.write(f"\n[cell {cell}] seeds={len(seeds)}\n")
        names = [("Ours", "Ours(T-patrol)"), ("hover", "B3-hover"),
                 ("chase", "B2-chase"), ("b6", "B6-noVWprune"),
                 ("B4-flat", "B4-flat")]
        for name, key in names:
            rows_sel = [cells[cell][s].get(key) for s in seeds]
            rows_sel = [r for r in rows_sel if r]
            if not rows_sel:
                continue
            succ = [num(r["success_rate"]) for r in rows_sel]
            delay = [num(r["mean_delay_s"]) for r in rows_sel]
            elat = [eff_latency(r, deadline) for r in rows_sel]
            out.write(
                f"  {name:8s} succ={np.mean(succ):.3f}±{np.std(succ):.3f} | "
                f"cond_delay={np.mean(delay):.3f}s | eff_lat={np.mean(elat):.3f}s\n")
        if ours and hover and all(ours) and all(hover):
            wilcox_gt([num(r["success_rate"]) for r in ours],
                      [num(r["success_rate"]) for r in hover],
                      "Ours-hover succ (gt 0)", out)
        if ours and b4 and all(ours) and all(b4):
            wilcox_gt([num(r["success_rate"]) for r in ours],
                      [num(r["success_rate"]) for r in b4],
                      "Ours-B4 succ (gt 0)", out)
        chase = [cells[cell][s].get("B2-chase") for s in seeds]
        if ours and chase and all(ours) and all(chase):
            out.write("  -- chase（红队对照，双侧） --\n")
            wilcox_2s([num(r["success_rate"]) for r in ours],
                      [num(r["success_rate"]) for r in chase],
                      "Ours-chase succ", out)
            wilcox_2s([num(r["mean_delay_s"]) for r in ours],
                      [num(r["mean_delay_s"]) for r in chase],
                      "Ours-chase cond_delay", out)
            wilcox_2s([num(r["energy_j"]) for r in ours],
                      [num(r["energy_j"]) for r in chase],
                      "Ours-chase energy", out)
            wilcox_2s([eff_latency(r, deadline) for r in ours],
                      [eff_latency(r, deadline) for r in chase],
                      "Ours-chase eff_lat", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", default="v2_p1_sens_grid_raw.csv")
    ap.add_argument("--sens10", default="v2_p1_sens_c10_raw.csv")
    ap.add_argument("--main", default="main_table_v2_raw.csv")
    ap.add_argument("--deadline", type=float, default=12.0)
    ap.add_argument("--out", default="v2_p1_stats_report.txt")
    a = ap.parse_args()
    out = (RESULTS / a.out).open("w", encoding="utf-8")
    out.write("# P1-lite 统计报告 (paired Wilcoxon, one-sided greater)\n")
    out.write(f"# deadline_s={a.deadline}  eff_lat=(done_delay_sum + failed*deadline)/arrived\n")

    sens = RESULTS / a.sens
    if sens.exists():
        summarize(sens, a.deadline, a.sens, out)

    sens10 = RESULTS / a.sens10
    if sens10.exists():
        summarize(sens10, a.deadline, a.sens10, out)

    det10 = RESULTS / "v2_p1_sens_det10_raw.csv"
    if det10.exists():
        rows = load_raw(det10)
        by = {}
        for r in rows:
            by.setdefault(r["method"], []).append(r)
        ours = sorted(by.get("Ours(T-patrol)", []),
                      key=lambda r: int(float(r["scene_seed"])))
        b6 = sorted(by.get("B6-noVWprune", []),
                    key=lambda r: int(float(r["scene_seed"])))
        if ours and b6:
            n = min(len(ours), len(b6))
            out.write("\n=== det-10 中心格 B6 vs Ours（同 scene_seed 配对） ===\n")
            wilcox_gt([num(b6[i]["mean_delay_s"]) for i in range(n)],
                      [num(ours[i]["mean_delay_s"]) for i in range(n)],
                      "B6-Ours delay (gt 0)", out)
            wilcox_2s([num(ours[i]["success_rate"]) for i in range(n)],
                      [num(b6[i]["success_rate"]) for i in range(n)],
                      "Ours-B6 succ", out)

    mainp = RESULTS / a.main
    if mainp.exists():
        rows = load_raw(mainp)
        by = {}
        for r in rows:
            by.setdefault(r["method"], []).append(r)
        ours = by["Ours(T-patrol)"]
        b6 = by["B6-noVWprune"]
        if ours and b6:
            n = min(len(ours), len(b6))
            out.write("\n=== 冻结中心格 B6 vs Ours (main_table_v2, 按行序配对) ===\n")
            wilcox_gt([num(b6[i]["mean_delay_s"]) for i in range(n)],
                      [num(ours[i]["mean_delay_s"]) for i in range(n)],
                      "B6-Ours delay (gt 0)", out)
            wilcox_gt([num(ours[i]["success_rate"]) for i in range(n)],
                      [num(b6[i]["success_rate"]) for i in range(n)],
                      "Ours-B6 succ (gt 0)", out)
    out.close()
    print(f"wrote {(RESULTS / a.out).resolve()}")


if __name__ == "__main__":
    main()
