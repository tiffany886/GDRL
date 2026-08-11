from pathlib import Path
import csv
import numpy as np
from scipy import stats

ROOT = Path("experiments/uav_leo_v2x_paper_final")

def read(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf8") as h:
        return list(csv.DictReader(h))

def fmt(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"

L = {"postmove_exact": "PMEO (ours)", "pmeo_e": "PMEO-E (energy-gated)",
     "current_exact": "Current-pos exact", "predict_tea": "Predict-TEA",
     "follow_tea": "Follow-TEA", "mpc_traj_h3": "MPC-H3", "random": "Random"}

lines = []
lines.append("# Route A analysis \u2014 sensitivity and multi-seed robustness")
lines.append("")

lines.append("## 1. Sensitivity (v2x_hotspot_hard, 40 episodes, seed 73)")
variants = [
    ("baseline", "hotspot-hard baseline"),
    ("scale_u8", "users = 8"),
    ("scale_u24", "users = 24"),
    ("deadline_050", "deadline = 0.50 s"),
    ("deadline_080", "deadline = 0.80 s"),
    ("hspeed_12_v2", "hotspot speed = 12 m/s"),
    ("hspeed_24_v2", "hotspot speed = 24 m/s"),
    ("no_hotspot", "uniform arrivals (no hotspot)"),
    ("energy_0_01_v2", "energy weight = 0.01"),
    ("energy_0_1_v2", "energy weight = 0.1"),
]
lines.append("| variant | method | reward | success | latency (s) | energy |")
lines.append("|---|---|---|---|---|---|")
for name, label in variants:
    if name == "baseline":
        base = {r["method"]: r for r in read(ROOT / "sweep_summary.csv")
                if r["difficulty"] == "v2x_hotspot_hard"}
    else:
        sub = next((ROOT / "sensitivity" / name).glob("*/none/*"), None)
        base = {r["method"]: r for r in read(sub / "uav_leo_summary.csv")} if sub else {}
    for m in ("postmove_exact", "pmeo_e", "current_exact", "predict_tea", "follow_tea", "mpc_traj_h3", "random"):
        if m not in base:
            continue
        r = base[m]
        lines.append(f"| {label} | {L.get(m, m)} | {fmt(r['reward_mean'])} | {fmt(r['success_rate'], 3)} | "
                     f"{fmt(r['latency_mean'], 4)} | {fmt(r['total_energy_mean'])} |")
    lines.append("")
lines.append("")

lines.append("## 2. Multi-seed robustness (40 episodes per seed \u00d7 5 seeds)")
lines.append("")
for difficulty in ("v2x_hotspot_hard", "v2x_hotspot_stress"):
    lines.append(f"### {difficulty}")
    lines.append("")
    lines.append("| method | reward (mean of seed means) | 95% CI | p (paired t, pooled over seeds) vs PMEO |")
    lines.append("|---|---|---|---|")
    seeds = [73, 101, 202, 303, 404]
    per_method = {}
    for s in seeds:
        if s == 73:
            ep_rows = [r for r in read(ROOT / "sweep_episodes.csv") if r["difficulty"] == difficulty]
        else:
            ep_rows = [r for r in read(ROOT / "multi_seed" / f"seed{s}" / "sweep_episodes.csv")
                       if r["difficulty"] == difficulty]
        for r in ep_rows:
            per_method.setdefault(r["method"], {}).setdefault(s, {})[int(r["episode"])] = float(r["reward"])
    for m in ("postmove_exact", "pmeo_e", "current_exact", "predict_tea", "follow_tea", "mpc_traj_h3", "random"):
        if m not in per_method:
            continue
        means = []
        for s in seeds:
            vals = per_method[m].get(s, {})
            if vals:
                means.append(float(np.mean(list(vals.values()))))
        arr = np.array(means)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else float("nan")
        ci = sem * stats.t.ppf(0.975, len(arr) - 1) if len(arr) > 1 else float("nan")
        pm = per_method.get("postmove_exact", {})
        pooled_g, pooled_m = [], []
        for s in seeds:
            g = pm.get(s, {})
            mv = per_method[m].get(s, {})
            common = sorted(set(g) & set(mv))
            pooled_g += [g[e] for e in common]
            pooled_m += [mv[e] for e in common]
        if len(pooled_g) > 5:
            pval = float(stats.ttest_rel(np.array(pooled_g), np.array(pooled_m)).pvalue)
        else:
            pval = float("nan")
        ci_str = f"\u00b1 {ci:.2f}" if ci == ci else "-"
        p_str = f"{pval:.4f}" if pval == pval else "-"
        lines.append(f"| {L.get(m, m)} | {np.mean(arr):.2f} | {ci_str} | {p_str} |")
    lines.append("")

out = ROOT / "routeA_analysis.md"
out.write_text("\n".join(lines) + "\n", encoding="utf8")
print("written", out)
