﻿# -*- coding: utf-8 -*-
"""Regenerate multi_uav_k3_table.tex / .csv with DRL baselines + significance (seed-aligned)."""
import csv, io, os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docs.GDRL.experiments.uav_leo_v2x_paper_final._load_data import by_seed, aligned, paired_p, stars, KEYS, BASE

acc = by_seed("k3")
ORDER = ["pmeo_m_eco", "mpc_m_h3", "pmeo_m", "current_exact_m", "follow_tea_m",
         "m_dqn", "m_td3", "m_sac", "random_m"]
LABELS = {"pmeo_m_eco": "PMEO-M-Eco (ours)", "mpc_m_h3": "MPC-M-H3",
          "pmeo_m": "PMEO-M", "current_exact_m": "Current-pos exact",
          "follow_tea_m": "Follow-TEA", "m_dqn": "M-DQN", "m_td3": "M-TD3",
          "m_sac": "M-SAC", "random_m": "Random"}

available = [m for m in ORDER if m in acc and acc[m]]
ours = acc["pmeo_m_eco"]
means = {}
for m in available:
    means[m] = {k: float(np.mean([d[k] for d in acc[m].values()])) for k in KEYS}

best = {}
for k in KEYS:
    better_higher = k in ("reward_mean", "success_rate")
    vals = [means[m][k] for m in available]
    best[k] = max(vals) if better_higher else min(vals)

csv_rows = []
for m in available:
    sigs = []
    for k in KEYS:
        av, bv = aligned(ours, acc[m])
        p = paired_p([d[k] for d in av], [d[k] for d in bv])
        sigs.append(stars(p))
    csv_rows.append({
        "method": LABELS[m],
        "reward_mean": round(means[m]["reward_mean"], 3),
        "success_rate": round(means[m]["success_rate"], 3),
        "latency_mean": round(means[m]["latency_mean"], 3),
        "total_energy_mean": round(means[m]["total_energy_mean"], 1),
        "sig": " / ".join(sigs),
    })
with io.open(f"{BASE}/multi_uav_k3_table.csv", "w", encoding="utf-8", newline="\n") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader(); w.writerows(csv_rows)

def fmt(m, k, nd=3):
    v = means[m][k]
    s = f"{v:.{nd}f}"
    if abs(v - best[k]) < 1e-9:
        s = "\\textbf{" + s + "}"
    return s

lines = [r"\begin{table}[t]", r"\centering",
         r"\caption{Performance comparison at k3 (3 UAVs, 51 users, 3 hotspots, 2 km; "
         r"10 seeds $\times$ 10 episodes). Best in bold. Significance vs PMEO-M-Eco by "
         r"paired t-test: $*p{<}0.05$, $**p{<}0.01$, $***p{<}0.001$; DRL baselines are "
         r"parameter-shared decentralized DQN/TD3/SAC trained for 30k env steps.}",
         r"\label{tab:k3_comparison}", r"\small",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Method & Reward & Success & Latency (s) & Energy (J/slot) \\", r"\midrule"]
for m in available:
    sig = []
    for k in KEYS:
        av, bv = aligned(ours, acc[m])
        sig.append(stars(paired_p([d[k] for d in av], [d[k] for d in bv])))
    lines.append(f"{LABELS[m]} & {fmt(m,'reward_mean')} & {fmt(m,'success_rate')} "
                 f"& {fmt(m,'latency_mean')} & {fmt(m,'total_energy_mean',1)} \\\\")
    lines.append(r"\footnotesize paired $p$: " + " / ".join(sig) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with io.open(f"{BASE}/multi_uav_k3_table.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("table written")
for r in csv_rows:
    print(f"{r['method']:18s} {r['reward_mean']:9.1f} {r['success_rate']:.3f} {r['latency_mean']:.3f} {r['total_energy_mean']:6.1f}  {r['sig']}")
