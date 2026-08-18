# -*- coding: utf-8 -*-
"""Shared k3 data loading aligned by seed for paired tests."""
import csv, io, collections

BASE = r"experiments/uav_leo_v2x_paper_final"
KEYS = ["reward_mean", "success_rate", "latency_mean", "total_energy_mean"]
SEED_ORDER = [1, 7, 42, 73, 314, 555, 888, 999, 12345, 2024]

def load(path):
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def by_seed(preset="k3"):
    """returns {method: {seed: {key: val}}} merged across eco/final/DRL/ablation files."""
    acc = collections.defaultdict(lambda: collections.defaultdict(dict))
    files = [
        f"{BASE}/multi_uav_eco/multi_uav_summary.csv",
        f"{BASE}/multi_uav_final/multi_uav_summary.csv",
        f"{BASE}/multi_uav_ablation2/multi_uav_summary.csv",
    ]
    for path in files:
        for r in load(path):
            if r.get("preset", "") != preset:
                continue
            m = r["method"]
            seed = int(float(r["seed"]))
            for k in KEYS:
                acc[m][seed][k] = float(r[k])
    for meth in ("m_dqn", "m_td3", "m_sac"):
        rows = load(f"{BASE}/mdrl_models/k3_{meth}_seed1/eval_summary.csv")
        for r in rows:
            seed = int(float(r["seed"]))
            for k in KEYS:
                acc[meth][seed][k] = float(r[k])
    return acc

def aligned(a, b):
    """paired value lists for methods a,b aligned by seed order."""
    av, bv = [], []
    for s in SEED_ORDER:
        if s in a and s in b:
            av.append(a[s]); bv.append(b[s])
    return av, bv

def paired_p(a, b):
    n = len(a)
    if n < 2:
        return 1.0
    d = [x - y for x, y in zip(a, b)]
    md = sum(d) / n
    sd = (sum((x - md) ** 2 for x in d) / (n - 1)) ** 0.5
    if sd == 0:
        return 1.0
    t = md / (sd / n ** 0.5)
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))

def stars(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
