﻿# -*- coding: utf-8 -*-
"""fig8: multi-UAV trajectories for PMEO-M-Eco / MPC-M-H3 / PMEO-M on the same episode."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docs.GDRL.uav_leo_experiment.run_multi_uav import build_config
from docs.GDRL.uav_leo_experiment.env import UavLeoEnv
from docs.GDRL.uav_leo_experiment.multi_uav import (PmeoMEcoPolicy, MpcMPolicy, PmeoMPolicy,
                                          uav_pos_array)

BASE = r"experiments/uav_leo_v2x_paper_final"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

cfg = build_config("k3", 1, episodes=1)
policies = {
    "PMEO-M-Eco (ours)": PmeoMEcoPolicy(),
    "MPC-M-H3": MpcMPolicy(),
    "PMEO-M": PmeoMPolicy(),
}

def record(pol):
    env = UavLeoEnv(cfg)
    obs = env.reset(seed=cfg.seed + 1)
    rng = np.random.default_rng(1)
    uav_traj, hot_traj, user_at = [], [], {}
    done = False
    t = 0
    while not done:
        uav_traj.append(uav_pos_array(obs, cfg).copy())
        hp = obs.get("hotspot_pos")
        if hp is not None:
            hp = np.asarray(hp)
            hot_traj.append(hp if hp.ndim == 2 else hp[None, :])
        else:
            hot_traj.append(None)
        if t in (0, 10, 20, 29):
            user_at[t] = np.asarray(obs["user_pos"], dtype=float).copy()
        action = pol.act(obs, rng, cfg)
        obs, reward, done, _ = env.step(action)
        t += 1
    return uav_traj, hot_traj, user_at

data = {name: record(pol) for name, pol in policies.items()}
area = cfg.area_size
C_UAV = ["#1f77b4", "#ff7f0e", "#2ca02c"]
C_HOT = "#d62728"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "axes.titlesize": 10, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
for ax, (name, (traj, hot, users)) in zip(axes, data.items()):
    for k in range(cfg.uavs):
        pts = np.array([tr[k] for tr in traj])
        ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, lw=1.4, color=C_UAV[k],
                label=f"UAV {k+1}")
        ax.plot(pts[0, 0], pts[0, 1], "s", color=C_UAV[k], ms=6)
    if hot[0] is not None:
        hp = np.array([h.mean(axis=0) for h in hot if h is not None])
        ax.plot(hp[:, 0], hp[:, 1], "--", lw=1.0, color=C_HOT, alpha=0.7,
                label="Hotspots")
    # users at t=0 and t=29
    u0 = users.get(0)
    u29 = users.get(29)
    if u0 is not None:
        ax.scatter(u0[:, 0], u0[:, 1], s=7, c="#555555", alpha=0.45,
                   marker=".", label="Users (t=0)")
    if u29 is not None:
        ax.scatter(u29[:, 0], u29[:, 1], s=7, c="#ffcc00", alpha=0.6,
                   marker=".", label="Users (t=29)")
    ax.set_xlim(0, area); ax.set_ylim(0, area)
    ax.set_aspect("equal")
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=6.6, loc="best")
    ax.grid(color="#e3e3e3", lw=0.5)
fig.suptitle("k3: UAV trajectories over one episode (seed 1) - hotspots move at 18 m/s",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG, "fig8_trajectories.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print("saved", out)

# also print flight distance stats
print("flight distance per UAV (m):")
for name, (traj, hot, users) in data.items():
    dists = []
    for k in range(cfg.uavs):
        pts = np.array([tr[k] for tr in traj])
        d = float(np.abs(np.diff(pts, axis=0)).sum())
        dists.append(d)
    print(f"  {name:18s} {['%.0f' % d for d in dists]}")
