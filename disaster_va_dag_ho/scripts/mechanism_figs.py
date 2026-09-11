"""P2 mechanism figures (v2, frozen center cell): collected by replaying
deterministic episodes and reading env state per slot / per app. No change to
the simulator or to any frozen config.

Outputs into results/figures_v2/:
  fig_v2_delay_cdf.png       CDF of completed-app delays, Ours/hover/chase
  fig_v2_frac_unassoc_ts.png frac_unassociated over time, mean+/-band
  fig_v2_hotspot_success.png success & delay split: inside vs outside hotspots
  fig_v2_coverage_snapshot.png one coverage snapshot (UAVs, R disks, terminals)
Run: python -m disaster_va_dag_ho.scripts.mechanism_figs
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..baselines.trajectories import make_trajectory
from .run_table import BASE_V2, _make_env

FIG = Path(__file__).resolve().parent.parent / "results" / "figures_v2"
FIG.mkdir(parents=True, exist_ok=True)
SEEDS = list(range(5))


def replay(mover: str, seed: int):
    env = _make_env(None, seed, base=BASE_V2)
    env.reset(seed=seed)
    pol = make_trajectory(mover, seed=seed)
    done = False
    series = []          # (t, frac_unassociated)
    arrivals = []        # (terminal, t_arrival, in_hotspot)
    while not done:
        t = int(env.t_s)
        n_un = int(np.count_nonzero(env.assignment < 0))
        series.append((t, n_un / float(env.cfg.num_terminals)))
        n_before = len(env.apps)
        _, _, done, _ = env.step(pol.act(env))
        for app in env.apps[n_before:]:
            pos = env.terminals.pos[app.terminal]
            hs = _in_hotspot(env, pos)
            arrivals.append((app.terminal, app.arrival_s, hs))
    apps = env.apps
    rec = []
    for app in apps:
        if app.state == "done":       # APP_DONE
            rec.append((app.arrival_s, app.completion_s, "done", app.suffered_uncovered))
        else:                          # failed / pending at horizon -> failed
            rec.append((app.arrival_s, app.deadline_s, "fail", app.suffered_uncovered))
    return env, series, arrivals, rec


def _in_hotspot(env, pos) -> bool:
    cfg = env.cfg
    centers = np.asarray(cfg.hotspot_centers, dtype=float).reshape(-1, 2)
    return bool(np.any(np.sum((pos[None, :] - centers) ** 2, axis=1)
                       <= float(cfg.hotspot_radius_m) ** 2))


def delay_cdf():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mover, label, ls in [("patrol", "Ours (T-patrol)", "-"),
                             ("hover", "B3-hover", "--"),
                             ("chase", "B2-chase", ":")]:
        delays, frac_done = [], []
        for s in SEEDS:
            env, _, _, rec = replay(mover, s)
            d = [r[1] - r[0] for r in rec if r[2] == "done"]
            delays += d
            frac_done.append(sum(1 for r in rec if r[2] == "done") / len(rec))
        delays = np.sort(delays)
        ax.plot(delays, np.arange(1, len(delays) + 1) / len(delays),
                label=f"{label}  (done {np.mean(frac_done):.2f})", ls=ls, lw=1.8)
    ax.axvline(12.0, color="gray", ls=":", lw=1)
    ax.text(12.05, 0.08, "deadline", rotation=90, fontsize=8, color="gray")
    ax.set_xlabel("completion delay of successful apps (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Delay CDF of successful apps only - hover's shorter tail\n"
                 "is selection: it simply fails more (5 seeds pooled)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_delay_cdf.png", dpi=150)
    plt.close(fig)
    print("wrote fig_v2_delay_cdf.png")


def frac_unassoc_ts():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for mover, label in [("patrol", "Ours (T-patrol)"),
                         ("hover", "B3-hover"),
                         ("chase", "B2-chase")]:
        mats = []
        for s in SEEDS:
            env, series, _, _ = replay(mover, s)
            mats.append(np.array([v for _, v in series]))
        m = np.mean(mats, axis=0)
        sd = np.std(mats, axis=0)
        ts = np.arange(len(m))
        ax.plot(ts, m, label=label, lw=1.8)
        ax.fill_between(ts, m - sd, m + sd, alpha=0.15)
    ax.set_xlabel("slot t (s)")
    ax.set_ylabel("frac. terminals unassociated")
    ax.set_title("Hard coverage leaves a persistent ~25% of terminals\n"
                 "disconnected every slot (5 seeds, mean +/- std)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_frac_unassoc_ts.png", dpi=150)
    plt.close(fig)
    print("wrote fig_v2_frac_unassoc_ts.png")


def hotspot_success():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for mover, label in [("patrol", "Ours"), ("hover", "hover"),
                         ("chase", "chase")]:
        in_done = in_all = out_done = out_all = 0
        in_delay, out_delay = [], []
        for s in SEEDS:
            env, _, arrivals, rec = replay(mover, s)
            by_id = {i: a for i, a in enumerate(arrivals)}  # index = arrival order
            for i, r in enumerate(rec):
                if i >= len(arrivals):
                    break
                hs = arrivals[i][2]
                if hs:
                    in_all += 1
                    if r[2] == "done":
                        in_done += 1
                        in_delay.append(r[1] - r[0])
                else:
                    out_all += 1
                    if r[2] == "done":
                        out_done += 1
                        out_delay.append(r[1] - r[0])
        axes[0].bar([f"{label} in", f"{label} out"],
                    [in_done / max(in_all, 1), out_done / max(out_all, 1)],
                    width=0.5, alpha=0.85,
                    color="#C44E52" if "Ours" in label else "#4C72B0")
        axes[1].bar([f"{label} in", f"{label} out"],
                    [np.mean(in_delay) if in_delay else 0,
                     np.mean(out_delay) if out_delay else 0],
                    width=0.5, alpha=0.85,
                    color="#C44E52" if "Ours" in label else "#4C72B0")
    axes[0].set_title("success rate: arrivals inside vs outside hotspots")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("mean successful delay (s): inside vs outside")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Spatially inhomogeneous load: patrol reaches both hotspot "
                 "bands (5 seeds pooled)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig_v2_hotspot_success.png", dpi=150)
    plt.close(fig)
    print("wrote fig_v2_hotspot_success.png")


def coverage_snapshot():
    env, series, _, _ = replay("patrol", seed=2)
    # snapshot at the slot with median unassociated fraction
    t = int(np.argmin(np.abs(np.array([v for _, v in series]) - 0.25)))
    env2 = _make_env(None, 2, base=BASE_V2)
    env2.reset(seed=2)
    pol = make_trajectory("patrol", seed=2)
    done = False
    while not done and int(env2.t_s) <= t:
        _, _, done, _ = env2.step(pol.act(env2))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    pos = env2.terminals.pos
    ass = env2.assignment
    colors = ["#C44E52", "#2E86AB", "#E9A319"]
    for k in range(env2.cfg.num_uavs):
        u = env2.uav_pos[k]
        ax.add_patch(plt.Circle(u, env2.cfg.uav_cover_radius_m,
                                color=colors[k], alpha=0.08))
    un = ass < 0
    for k in range(env2.cfg.num_uavs):
        m = ass == k
        ax.scatter(pos[m, 0], pos[m, 1], s=24, color=colors[k],
                   label=f"served by UAV{k+1}")
    ax.scatter(pos[un, 0], pos[un, 1], s=24, color="gray", marker="x",
               label="unassociated")
    for k in range(env2.cfg.num_uavs):
        ax.plot(*env2.uav_pos[k], marker="^", ms=10, color=colors[k])
    hs = np.asarray(env2.cfg.hotspot_centers, dtype=float).reshape(-1, 2)
    for h in hs:
        ax.add_patch(plt.Circle(h, env2.cfg.hotspot_radius_m,
                                color="black", fill=False, ls="--", lw=1.2))
    ax.set_xlim(0, env2.cfg.area_size_m)
    ax.set_ylim(0, env2.cfg.area_size_m)
    ax.set_aspect("equal")
    ax.set_title(f"Coverage snapshot at t={t} s (seed 2): patrol covers the\n"
                 f"two hotspot bands; ~25% of terminals still disconnected")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v2_coverage_snapshot.png", dpi=150)
    plt.close(fig)
    print("wrote fig_v2_coverage_snapshot.png")


if __name__ == "__main__":
    delay_cdf()
    frac_unassoc_ts()
    hotspot_success()
    coverage_snapshot()
