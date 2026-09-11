"""Diagnostic: where does adaptive sync fire relative to burst windows?

Runs one burst episode and records (a) per-slot task activity in the physical
world and (b) the slots at which the twin resyncs.  Saves
results/fig_sync_timing_burst.png and prints a slot-by-slot table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from uav_leo_experiment.env import UavLeoEnv
from uav_leo_experiment.baselines import ExpertExactOffloadPolicy
from digital_twin_uav_leo.run_experiments import make_single_cfg

RESULTS = Path(__file__).resolve().parent / "results"


def run_trace(seed=73, delta=40.0, probe_m=1):
    cfg = make_single_cfg("hard", seed, 1, burst=True)
    env = UavLeoEnv(cfg)
    policy = ExpertExactOffloadPolicy(offload_at="post_move")
    rng = np.random.default_rng(0)
    from digital_twin_uav_leo.dt_env import TwinnedWorld

    obs = env.reset(seed=seed)
    tw = TwinnedWorld(env, tau=9999, predictor="linear", rng=rng,
                      sync_mode="adaptive", probe_m=probe_m, probe_n=3,
                      delta=delta, task_mode="load")
    tw.sync(obs)
    activity = [int((obs["task_bits"] > 0).sum())]
    sync_slots = [0]
    done = False
    while not done:
        obs_dt = tw.get_obs()
        action = policy.act(obs_dt, rng, cfg)
        obs, _, done, _ = env.step(action)
        activity.append(int((obs["task_bits"] > 0).sum()))
        tw.step_forward(dt=float(cfg.slot_seconds))
        before = tw.sync_count
        tw._maybe_probe_and_resync(obs)
        if tw.sync_count > before:
            sync_slots.append(int(env.t))
    return cfg, activity, sync_slots


def main():
    cfg, activity, sync_slots = run_trace()
    burst_slots = [t for t in range(cfg.horizon) if 4 <= (t % 15) < 7]
    print(f"burst slots: {burst_slots}")
    print(f"adaptive sync slots (delta=40m, load-aware): {sync_slots}")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(activity))
    ax.bar(x, activity, color="#BBDEFB", label="active tasks (physical)")
    for s in burst_slots:
        ax.axvspan(s - 0.5, s + 0.5, color="#FFCDD2", alpha=0.5, lw=0)
    ax.plot(sync_slots, [max(activity) * 1.05] * len(sync_slots), "v",
            color="#C62828", label="adaptive sync", markersize=8)
    ax.set_xlabel("time slot")
    ax.set_ylabel("# active tasks")
    ax.set_title("bursty traffic: load-aware adaptive sync fires at burst boundaries")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_sync_timing_burst.png", bbox_inches="tight")
    print("saved results/fig_sync_timing_burst.png")


if __name__ == "__main__":
    main()
