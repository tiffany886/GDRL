"""Sensitivity sweep over ``base_arrival_prob`` (task arrival outside hotspot).

Answers the reviewer/mentor question: "the 0.05 baseline arrival probability
is a modelling choice -- what happens if tasks are more uniformly spread?"

Runs, per arrival probability value and per seed, the same paired evaluation
used by the paper:

- PMEO (postmove_exact): post-move exact offloading (ours)
- Current-pos exact (current_exact): same trajectory, offloading solved at the
  pre-move position (decision-order ablation)
- MPC-H3 / MPC-H10 (mpc_traj_h3 / mpc_traj_h10): short-horizon receding-horizon
  trajectory planners with exact offloading (strongest training-free baselines)

Episodes are seeded identically across methods (config.seed + episode), so the
post-move gain and PMEO-vs-MPC deltas are paired per (seed, episode).

Outputs (default dir ``experiments/uav_leo_v2x_paper_final/arrival_sweep``):

- ``arrival_sweep_summary.csv``  pooled reward/success/latency/energy per prob x method
- ``arrival_sweep_paired.csv``   paired post-move gain and PMEO-vs-MPC deltas per prob
- ``fig_arrival_sweep.png``      2-panel figure (reward curves + post-move gain)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from .config import make_config
from .run_experiment import run_policy
from .baselines import ExpertExactOffloadPolicy
from .mpc_traj import MPCTrajPolicy
from .scenario_spec import MY_SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description="base_arrival_prob sensitivity sweep")
    parser.add_argument("--difficulty", default="v2x_hotspot_hard")
    parser.add_argument("--arrival-probs", default="0.05,0.15,0.3,0.5",
                        help="Comma-separated base_arrival_prob values")
    parser.add_argument("--seeds", default="73,74,75")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--leos", type=int, default=4)
    parser.add_argument("--mpc-horizons", default="3,10",
                        help="Comma-separated MPC lookahead horizons")
    parser.add_argument("--output-dir", default="experiments/uav_leo_v2x_paper_final/arrival_sweep")
    return parser.parse_args()


def build_policies(mpc_horizons):
    policies = [
        ExpertExactOffloadPolicy(offload_at="post_move"),   # PMEO
        ExpertExactOffloadPolicy(offload_at="current"),     # current-pos exact
    ]
    for h in mpc_horizons:
        policies.append(MPCTrajPolicy(horizon=h, offload="exact"))
    return policies


def main():
    args = parse_args()
    probs = [float(x) for x in args.arrival_probs.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    mpc_horizons = [int(x) for x in args.mpc_horizons.split(",")]
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    policies = build_policies(mpc_horizons)
    method_names = [p.name for p in policies]

    # rewards[prob][method] -> list of episode rewards pooled over seeds
    rewards = {p: {m: [] for m in method_names} for p in probs}
    extras = {p: {m: {} for m in method_names} for p in probs}

    for prob in probs:
        print(f"\n===== base_arrival_prob = {prob} =====", flush=True)
        for seed in seeds:
            scen = MY_SCENARIOS.get(args.difficulty, {})
            cfg = make_config(args.difficulty, "none", seed=seed,
                              users=scen.get("users", 12), leos=scen.get("leos", 4),
                              episodes=args.episodes, horizon=args.horizon,
                              base_arrival_prob=prob)
            for pol in policies:
                summary, episodes, _ = run_policy(pol, cfg)
                rewards[prob][pol.name] += [row["reward"] for row in episodes]
                extras[prob][pol.name] = {
                    "success_rate": summary["success_rate"],
                    "drop_rate": summary["drop_rate"],
                    "latency_mean": summary["latency_mean"],
                    "total_energy_mean": summary["total_energy_mean"],
                }
                print(f"  seed={seed} {pol.name}: reward={summary['reward_mean']:.2f} "
                      f"success={summary['success_rate']:.3f} drop={summary['drop_rate']:.3f}",
                      flush=True)

    # ---- summary table: pooled per prob x method ----
    summary_rows = []
    users = int(MY_SCENARIOS.get(args.difficulty, {}).get("users", 12))
    for prob in probs:
        for m in method_names:
            r = np.asarray(rewards[prob][m], dtype=float)
            summary_rows.append({
                "base_arrival_prob": prob,
                "method": m,
                "episodes": len(r),
                "reward_mean": float(r.mean()),
                "reward_std": float(r.std()),
                "per_user_slot_reward": float(r.mean() / (users * args.horizon)),
                "success_rate": extras[prob][m]["success_rate"],
                "drop_rate": extras[prob][m]["drop_rate"],
                "latency_mean": extras[prob][m]["latency_mean"],
                "total_energy_mean": extras[prob][m]["total_energy_mean"],
            })
    _write_csv(root / "arrival_sweep_summary.csv", summary_rows)

    # ---- paired stats ----
    pm = "postmove_exact"
    cur = "current_exact"
    paired_rows = []
    for prob in probs:
        r_pm = np.asarray(rewards[prob][pm], dtype=float)
        r_cur = np.asarray(rewards[prob][cur], dtype=float)
        d_order = r_pm - r_cur
        row = {
            "base_arrival_prob": prob,
            "postmove_reward": float(r_pm.mean()),
            "current_reward": float(r_cur.mean()),
            "order_gain_abs": float(d_order.mean()),
            "order_gain_per_user_slot": float(d_order.mean() / (users * args.horizon)),
            "order_gain_rel_pct": float(100.0 * d_order.mean() / max(abs(r_cur.mean()), 1e-9)),
            "order_wins": int((d_order > 0).sum()),
            "order_n": int(len(d_order)),
            "order_p_ttest": float(stats.ttest_rel(r_pm, r_cur).pvalue),
        }
        for h in mpc_horizons:
            mpc = f"mpc_traj_h{h}"
            r_mpc = np.asarray(rewards[prob][mpc], dtype=float)
            d_mpc = r_pm - r_mpc
            row[f"pmeo_vs_mpc_h{h}_delta"] = float(d_mpc.mean())
            row[f"pmeo_vs_mpc_h{h}_wins"] = int((d_mpc > 0).sum())
            row[f"pmeo_vs_mpc_h{h}_p_ttest"] = float(stats.ttest_rel(r_pm, r_mpc).pvalue)
        paired_rows.append(row)
        print(f"\n[prob={prob}] post-move gain = {d_order.mean():+.3f} "
              f"({int((d_order > 0).sum())}/{len(d_order)} wins, "
              f"p={stats.ttest_rel(r_pm, r_cur).pvalue:.2e})", flush=True)
    _write_csv(root / "arrival_sweep_paired.csv", paired_rows)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = {"postmove_exact": "#d62728", "current_exact": "#ff7f0e",
                  "mpc_traj_h3": "#2ca02c", "mpc_traj_h10": "#1f77b4"}
        labels = {"postmove_exact": "PMEO (post-move, ours)", "current_exact": "Current-pos exact",
                  "mpc_traj_h3": "MPC-H3", "mpc_traj_h10": "MPC-H10"}
        for m in method_names:
            ys = [float(np.mean(rewards[p][m])) for p in probs]
            err = [float(np.std(rewards[p][m])) for p in probs]
            ax1.errorbar(probs, ys, yerr=err, marker="o", capsize=3,
                         color=colors[m], label=labels[m])
        ax1.set_xlabel("base arrival probability (outside hotspot)")
        ax1.set_ylabel("episode reward (higher = better)")
        ax1.set_title("Reward vs task spread")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        d_abs = [float(np.mean(np.asarray(rewards[p][pm]) - np.asarray(rewards[p][cur]))) for p in probs]
        d_rel = [100.0 * d_abs[i] / max(abs(float(np.mean(rewards[p][cur]))), 1e-9)
                 for i, p in enumerate(probs)]
        ax2.plot(probs, d_abs, marker="o", color="#d62728", label="post-move gain (abs)")
        ax2.set_xlabel("base arrival probability (outside hotspot)")
        ax2.set_ylabel("post-move gain (reward units)")
        ax2.set_title("Decision-order gain vs task spread")
        ax2.grid(alpha=0.3)
        ax2r = ax2.twinx()
        ax2r.plot(probs, d_rel, marker="s", ls="--", color="#7f7f7f", label="post-move gain (%)")
        ax2r.set_ylabel("relative gain (%)")
        fig.tight_layout()
        fig.savefig(root / "fig_arrival_sweep.png", dpi=150)
        print(f"\nSaved figure: {root / 'fig_arrival_sweep.png'}", flush=True)
    except Exception as exc:  # plotting must never kill the results
        print(f"[warn] figure failed: {exc}", flush=True)

    print(f"\nSaved summary: {root / 'arrival_sweep_summary.csv'}", flush=True)
    print(f"Saved paired : {root / 'arrival_sweep_paired.csv'}", flush=True)


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
