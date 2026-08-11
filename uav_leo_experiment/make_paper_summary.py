"""Assemble the final paper-ready results summary for the UAV-LEO hotspot line.

Reads the 40-episode main sweep (sweep_summary.csv / sweep_episodes.csv) and
the GDRL contribution ablations (run_experiment outputs under ablation/) and
writes a single ``paper_results.md`` with:

- scenario parameter table,
- main comparison table per difficulty (reward / success / latency / energy /
  offload ratio / drop),
- paired significance (diff, wins, paired t-test and Wilcoxon p) vs GDRL,
- GDRL contribution ablation table (full vs no-trajectory vs no-exact-offload),
- reproduction commands and artifact paths.

Usage:
    python -m uav_leo_experiment.make_paper_summary [--sweep_dir DIR] [--output FILE]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from .scenario_spec import MY_SCENARIOS

SHORT = {
    "gdrl": "GDRL (ours)", "predict_tea": "Predict-TEA", "follow_tea": "Follow-TEA",
    "deadline_tea": "Deadline-TEA", "tea_partial": "TEA", "energy_guarded_tea": "E-Guard TEA",
    "full_offload_tea": "Full-offload", "greedy_partial": "Greedy", "lyapunov": "Lyapunov",
    "rate_aware": "Rate-aware", "uav_only": "UAV-only", "leo_only": "LEO-only",
    "local_only": "Local-only", "random": "Random", "ppo": "PPO (BC+KL)",
    "dqn": "DQN", "d3qn": "P-D3QN (2024)", "gat_ppo": "GAT-PPO (2024)",
    "transformer_ppo": "Transformer-PPO (2025)", "td3": "TD3", "sac": "SAC",
    "genetic": "GA", "pso": "PSO", "sa": "SA", "aco": "ACO",
    "exhaustive_optimal": "Exhaustive",
    "gdrl_no_traj": "GDRL - traj (expert move)", "gdrl_no_opt": "GDRL - exact offload",
    "mpc_traj_h3": "MPC-H3 (receding horizon)", "mpc_traj_h5": "MPC-H5 (receding horizon)",
    "mpc_traj_h10": "MPC-H10 (receding horizon)", "mpc_traj_h30": "MPC-H30 (receding horizon)",
    "qa_exact": "Queue-aware exact", "postmove_exact": "PMEO (ours)",
    "current_exact": "Current-pos exact", "pmeo_e": "PMEO-E (energy-gated)",
    "gdrl_expert": "GDRL - both layers removed",
}
GROUP = {
    "proposed": ["gdrl"],
    "drl": ["ppo", "dqn", "d3qn", "gat_ppo", "transformer_ppo", "td3", "sac"],
    "heuristic": ["predict_tea", "follow_tea", "deadline_tea", "tea_partial",
                  "energy_guarded_tea", "full_offload_tea", "greedy_partial", "lyapunov",
                  "rate_aware", "uav_only", "leo_only"],
    "meta": ["genetic", "pso", "sa", "aco", "exhaustive_optimal"],
    "trivial": ["random", "local_only"],
}
def _fmt_ghz(v):
    return f"{v / 1.0e9:.1f} GHz" if v < 1.0e12 else f"{v / 1.0e12:.1f} THz"


def _scenario_params(difficulty):
    """Build the paper scenario table from the authoritative scenario spec."""
    sc = dict(MY_SCENARIOS.get(difficulty, {}))
    if not sc:
        return {}
    return {
        "users": sc.get("users", "-"),
        "horizon": sc.get("horizon", "-"),
        "bandwidth": f"{sc.get('bandwidth_hz', 0) / 1.0e6:.1f} MHz",
        "deadline": f"{sc.get('success_deadline_s', 0):.2f} s",
        "user_cpu": _fmt_ghz(sc.get("user_cpu_cycles_per_s", 0)),
        "uav_cpu": _fmt_ghz(sc.get("uav_cpu_cycles_per_s", 0)),
        "leo_cpu": _fmt_ghz(sc.get("leo_cpu_cycles_per_s", 0)),
        "tasks": (f"{sc.get('task_bits_min', 0) / 1.0e6:.1f}-{sc.get('task_bits_max', 0) / 1.0e6:.1f} Mb, "
                  f"{sc.get('cycles_per_bit_min', 0):.0f}-{sc.get('cycles_per_bit_max', 0):.0f} cyc/bit"),
        "backhaul": f"{sc.get('backhaul_mbps', 0):.0f} Mbps",
        "path_loss": sc.get("path_loss_exp", "-"),
        "hotspot": (f"radius {sc.get('hotspot_radius', 0):.0f} m, "
                    f"{sc.get('hotspot_speed', 0):.0f} m/s"),
        "drop_penalty": f"{sc.get('drop_penalty', 0):.1f}",
        "energy_weight": f"{sc.get('energy_weight', 0):.3f}",
    }


def read_csv(path):
    with path.open(encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def fmt(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def paired_stats(g, m):
    common = sorted(set(g) & set(m))
    if len(common) < 5:
        return None
    gv = np.array([g[e] for e in common])
    mv = np.array([m[e] for e in common])
    d = gv - mv
    t_p = float(stats.ttest_rel(gv, mv).pvalue)
    try:
        w_p = float(stats.wilcoxon(gv, mv).pvalue)
    except ValueError:
        w_p = float("nan")
    return {"n": len(common), "diff": float(d.mean()), "wins": int((d > 0).sum()),
            "t_p": t_p, "w_p": w_p}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default="experiments/uav_leo_v2x_paper_final")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.sweep_dir)
    output = Path(args.output) if args.output else root / "paper_results.md"

    summary = read_csv(root / "sweep_summary.csv")
    episodes = read_csv(root / "sweep_episodes.csv")
    difficulties = sorted({r["difficulty"] for r in summary})

    lines = []
    lines.append("# UAV-LEO Task Offloading & Trajectory Optimization \u2014 Paper Results")
    lines.append("")
    lines.append("> Generated automatically from the 40-episode main sweep and the GDRL "
                 "contribution ablations. Paired statistics use per-episode rewards of the "
                 "same seeded episodes (seed 73 + episode index).")
    lines.append("")

    # scenario table
    lines.append("## 1. Evaluation scenarios")
    lines.append("")
    lines.append("| parameter | hotspot-hard | hotspot-stress |")
    lines.append("|---|---|---|")
    for key, label in [("users", "users"), ("horizon", "horizon (slots)"),
                       ("bandwidth", "user-UAV bandwidth"), ("deadline", "task deadline"),
                       ("user_cpu", "user CPU"), ("uav_cpu", "UAV edge CPU"),
                       ("leo_cpu", "LEO edge CPU"), ("tasks", "task size"),
                       ("backhaul", "UAV-LEO backhaul"), ("path_loss", "path-loss exponent"),
                       ("hotspot", "hotspot motion"), ("drop_penalty", "drop penalty"),
                       ("energy_weight", "energy weight")]:
        row = "| " + label + " | " + " | ".join(
            str(_scenario_params(d).get(key, "-")) for d in difficulties) + " |"
        lines.append(row)
    lines.append("")

    # main table per difficulty
    for difficulty in difficulties:
        rows = [r for r in summary if r["difficulty"] == difficulty]
        rows = sorted(rows, key=lambda r: float(r["reward_mean"]), reverse=True)
        ep = {}
        for r in episodes:
            if r["difficulty"] == difficulty:
                ep.setdefault(r["method"], {})[int(r["episode"])] = float(r["reward"])
        ref_method = "postmove_exact" if "postmove_exact" in ep else "gdrl"
        g = ep.get(ref_method, {})
        ref_label = "PMEO" if ref_method == "postmove_exact" else "GDRL"
        lines.append(f"## 2. Main results \u2014 {difficulty}")
        lines.append("")
        lines.append(f"| rank | method | reward | success | latency (s) | energy | offload | drop | \u0394 vs {ref_label} | wins | p (t / Wilcoxon) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for rank, r in enumerate(rows, 1):
            m = r["method"]
            cell = f"| {rank} | {SHORT.get(m, m)} | {fmt(r['reward_mean'])} | {fmt(r['success_rate'], 3)} | "
            cell += f"{fmt(r['latency_mean'], 4)} | {fmt(r['total_energy_mean'])} | {fmt(r['offload_ratio_mean'], 3)} | {fmt(r['drop_rate'], 3)} | "
            if m == ref_method:
                cell += " \u2014 | \u2014 | \u2014 |"
            else:
                st = paired_stats(g, ep.get(m, {}))
                if st:
                    cell += f"{st['diff']:+.1f} | {st['wins']}/{st['n']} | {st['t_p']:.4f} / {st['w_p']:.4f} |"
                else:
                    cell += " - | - | - |"
            lines.append(cell)
        lines.append("")

    # ablation table
    lines.append("## 3. Ablation \u2014 is trajectory learning necessary?")
    lines.append("")
    lines.append("| difficulty | method | reward | success | latency (s) | energy | \u0394 (PMEO - variant) | p (paired t) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for difficulty in difficulties:
        cfg_dir = {
            "v2x_hotspot_hard": "U12_L4_40ep_T30",
            "v2x_hotspot_stress": "U16_L4_40ep_T30",
        }.get(difficulty)
        base = root / "ablation" / difficulty / "none" / cfg_dir
        all_ep_rows = read_csv(base / "uav_leo_episodes.csv") if (base / "uav_leo_episodes.csv").exists() else []
        all_sum_rows = read_csv(base / "uav_leo_summary.csv") if (base / "uav_leo_summary.csv").exists() else []
        for method in ("gdrl", "gdrl_no_traj", "gdrl_no_opt"):
            ep_rows = [r for r in all_ep_rows if r["method"] == method]
            sum_rows = [r for r in all_sum_rows if r["method"] == method]
            if not sum_rows:
                continue
            s = sum_rows[0]
            # paired vs full gdrl in the same ablation run
            g_ep = {int(r["episode"]): float(r["reward"]) for r in all_ep_rows if r["method"] == "gdrl"}
            m_ep = {int(r["episode"]): float(r["reward"]) for r in ep_rows}
            diff_cell, p_cell = " \u2014", " \u2014"
            if method != "gdrl":
                st = paired_stats(g_ep, m_ep)
                if st:
                    diff_cell = f"{st['diff']:+.1f}"
                    p_cell = f"{st['t_p']:.4f}"
            lines.append(f"| {difficulty} | {SHORT.get(method, method)} | {fmt(s['reward_mean'])} | "
                         f"{fmt(s['success_rate'], 3)} | {fmt(s['latency_mean'], 4)} | {fmt(s['total_energy_mean'])} | "
                         f"{diff_cell} | {p_cell} |")
        lines.append("")

    # mechanism decomposition: post-move vs current-position exact offloading
    lines.append("## 3b. Mechanism \u2014 decision-order sensitivity (post-move vs current-position offloading)")
    lines.append("")
    lines.append("| difficulty | method | reward | success | latency (s) | \u0394 vs PMEO |")
    lines.append("|---|---|---|---|---|---|")
    for difficulty in difficulties:
        cfg_dir = {"v2x_hotspot_hard": "U12_L4_40ep_T30",
                   "v2x_hotspot_stress": "U16_L4_40ep_T30"}.get(difficulty)
        base = root / "mechanism" / difficulty / "none" / cfg_dir
        mech_sum = read_csv(base / "uav_leo_summary.csv") if (base / "uav_leo_summary.csv").exists() else []
        mech_ep = read_csv(base / "uav_leo_episodes.csv") if (base / "uav_leo_episodes.csv").exists() else []
        cur_ep = {int(r["episode"]): float(r["reward"]) for r in mech_ep if r["method"] == "postmove_exact"}
        if not cur_ep:
            cur_ep = {int(r["episode"]): float(r["reward"]) for r in mech_ep if r["method"] == "current_exact"}
        rows_by_method = {r["method"]: r for r in mech_sum}
        rows_by_method.update({r["method"]: r for r in summary
                               if r["difficulty"] == difficulty and r["method"] in ("gdrl", "predict_tea")})
        order = ["postmove_exact", "gdrl", "current_exact", "predict_tea"]
        for method in order:
            if method not in rows_by_method:
                continue
            r = rows_by_method[method]
            diff_cell = " \u2014"
            if method != "postmove_exact" and cur_ep:
                m_ep = {int(x["episode"]): float(x["reward"]) for x in mech_ep
                        if x["method"] == method} if method not in ("gdrl", "predict_tea") else {
                        int(x["episode"]): float(x["reward"]) for x in episodes
                        if x["difficulty"] == difficulty and x["method"] == method}
                st = paired_stats(m_ep, cur_ep)
                if st:
                    diff_cell = f"{st['diff']:+.1f}"
            lines.append(f"| {difficulty} | {SHORT.get(method, method)} | {fmt(r['reward_mean'])} | "
                         f"{fmt(r['success_rate'], 3)} | {fmt(r['latency_mean'], 4)} | {diff_cell} |")
        lines.append("")

    lines.append("## 4. Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# proposed method PMEO is training-free: run it directly (per difficulty)")
    lines.append("python -m uav_leo_experiment.run_experiment --difficulty v2x_hotspot_hard --episodes 40 "
                 "--horizon 30 --users 12 --seed 73 --skip_slow "
                 "--methods postmove_exact,current_exact,predict_tea,follow_tea,mpc_traj_h3 "
                 "--mpc_horizon 3 --mpc_offload exact --output_dir experiments/uav_leo_v2x_paper_final")
    lines.append("# optional: trained residual-PPO variant GDRL (per difficulty)")
    lines.append("python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_hard --steps 40000")
    lines.append("python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_stress --steps 40000")
    lines.append("```")
    lines.append("")
    lines.append("Figures: `sweep_reward.png`, `sweep_success_rate.png`, `sweep_latency.png`, "
                 "`sweep_energy.png`, `sweep_offload_ratio.png`, `sweep_latency_energy_tradeoff.png`, "
                 "`gdrl_gain_<difficulty>.png` (see `figures/`).")

    output.write_text(chr(10).join(lines).replace("\\u2014", "\u2014").replace("\\u0394", "\u0394") + chr(10), encoding="utf8")


if __name__ == "__main__":
    main()
