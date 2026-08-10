"""
analyze_decisions.py
====================
加载已训练的模型，运行 N 个 episode 收集决策数据，生成可视化图。
用法：
    python analyze_decisions.py \
        --exp_dir experiments/hybrid_small_v2_1200ep \
        --eval_episodes 100 \
        --output_dir experiments/hybrid_small_v2_1200ep/decision_figures
"""

# gdrl/models/feature.py calls get_args() at module level when imported, which
# parses sys.argv immediately and fails on our own --exp_dir/--eval_episodes flags.
# Save and clear argv now (before any GDRL imports), restore before our parse_args().
import sys as _sys
_FULL_ARGV = _sys.argv[:]
_sys.argv = _sys.argv[:1]

import argparse
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


# ─── 样式 ──────────────────────────────────────────────
NODE_COLORS = {
    "local": "#a8dadc",
    "gnb":   "#457b9d",
    "uav":   "#f4a261",
    "leo":   "#2a9d8f",
    "mec":   "#e63946",
}
NODE_ORDER  = ["local", "gnb", "uav", "leo", "mec"]
METHOD_COLORS = {
    "random":   "#8c8c8c",
    "trpo_mlp": "#e07a5f",
    "gdrl":     "#2a9d8f",
    "gdrl_sac": "#f4a261",
}
METHOD_LABELS = {
    "random":   "Random",
    "trpo_mlp": "TRPO-MLP",
    "gdrl":     "GDRL (GCN+TRPO)",
    "gdrl_sac": "GDRL-SAC (GCN+SAC)",
}


def parse_args():
    # Parse _FULL_ARGV directly — do NOT restore sys.argv, because get_args() in
    # feature.py is called again on every model instantiation and would choke on
    # --exp_dir / --eval_episodes if sys.argv were restored to our full arg list.
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir",       type=str, default="experiments/hybrid_small_v2_1200ep")
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--output_dir",    type=str, default=None)
    p.add_argument("--scenario",      type=str, default="hybrid_small")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--strict_load",   action="store_true", default=True,
                   help="严格加载训练时模型与 AMN；默认开启。")
    p.add_argument("--no_strict_load", dest="strict_load", action="store_false")
    p.add_argument("--allow_retrain_amn", action="store_true",
                   help="允许评估阶段重训 AMN；默认关闭。")
    return p.parse_args(_FULL_ARGV[1:])


# ─── 环境构建（复用 compare_baselines 逻辑）──────────────────────────────
def build_env(scenario, exp_dir, seed, allow_retrain_amn=False):
    """搭建 hybrid 环境，加载 AMN。"""
    import json, sys
    # compare_baselines imports arg_parser which calls parse_args() on sys.argv at module level;
    # temporarily clear argv so it doesn't choke on analyze_decisions' own flags.
    _saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        from compare_baselines import build_components, train_or_load_amn, make_raw_env
        from experiment_config import apply_scenario, validate_scenario
    finally:
        sys.argv = _saved_argv

    cfg = json.loads((Path(exp_dir) / "run_config.json").read_text())

    # 构造 args namespace
    import types
    args = types.SimpleNamespace(
        scenario=cfg["scenario"],
        U=cfg["U"], L=cfg["L"], N=cfg.get("N", 0), T=cfg["T"],
        G=cfg.get("G", 0), V=cfg.get("V", 0), M=cfg.get("M", 0),
        experiment_root="experiments",
        amn_epochs=cfg.get("amn_epochs", 100),
        episodes=cfg.get("episodes", 1200),
        force_retrain_amn=allow_retrain_amn,
        device="cpu",
        seed=seed,
    )
    validate_scenario(args)

    output_dir = Path(exp_dir)
    user_requests, user_lists, autoencoder_dis, autoencoder_con, ResetFunction = build_components(args, output_dir)
    autoencoder_dis, autoencoder_con, encoder_dis, encoder_con = train_or_load_amn(
        args, autoencoder_dis, autoencoder_con, output_dir
    )

    def make_env():
        return make_raw_env(args, user_requests, user_lists, encoder_dis, encoder_con, ResetFunction)

    return make_env, args


# ─── 评估单个方法 ───────────────────────────────────────────────────────
def evaluate_method(method, model, make_env_fn, n_episodes):
    """运行 n_episodes，收集决策。"""
    env = make_env_fn()
    decisions = []
    rewards   = []
    diagnostics = {
        "target_user_decisions": 0,
        "recorded_user_decisions": 0,
        "invalid_steps": 0,
        "full_steps": 0,
        "processed_users_hist": Counter(),
        "invalid_reasons": Counter(),
    }

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_decisions = []

    total_steps = 0
    target = n_episodes * getattr(env, "T", 100)
    diagnostics["target_user_decisions"] = target * getattr(env, "U", 0)

    while total_steps < target:
        if model == "random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward  += reward
        total_steps += 1

        d = info.get("decisions")
        if d:
            ep_decisions.append(d)
            processed_users = int(info.get("processed_users", len(d.get("node_types", []))))
            diagnostics["recorded_user_decisions"] += len(d.get("node_types", []))
            diagnostics["processed_users_hist"][processed_users] += 1
            if processed_users == getattr(env, "U", processed_users):
                diagnostics["full_steps"] += 1
            else:
                diagnostics["invalid_steps"] += 1
            invalid_reason = info.get("invalid_reason")
            if invalid_reason:
                diagnostics["invalid_reasons"][invalid_reason] += 1

        if terminated or truncated:
            rewards.append(ep_reward)
            decisions.extend(ep_decisions)
            ep_reward = 0.0
            ep_decisions = []
            obs, _ = env.reset()

    env.close()
    return decisions, rewards, diagnostics


def summarize_diagnostics(diag):
    target = max(int(diag.get("target_user_decisions", 0)), 1)
    recorded = int(diag.get("recorded_user_decisions", 0))
    full_steps = int(diag.get("full_steps", 0))
    invalid_steps = int(diag.get("invalid_steps", 0))
    processed_hist = {str(k): int(v) for k, v in sorted(diag.get("processed_users_hist", {}).items())}
    invalid_reasons = {k: int(v) for k, v in diag.get("invalid_reasons", {}).items()}
    return {
        "target_user_decisions": target,
        "recorded_user_decisions": recorded,
        "recorded_ratio": recorded / target,
        "full_steps": full_steps,
        "invalid_steps": invalid_steps,
        "processed_users_hist": processed_hist,
        "invalid_reasons": invalid_reasons,
    }


# ─── 统计决策 ───────────────────────────────────────────────────────────
def aggregate(decisions):
    node_types    = []
    offload_ratios = []
    is_local_list  = []
    for step in decisions:
        node_types.extend(step["node_types"])
        offload_ratios.extend(step["offload_ratios"])
        is_local_list.extend(step["is_local"])
    return node_types, offload_ratios, is_local_list


# ─── 可视化 ─────────────────────────────────────────────────────────────
def plot_node_distribution(all_data, output_dir, methods):
    """节点选择分布堆叠柱状图（各方法对比）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))

    for node in NODE_ORDER:
        vals = []
        for m in methods:
            cnt = Counter(all_data[m]["node_types"])
            total = sum(cnt.values()) or 1
            vals.append(cnt.get(node, 0) / total * 100)
        bars = ax.bar(x, vals, bottom=bottoms, color=NODE_COLORS[node],
                      label=node.upper(), width=0.55, edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_ylabel("节点选择比例 (%)", fontsize=11)
    ax.set_title("各方法的卸载目标节点分布", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    path = output_dir / "decision_node_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_offload_ratio(all_data, output_dir, methods):
    """卸载比例分布箱线图。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    data_list  = []
    colors_list = []
    labels     = []
    for m in methods:
        # 只取卸载决策（排除 is_local=True）的 ratio
        ratios = [r for r, loc in zip(all_data[m]["offload_ratios"], all_data[m]["is_local"]) if not loc]
        data_list.append(ratios)
        colors_list.append(METHOD_COLORS.get(m, "#333"))
        labels.append(METHOD_LABELS.get(m, m))

    bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("卸载比例 offload_ratio", fontsize=11)
    ax.set_title("各方法卸载比例分布（排除纯本地）", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = output_dir / "decision_offload_ratio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_local_rate(all_data, output_dir, methods):
    """本地处理率对比柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.style.use("seaborn-v0_8-whitegrid")

    x = np.arange(len(methods))
    local_rates = []
    for m in methods:
        is_local = all_data[m]["is_local"]
        rate = sum(is_local) / max(len(is_local), 1) * 100
        local_rates.append(rate)

    bars = ax.bar(x, local_rates,
                  color=[METHOD_COLORS.get(m, "#555") for m in methods],
                  width=0.5, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, local_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_ylabel("本地处理比例 (%)", fontsize=11)
    ax.set_title("各方法选择纯本地处理的频率", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(local_rates) * 1.3 + 5)
    plt.tight_layout()
    path = output_dir / "decision_local_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_node_heatmap(all_data, output_dir, methods):
    """节点选择热力图：方法 × 节点类型。"""
    node_types = [n for n in NODE_ORDER if any(n in all_data[m]["node_types"] for m in methods)]
    matrix = np.zeros((len(methods), len(node_types)))
    for i, m in enumerate(methods):
        cnt = Counter(all_data[m]["node_types"])
        total = sum(cnt.values()) or 1
        for j, n in enumerate(node_types):
            matrix[i, j] = cnt.get(n, 0) / total * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="选择比例 (%)")
    ax.set_xticks(range(len(node_types)))
    ax.set_xticklabels([n.upper() for n in node_types], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=10)
    ax.set_title("节点选择热力图（方法 × 节点类型）", fontsize=13, fontweight="bold")
    for i in range(len(methods)):
        for j in range(len(node_types)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i,j] < 60 else "white")
    plt.tight_layout()
    path = output_dir / "decision_node_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ─── 主流程 ─────────────────────────────────────────────────────────────
def main():
    import sys
    args = parse_args()
    exp_dir    = Path(args.exp_dir)
    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "decision_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Building environment from {exp_dir} ...")
    make_env_fn, env_args = build_env(args.scenario, exp_dir, args.seed, args.allow_retrain_amn)

    # The force-retrain above may have regenerated edge_attr.pt/edge_index.pt in
    # the shared cache with a different random state.  Copy the originals from
    # exp_dir (saved at training time) back to the shared cache so that
    # CustomFeaturesExtractor.load() reconstructs the exact same graph topology.
    import shutil as _shutil
    from experiment_config import scenario_tag as _stag
    import types as _types, json as _jj
    _cfg2 = _jj.loads((exp_dir / "run_config.json").read_text())
    _fake = _types.SimpleNamespace(**{k: _cfg2.get(k, 0) for k in ("U","G","V","L","M","T","N")},
                                   scenario=_cfg2["scenario"], experiment_root="experiments")
    _shared = Path("experiments") / _stag(_fake)
    _shared.mkdir(parents=True, exist_ok=True)
    for _fname in ("edge_attr.pt", "edge_index.pt"):
        _src = exp_dir / _fname
        if _src.exists():
            _shutil.copy2(_src, _shared / _fname)
    print(f"Copied edge files from {exp_dir.name} → {_shared.name}")

    # Build a sys.argv that matches the training configuration so that
    # CustomFeaturesExtractor.get_args() inside model.load() reconstructs
    # the correct layer dimensions (graph_norm, var_norm, fnn1 sizes).
    import json as _json
    _cfg = _json.loads((exp_dir / "run_config.json").read_text())
    _model_argv = [
        sys.argv[0],
        "--scenario", _cfg.get("scenario", "hybrid_small"),
        "--U", str(_cfg.get("U", 5)),
        "--G", str(_cfg.get("G", 0)),
        "--V", str(_cfg.get("V", 0)),
        "--L", str(_cfg.get("L", 8)),
        "--M", str(_cfg.get("M", 0)),
        "--T", str(_cfg.get("T", 100)),
    ]

    methods = ["random", "trpo_mlp", "gdrl", "gdrl_sac"]
    all_data = {}

    for method in methods:
        model_path = exp_dir / f"{method}_model.zip"
        print(f"\n── {method} ──")

        if method == "random":
            model = "random"
        elif model_path.exists():
            sys.argv = _model_argv   # ensure get_args() returns correct dims
            try:
                if method in ("gdrl", "trpo_mlp"):
                    from sb3_contrib import TRPO
                    model = TRPO.load(str(model_path), device="cpu")
                elif method == "gdrl_sac":
                    from stable_baselines3 import SAC
                    model = SAC.load(str(model_path), device="cpu")
                else:
                    from stable_baselines3 import PPO
                    model = PPO.load(str(model_path), device="cpu")
            finally:
                sys.argv = sys.argv[:1]  # clear again after load
            print(f"  Loaded {model_path.name}")
        else:
            print(f"  Model not found, skipping: {model_path}")
            continue

        decisions, rewards, diagnostics = evaluate_method(method, model, make_env_fn, args.eval_episodes)
        node_types, offload_ratios, is_local = aggregate(decisions)
        all_data[method] = {
            "node_types":     node_types,
            "offload_ratios": offload_ratios,
            "is_local":       is_local,
            "rewards":        rewards,
            "diagnostics":    summarize_diagnostics(diagnostics),
        }
        cnt = Counter(node_types)
        total = sum(cnt.values()) or 1
        print(f"  steps={total} | node dist: " +
              " | ".join(f"{k}={v/total*100:.1f}%" for k, v in sorted(cnt.items())))
        print(f"  avg offload_ratio={np.mean(offload_ratios):.3f}  "
              f"local_rate={sum(is_local)/max(len(is_local),1)*100:.1f}%  "
              f"reward_mean={np.mean(rewards):.1f}")
        diag = all_data[method]["diagnostics"]
        print(f"  valid_user_decisions={diag['recorded_user_decisions']}/{diag['target_user_decisions']} "
              f"({diag['recorded_ratio']*100:.2f}%) | invalid_steps={diag['invalid_steps']} "
              f"| full_steps={diag['full_steps']}")
        if diag["invalid_reasons"]:
            print("  invalid_reasons=" + ", ".join(f"{k}:{v}" for k, v in sorted(diag["invalid_reasons"].items())))

        # 保存 pkl
        with open(output_dir / f"decisions_{method}.pkl", "wb") as f:
            pickle.dump(decisions, f)
        with open(output_dir / f"decision_diagnostics_{method}.pkl", "wb") as f:
            pickle.dump(all_data[method]["diagnostics"], f)

    present = [m for m in methods if m in all_data]
    print("\nGenerating figures ...")
    plot_node_distribution(all_data, output_dir, present)
    plot_offload_ratio(all_data, output_dir, present)
    plot_local_rate(all_data, output_dir, present)
    plot_node_heatmap(all_data, output_dir, present)
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
