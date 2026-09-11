"""Figure generation for the digital-twin UAV-LEO line (reads results/*.csv)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

plt.rcParams.update({
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C_METHODS = {"postmove_exact": "#1565C0", "current_exact": "#2E7D32",
             "mpc_traj_h3": "#C62828", "pmeo_m_eco": "#1565C0",
             "mpc_m_h3": "#C62828", "current_exact_m": "#2E7D32"}


def _load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"skip: {p} missing")
        return None
    return pd.read_csv(p)


def fig_tau_tradeoff():
    df = _load("rq1_single.csv")
    if df is None:
        return
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario]
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
        for ax, metric in zip(axes, ("reward", "completion", "agree")):
            for method in ["postmove_exact", "current_exact", "mpc_traj_h3"]:
                d = sub[sub["method"] == method].sort_values("tau")
                ax.plot(d["tau"], d[metric], marker="o", label=method,
                        color=C_METHODS[method])
            ax.set_xlabel("sync period tau (slots)")
            ax.set_title(f"{scenario} - {metric}")
            ax.grid(alpha=0.3)
            if metric == "reward":
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_tau_tradeoff_{scenario}.png", bbox_inches="tight")
        print(f"saved fig_tau_tradeoff_{scenario}.png")


def fig_tau_tradeoff_k3():
    df = _load("rq1_k3.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for ax, metric in zip(axes, ("reward", "completion", "agree")):
        for method in ["pmeo_m_eco", "current_exact_m", "mpc_m_h3"]:
            d = df[df["method"] == method].groupby("tau")[metric].mean().reset_index()
            d = d.sort_values("tau")
            ax.plot(d["tau"], d[metric], marker="o", label=method, color=C_METHODS[method])
        ax.set_xlabel("sync period tau (slots)")
        ax.set_title(f"k3 - {metric}")
        ax.grid(alpha=0.3)
        if metric == "reward":
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_tau_tradeoff_k3.png", bbox_inches="tight")
    print("saved fig_tau_tradeoff_k3.png")


def fig_tau_tradeoff_burst():
    for name, out_name, methods in [
            ("rq1_single_burst.csv", "fig_tau_tradeoff_burst_single.png",
             ["postmove_exact", "current_exact", "mpc_traj_h3"]),
            ("rq1_k3_burst.csv", "fig_tau_tradeoff_burst_k3.png",
             ["pmeo_m_eco", "current_exact_m", "mpc_m_h3"]),
            ("rq1_single_burst_random.csv",
             "fig_tau_tradeoff_burst_random_single.png",
             ["postmove_exact", "current_exact", "mpc_traj_h3"]),
            ("rq1_k3_burst_random.csv", "fig_tau_tradeoff_burst_random_k3.png",
             ["pmeo_m_eco", "current_exact_m", "mpc_m_h3"])]:
        df = _load(name)
        if df is None:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
        for ax, metric in zip(axes, ("reward", "completion", "agree")):
            for method in methods:
                d = df[df["method"] == method].groupby("tau")[metric].mean()
                d = d.reset_index().sort_values("tau")
                ax.plot(d["tau"], d[metric], marker="o", label=method,
                        color=C_METHODS[method])
            ax.set_xlabel("sync period tau (slots)")
            ax.set_title(
                f"{'random-burst' if 'random' in name else 'burst'} - {metric}")
            ax.grid(alpha=0.3)
            if metric == "reward":
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / out_name, bbox_inches="tight")
        print(f"saved {out_name}")


def fig_predictor_robustness():
    df = _load("rq2_single.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    order = ["freeze", "linear", "resample", "linear_noise"]
    eps_map = {0.05: "noise 5%", 0.1: "noise 10%", 0.2: "noise 20%"}
    for name in order:
        d = df[df["predictor"] == name]
        if name == "linear_noise":
            for _, row in d.iterrows():
                ax.plot(0.0, row["reward"], "x", color="#C62828",
                        label=eps_map[row["eps"]] if row["eps"] == 0.05 else None)
        elif len(d):
            ax.plot(0.0, d["reward"].iloc[0], "o", color="#1565C0", label=name)
    ax.axhline(df["reward"].max(), color="#90A4AE", ls="--", lw=1)
    ax.set_xticks([])
    ax.set_ylabel("reward (tau=3, hard)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_predictor_robustness.png", bbox_inches="tight")
    print("saved fig_predictor_robustness.png")


def fig_adaptive_sync():
    df = _load("rq4_single.csv")
    if df is None:
        return
    fixed = df[df["sync_mode"] == "fixed"]
    adap = df[df["sync_mode"] == "adaptive"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(fixed["syncs"], fixed["reward"], "o-", color="#1565C0",
            label="fixed-period (tau=1..10)")
    for _, row in adap.iterrows():
        ax.plot(row["syncs"], row["reward"], "s", color="#C62828",
                label=f"adaptive delta={int(row['delta'])}m")
    ax.set_xlabel("avg sync count per episode")
    ax.set_ylabel("reward (hard)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_adaptive_sync.png", bbox_inches="tight")
    print("saved fig_adaptive_sync.png")


def fig_burst():
    """Bursty-traffic variants: adaptive (load-aware) sync dominates the
    fixed-period frontier in the syncs-vs-reward plane."""
    single = _load("rq4_single_burst.csv")
    k3 = _load("rq4_k3_burst.csv")
    for df, name, ylab in [(single, "single", "reward (burst, single-UAV)"),
                           (k3, "k3", "reward (burst, k3)")]:
        if df is None:
            continue
        fixed = df[df["sync_mode"] == "fixed"].copy().sort_values("syncs")
        adap = df[df["sync_mode"] == "adaptive"].copy()
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(fixed["syncs"], fixed["reward"], "o-", color="#1565C0",
                label="fixed-period (tau=1..10)")
        load = adap[adap["task_mode"] == "load"]
        pu = adap[adap["task_mode"] != "load"]
        if len(load):
            ax.plot(load["syncs"], load["reward"], "s", color="#C62828",
                    label="adaptive load-aware")
            for _, r in load.iterrows():
                ax.annotate(f"{int(r['delta'])}m", (r["syncs"], r["reward"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
        if len(pu):
            ax.plot(pu["syncs"], pu["reward"], "^", color="#F9A825",
                    label="adaptive per-user")
        ax.set_xlabel("avg sync count per episode")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_burst_{name}.png", bbox_inches="tight")
        print(f"saved fig_burst_{name}.png")


def fig_burst_random():
    """Random-offset burst variant: adaptive load-aware sync dominates the
    fixed-period frontier even when the twin does not know the burst phase."""
    for fname, name, ylab in [("rq4_single_burst_random.csv", "single",
                               "reward (random-burst, single-UAV)"),
                              ("rq4_k3_burst_random.csv", "k3",
                               "reward (random-burst, k3)")]:
        df = _load(fname)
        if df is None:
            continue
        fixed = df[df["sync_mode"] == "fixed"].copy().sort_values("syncs")
        adap = df[df["sync_mode"] == "adaptive"].copy()
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(fixed["syncs"], fixed["reward"], "o-", color="#1565C0",
                label="fixed-period (tau=1..10)")
        load = adap[adap["task_mode"] == "load"]
        pu = adap[adap["task_mode"] != "load"]
        if len(load):
            ax.plot(load["syncs"], load["reward"], "s", color="#C62828",
                    label="adaptive load-aware")
            for _, r in load.iterrows():
                ax.annotate(f"{int(r['delta'])}m", (r["syncs"], r["reward"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
        if len(pu):
            ax.plot(pu["syncs"], pu["reward"], "^", color="#F9A825",
                    label="adaptive per-user")
        ax.set_xlabel("avg sync count per episode")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"fig_burst_random_{name}.png", bbox_inches="tight")
        print(f"saved fig_burst_random_{name}.png")


if __name__ == "__main__":
    fig_tau_tradeoff()
    fig_tau_tradeoff_k3()
    fig_tau_tradeoff_burst()
    fig_predictor_robustness()
    fig_adaptive_sync()
    fig_burst()
    fig_burst_random()
