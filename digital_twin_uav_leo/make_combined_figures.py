"""Combined paper-ready figures.

fig_combined_k3_tau.png       tau vs reward: PMEO-M-Eco under i.i.d. / burst /
                              random-burst on the same axes (single-UAV panel too).
fig_combined_k3_adaptive.png  2 panels (burst / random-burst): fixed-period curve
                              + load-aware adaptive points (reward vs syncs).
"""
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

C = {"iid": "#1565C0", "burst": "#C62828", "burst_random": "#E65100"}


def _load(name):
    p = RESULTS / name
    if not p.exists():
        print(f"skip: {p} missing")
        return None
    return pd.read_csv(p)


def fig_combined_tau():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    panels = [
        ("k3 (3 UAV / 51 users)", "rq1_k3.csv", "rq1_k3_burst.csv",
         "rq1_k3_burst_random.csv", "pmeo_m_eco", axes[0]),
        ("single UAV (12 users)", "rq1_single.csv", "rq1_single_burst.csv",
         "rq1_single_burst_random.csv", "postmove_exact", axes[1]),
    ]
    for title, f_iid, f_b, f_br, method, ax in panels:
        for fname, key, color in [(f_iid, "iid", C["iid"]),
                                  (f_b, "burst", C["burst"]),
                                  (f_br, "burst_random", C["burst_random"])]:
            df = _load(fname)
            if df is None:
                continue
            d = df[df["method"] == method].groupby("tau")["reward"].mean()
            d = d.reset_index().sort_values("tau")
            ax.plot(d["tau"], d["reward"], "o-", color=color, label={
                "iid": "i.i.d. (ordinary)",
                "burst": "burst (known window)",
                "burst_random": "burst (random window)",
            }[key])
        ax.set_xlabel("sync period tau (slots)")
        ax.set_ylabel("reward (higher is better)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Staleness cost: sync-period tradeoff under different traffic dynamics",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS / "fig_combined_k3_tau.png", bbox_inches="tight")
    print("saved fig_combined_k3_tau.png")


def fig_combined_adaptive():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    panels = [("rq4_k3_burst.csv", "burst (known window)", axes[0]),
              ("rq4_k3_burst_random.csv", "burst (random window)", axes[1])]
    for fname, title, ax in panels:
        df = _load(fname)
        if df is None:
            continue
        fixed = df[df["sync_mode"] == "fixed"].copy().sort_values("syncs")
        adap = df[df["sync_mode"] == "adaptive"].copy()
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
        ax.set_ylabel("reward (higher is better)")
        ax.set_title(f"k3 - {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Event-triggered sync beats fixed period in bursty traffic",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RESULTS / "fig_combined_k3_adaptive.png", bbox_inches="tight")
    print("saved fig_combined_k3_adaptive.png")


if __name__ == "__main__":
    fig_combined_tau()
    fig_combined_adaptive()
