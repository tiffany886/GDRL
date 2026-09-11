"""Digital-twin architecture figure for the thesis (Chapter 4).

Physical world (users -> UAV -> LEO), digital twin layer (state replica +
predictor + sync control), and the PMEO decision module.  Solid arrows = data
flow; dashed arrows = sync / probing.
Output: results/fig_dt_architecture.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

C_USER = "#2E7D32"
C_UAV = "#1565C0"
C_LEO = "#FFB300"
C_DT = "#6A1B9A"
C_SYNC = "#C62828"
C_DEC = "#00695C"
C_BG = "#FFFFFF"
TEXT_C = "#333333"


def _box(ax, x, y, w, h, text, fc, lw=1.6, fs=10.5, sub=None, sub_fs=8.5,
         tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec="white", lw=lw, zorder=3))
    ax.text(x + w / 2, y + h - 0.09, text, ha="center", va="top",
            fontsize=fs, color=tc, zorder=4, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + 0.08, sub, ha="center", va="bottom",
                fontsize=sub_fs, color="white", zorder=4)


def _arrow(ax, p1, p2, color, ls="-", lw=1.8, zorder=5):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=2, shrinkB=2), zorder=zorder)


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------- left: physical world ----------
    ax.text(15, 96.5, "Physical world (simulator)", ha="center",
            fontsize=12, color=TEXT_C, fontweight="bold")
    _box(ax, 3, 66, 24, 26, "Vehicle users  (U)", C_USER,
         sub="positions / velocities / task bits\narrival near mobile hotspots")
    _box(ax, 3, 34, 24, 26, "UAV edge  (K)", C_UAV,
         sub="access relay + edge server\nposition / battery / queue")
    _box(ax, 3, 2, 24, 26, "LEO edge  (L)", C_LEO,
         sub="high-compute satellite\nbackhaul-limited", tc="#333333")
    _arrow(ax, (15, 66), (15, 62), C_USER)
    _arrow(ax, (15, 34), (15, 30), C_UAV)
    ax.text(2, 89.5, "state:", fontsize=9, color=TEXT_C)

    # ---------- right: digital twin ----------
    ax.text(80, 96.5, "Digital twin layer (Chapter 4)", ha="center",
            fontsize=12, color=C_DT, fontweight="bold")
    _box(ax, 68, 66, 24, 26, "Twin state replica", C_DT,
         sub="users / hotspots / tasks\n(copy of last sync)")
    _box(ax, 68, 34, 24, 26, "Predictor (extrapolation)", C_DT,
         sub="linear positions\nfreeze / resample tasks")
    _box(ax, 68, 2, 24, 26, "Sync control", C_SYNC,
         sub="load-aware error metric\nthreshold delta -> resync")

    # ---------- bottom: decision module ----------
    _box(ax, 33, 34, 30, 26, "PMEO decision module (Chapter 3)", C_DEC,
         sub="demand-aware trajectory +\npost-move exact offloading\nO(U*K*15) per slot")
    _box(ax, 33, 2, 30, 26, "Physical execution", "#455A64",
         sub="UAV moves / offload actions\nlatency, energy, drops")

    # ---------- arrows ----------
    # physical -> twin: full sync (solid) and probe (dashed)
    _arrow(ax, (27, 79), (68, 79), C_SYNC, ls="-", lw=2.2)
    ax.text(47.5, 80.8, "full-state sync (periodic / event-triggered)",
            ha="center", fontsize=8.5, color=C_SYNC)
    _arrow(ax, (27, 72), (68, 70), C_SYNC, ls="--", lw=1.6)
    ax.text(47.5, 73.6, "probe (few users) -> load-aware resync",
            ha="center", fontsize=8.5, color=C_SYNC)
    # twin -> predictor -> sync control (internal)
    _arrow(ax, (80, 66), (80, 62), C_DT, lw=1.6)
    _arrow(ax, (80, 34), (80, 30), C_DT, lw=1.6)
    _arrow(ax, (92, 28), (92, 6), C_SYNC, lw=1.6)
    # twin observation -> decision
    _arrow(ax, (68, 53), (63, 53), C_DEC, lw=2.0)
    ax.text(65.2, 55.0, "twin observation", ha="center", fontsize=8.5,
            color=C_DEC)
    # decision -> physical execution
    _arrow(ax, (48, 34), (48, 30), C_DEC, lw=2.0)
    ax.text(50, 32.2, "actions", ha="center", fontsize=8.5, color=C_DEC)
    # execution -> physical world (feedback)
    _arrow(ax, (48, 6), (27, 6), "#455A64", lw=1.6)

    fig.tight_layout()
    fig.savefig(OUT / "fig_dt_architecture.png", bbox_inches="tight")
    print("saved", OUT / "fig_dt_architecture.png")


if __name__ == "__main__":
    main()
