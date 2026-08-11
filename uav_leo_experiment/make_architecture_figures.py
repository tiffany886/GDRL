"""Generate the system-architecture and PMEO decision-pipeline figures.

Outputs (paper figures, style consistent with make_paper_figures.py):
  experiments/uav_leo_v2x_paper_final/figures/architecture.png
  experiments/uav_leo_v2x_paper_final/figures/pmeo_pipeline.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, Circle, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = Path("experiments/uav_leo_v2x_paper_final/figures")
OUT.mkdir(parents=True, exist_ok=True)

LOCAL_C = "#2E7D32"
UAV_C = "#1565C0"
LEO_C = "#FFB300"
LEO_EDGE = "#B28704"
HOT_C = "#C62828"
ROAD_C = "#CFD8DC"
TEXT_C = "#333333"
GREY = "#757575"


def save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _arrow(ax, p1, p2, color, style="-", lw=1.6):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=style))


def architecture():
    """System architecture: roads, vehicles, hotspot, UAV relay, LEO edge."""
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.axis("off")

    for y in (230, 420, 610, 800):
        ax.plot([0, 1000], [y, y], color=ROAD_C, lw=9, solid_capstyle="round", zorder=0)
    ax.add_patch(mpatches.Rectangle((0, 0), 1000, 1000, fill=False, ec="#9e9e9e", lw=1.4, zorder=1))
    ax.text(18, 966, "Ground service area: 1000 x 1000 m, road network", fontsize=9, color=TEXT_C, zorder=6)

    hx, hy, hr = 590, 445, 105
    ax.add_patch(Circle((hx, hy), hr, facecolor="#FFCDD2", edgecolor=HOT_C, lw=2.0,
                        alpha=0.5, zorder=2))
    _arrow(ax, (hx - 8, hy + 6), (hx + 55, hy - 42), HOT_C, lw=2.4)
    ax.text(hx, hy + hr + 12, "Mobile hotspot", ha="center", fontsize=10, color=HOT_C,
            fontweight="bold", zorder=6)
    ax.text(hx, hy + hr - 8, "radius 90-100 m, 18-20 m/s", ha="center", fontsize=8.2,
            color=HOT_C, zorder=6)

    rng = np.random.default_rng(11)
    vehicles = []
    for off in rng.normal(0, 1.0, (9, 2)) * 30:
        vx, vy = hx + off[0], hy + off[1]
        vehicles.append((vx, vy))
        ax.add_patch(mpatches.Rectangle((vx - 7, vy - 5), 14, 10, facecolor="#455A64",
                                        edgecolor="white", lw=0.7, zorder=4))
    ax.text(hx - 165, hy + 62, "Vehicles with tasks\n(12-16 users)", fontsize=8.8,
            color="#455A64", zorder=6)

    ux, uy = 415, 250
    ax.add_patch(Circle((ux, uy), 15, facecolor=UAV_C, edgecolor="white", lw=1.4, zorder=5))
    for dx in (-20, 20):
        ax.plot([ux + dx - 9, ux + dx + 9], [uy + 13, uy + 13], color=UAV_C, lw=2.4, zorder=4)
    ax.text(ux - 4, uy - 34, "UAV\nedge server\n50-60 GHz", ha="center", fontsize=8.8,
            color=UAV_C, fontweight="bold", zorder=6)
    _arrow(ax, (ux + 18, uy + 20), (hx - 65, hy + 25), UAV_C, style=(0, (5, 3)), lw=1.8)
    ax.text((ux + hx) / 2 + 30, (uy + hy) / 2 + 34, "Demand-predictive\nUAV trajectory",
            fontsize=8.4, color=UAV_C, zorder=6)

    ax.add_patch(Arc((500, 880), 980, 330, theta1=180, theta2=360, ec="#E0C46B", lw=2.0, zorder=1))
    for lx, ly in ((150, 900), (500, 940), (850, 900)):
        ax.add_patch(Circle((lx, ly), 13, facecolor=LEO_C, edgecolor=LEO_EDGE, lw=1.3, zorder=5))
        _arrow(ax, (lx + 24, ly + 15), (lx - 24, ly + 15), LEO_EDGE, lw=1.4)
    ax.text(500, 975, "LEO constellation (4 sats, 1 THz edge compute)", ha="center", fontsize=9.5,
            color=LEO_EDGE, fontweight="bold", zorder=6)

    for (vx, vy) in vehicles[1:5]:
        _arrow(ax, (vx + 8, vy + 6), (ux - 18, uy - 10), UAV_C, lw=1.1)
    ax.text(322, 168, "User-UAV access link\n(1.0-1.1 MHz)", fontsize=8.4, color=UAV_C, zorder=6)

    _arrow(ax, (ux + 20, uy + 16), (478, 926), LEO_C, style=(0, (4, 2)), lw=1.8)
    ax.text(545, 620, "UAV-LEO backhaul\n(20 Mbps)", fontsize=8.4, color=LEO_EDGE, zorder=6)

    ax.text(hx - 20, hy - hr - 34, "Local compute\n(user CPU 5 GHz)", fontsize=8.4,
            color=LOCAL_C, zorder=6)

    for i, (c, label) in enumerate([
            (LOCAL_C, "Local computation"),
            (UAV_C, "UAV edge offload"),
            (LEO_C, "LEO edge offload")]):
        y = 60 - i * 34
        ax.add_patch(mpatches.Rectangle((30, y - 8), 16, 16, facecolor=c, edgecolor="white",
                                        lw=0.8, zorder=6))
        ax.text(56, y, label, fontsize=9, va="center", color=TEXT_C, zorder=6)

    save(fig, "architecture.png")


def pipeline():
    """PMEO per-slot decision pipeline with the per-user cost grid."""
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, face, edge, text, fs=8.6, tc=TEXT_C):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, lw=1.4, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4)

    def flow(x1, x2, y):
        _arrow(ax, (x1, y), (x2, y), GREY, lw=1.8)

    ax.text(50, 41.5, "PMEO per-slot decision pipeline  (training-free, no lookahead)",
            ha="center", fontsize=10.5, color=TEXT_C, fontweight="bold")

    box(1, 26, 21, 12, "#E8EAF6", "#3F51B5",
        "Observe at slot t\nuser tasks & positions,\nhotspot, UAV, backlogs", fs=8.4)
    flow(22.5, 26.5, 32)

    box(27, 26, 22, 12, "#E3F2FD", UAV_C,
        "Step 1: Expert move\n"
        "target = 0.5*centroid\n"
        "        + 0.5*hotspot(t+1)\n"
        "|move| <= v_max", fs=8.2)
    flow(49.5, 53.5, 32)

    box(54, 26, 26, 12, "#FFF8E1", HOT_C,
        "Step 2: Post-move exact offload\n"
        "argmin over {local, UAV, LEO}\n"
        "x ratios {0, .25, .5, .75, 1}\n"
        "using real sim at pos_new", fs=8.2)
    flow(80.5, 84.5, 32)

    box(85, 26, 14, 12, "#E8F5E9", LOCAL_C,
        "Execute\nmove +\noffload", fs=8.4)

    gx, gy, gw, gh = 56, 4, 22, 16
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="white", edgecolor="#9e9e9e", lw=1.1, zorder=2))
    ax.text(gx + gw / 2, gy + gh - 2.2, "Per-user cost grid (15 cells)",
            ha="center", fontsize=7.8, color=TEXT_C, zorder=4)
    rows = ["local", "UAV", "LEO"]
    cols = ["0", ".25", ".5", ".75", "1"]
    cell_w, cell_h = gw / 5.0, (gh - 5.0) / 3.0
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            cx = gx + j * cell_w
            cy = gy + 5.0 + i * cell_h
            ax.add_patch(mpatches.Rectangle((cx, cy), cell_w, cell_h, facecolor="none",
                                            edgecolor="#CFD8DC", lw=0.6, zorder=3))
            if i == 0:
                ax.text(cx + cell_w / 2, cy + cell_h / 2, col, ha="center", va="center",
                        fontsize=6.2, color="#9e9e9e", zorder=4)
        ax.text(gx - 0.3, cy + cell_h / 2, row, ha="right", va="center", fontsize=6.6,
                color=TEXT_C, zorder=4)
    _arrow(ax, (gx + gw / 2, gy + gh + 0.2), (gx + gw / 2, gy + gh + 0.9), GREY, lw=1.4)

    ax.text(50, 1.6,
            "Complexity: O(U x 15) link/compute simulations per slot;  post-move position "
            "=> exact offload is per-slot globally optimal",
            ha="center", fontsize=8.0, color=GREY)

    save(fig, "pmeo_pipeline.png")


if __name__ == "__main__":
    architecture()
    pipeline()
