"""Multi-UAV architecture figures for the paper (PMEO-M / PMEO-M-Eco line).

Outputs:
  experiments/uav_leo_v2x_paper_final/figures/architecture_multi_uav.png
  experiments/uav_leo_v2x_paper_final/figures/pmeo_m_pipeline.png
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


def _uav(ax, x, y, color=UAV_C):
    ax.add_patch(Circle((x, y), 15, facecolor=color, edgecolor="white", lw=1.4, zorder=5))
    for dx in (-20, 20):
        ax.plot([x + dx - 9, x + dx + 9], [y + 13, y + 13], color=color, lw=2.4, zorder=4)


def architecture_multi_uav():
    """Multi-UAV system: roads, vehicles, mobile hotspots, K UAVs, LEO edge."""
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.axis("off")

    for y in (230, 420, 610, 800):
        ax.plot([0, 1000], [y, y], color=ROAD_C, lw=9, solid_capstyle="round", zorder=0)
    ax.add_patch(mpatches.Rectangle((0, 0), 1000, 1000, fill=False, ec="#9e9e9e", lw=1.4, zorder=1))
    ax.text(18, 966, "Ground service area: 2 x 2 km, road network (K = 3 UAVs, 51 users)",
            fontsize=9, color=TEXT_C, zorder=6)

    # hotspots (3 in k3): one main + two secondary
    hotspots = [((590, 445), 105, "Mobile hotspot", "radius 90-130 m, 18-20 m/s"),
                ((240, 640), 60, None, None),
                ((780, 290), 60, None, None)]
    for (hx, hy), hr, label, sub in hotspots:
        ax.add_patch(Circle((hx, hy), hr, facecolor="#FFCDD2", edgecolor=HOT_C, lw=1.6,
                            alpha=0.45, zorder=2))
        if label:
            _arrow(ax, (hx - 8, hy + 6), (hx + 55, hy - 42), HOT_C, lw=2.4)
            ax.text(hx, hy + hr + 12, label, ha="center", fontsize=10, color=HOT_C,
                    fontweight="bold", zorder=6)
            ax.text(hx, hy + hr - 8, sub, ha="center", fontsize=8.2, color=HOT_C, zorder=6)

    rng = np.random.default_rng(11)
    vehicles = []
    for _ in range(21):
        vx = rng.uniform(60, 940)
        vy = rng.choice([230, 420, 610, 800]) + rng.normal(0, 12)
        vehicles.append((vx, vy))
        ax.add_patch(mpatches.Rectangle((vx - 7, vy - 5), 14, 10, facecolor="#455A64",
                                        edgecolor="white", lw=0.7, zorder=4))
    ax.text(30, 615, "Vehicles with tasks\n(load-balanced association,\n~U/K users per UAV)",
            fontsize=8.6, color="#455A64", zorder=6)

    # 3 UAVs
    uavs = [(330, 300), (520, 560), (700, 350)]
    for (ux, uy) in uavs:
        _uav(ax, ux, uy)
    ax.text(330, 262, "UAV 1", ha="center", fontsize=8.8, color=UAV_C, fontweight="bold", zorder=6)
    ax.text(520, 522, "UAV 2", ha="center", fontsize=8.8, color=UAV_C, fontweight="bold", zorder=6)
    ax.text(700, 312, "UAV 3", ha="center", fontsize=8.8, color=UAV_C, fontweight="bold", zorder=6)
    ax.text(520, 600, "UAV edge servers\n(50-60 GHz, shared compute queues)",
            ha="center", fontsize=8.4, color=UAV_C, zorder=6)

    # user -> UAV access links (sample)
    for (vx, vy), (ux, uy) in zip(vehicles[:12], [uavs[0]] * 4 + [uavs[1]] * 4 + [uavs[2]] * 4):
        _arrow(ax, (vx + 8, vy + 6), (ux - 18, uy - 10), UAV_C, lw=1.0)
    ax.text(300, 150, "User-UAV access links (1.0-1.1 MHz)\nassociation: best-rate + load balance",
            fontsize=8.4, color=UAV_C, zorder=6)

    # LEO constellation
    ax.add_patch(Arc((500, 880), 980, 330, theta1=180, theta2=360, ec="#E0C46B", lw=2.0, zorder=1))
    for lx, ly in ((150, 900), (500, 940), (850, 900)):
        ax.add_patch(Circle((lx, ly), 13, facecolor=LEO_C, edgecolor=LEO_EDGE, lw=1.3, zorder=5))
        _arrow(ax, (lx + 24, ly + 15), (lx - 24, ly + 15), LEO_EDGE, lw=1.4)
    ax.text(500, 975, "LEO constellation (4 sats, 1 THz edge compute)", ha="center", fontsize=9.5,
            color=LEO_EDGE, fontweight="bold", zorder=6)

    for (ux, uy) in uavs[:2]:
        _arrow(ax, (ux + 20, uy + 16), (478, 926), LEO_C, style=(0, (4, 2)), lw=1.6)
    ax.text(545, 640, "UAV-LEO backhaul (6.5-7 Mbps)", fontsize=8.4, color=LEO_EDGE, zorder=6)

    ax.text(30, 90, "Local compute\n(user CPU 5 GHz)", fontsize=8.4, color=LOCAL_C, zorder=6)

    for i, (c, label) in enumerate([
            (LOCAL_C, "Local computation"),
            (UAV_C, "UAV edge offload (K UAVs)"),
            (LEO_C, "LEO edge offload")]):
        y = 60 - i * 34
        ax.add_patch(mpatches.Rectangle((760, y - 8), 16, 16, facecolor=c, edgecolor="white",
                                        lw=0.8, zorder=6))
        ax.text(786, y, label, fontsize=9, va="center", color=TEXT_C, zorder=6)

    save(fig, "architecture_multi_uav.png")


def pipeline_m():
    """PMEO-M-Eco per-slot pipeline: association -> candidate trajectory -> post-move exact offload."""
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, face, edge, text, fs=8.0, tc=TEXT_C):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, lw=1.4, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=4)

    def flow(x1, x2, y):
        _arrow(ax, (x1, y), (x2, y), GREY, lw=1.8)

    ax.text(59, 43.2, "PMEO-M-Eco per-slot decision pipeline  (multi-UAV, training-free, energy-priced)",
            ha="center", fontsize=10.5, color=TEXT_C, fontweight="bold")

    # Stage 0: observe
    box(0.6, 27, 21, 12, "#E8EAF6", "#3F51B5",
        "Observe at slot t\nuser tasks & positions,\nK hotspots, K UAVs,\nbacklogs", fs=8.0)
    flow(22.1, 26.1, 33)

    # Stage 1: association
    box(26.6, 27, 22, 12, "#E3F2FD", UAV_C,
        "Step 1: Load-balanced\nassociation\nbest-rate user->UAV,\n~U/K users per UAV", fs=7.8)
    flow(49.1, 53.1, 33)

    # Stage 2: candidate trajectory
    box(53.6, 27, 27, 12, "#FFF8E1", HOT_C,
        "Step 2: Energy-priced candidate\nmove selection (H=3 lookahead)\n"
        "{hover, expert, centroid, load-\nbalanced centroid, near-LEO}\n"
        "min latency+energy+drop cost", fs=7.4)
    flow(81.1, 85.1, 33)

    # Stage 3: post-move exact offload
    box(85.6, 27, 19, 12, "#FCE4EC", "#AD1457",
        "Step 3: Post-move exact\noffload per user\nargmin {local, UAV_1..K,\nLEO} x ratio grid", fs=7.6)
    flow(105.1, 108.6, 33)

    box(109.1, 27, 8.6, 12, "#E8F5E9", LOCAL_C, "Execute\nmove +\noffload", fs=7.8)

    # per-user cost grid: header row (ratios) + (K+2) target rows
    gx, gy, gw, gh = 66, 2.2, 34, 20.6
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="white", edgecolor="#9e9e9e", lw=1.1, zorder=2))
    ax.text(gx + gw / 2, gy + gh - 1.3, "Per-user cost grid at post-move positions",
            ha="center", fontsize=7.6, color=TEXT_C, zorder=4)
    rows = ["local", "UAV 1", "UAV 2", "UAV 3", "LEO"]
    cols = ["0", ".25", ".5", ".75", "1"]
    cell_w = gw / 5.0
    cell_h = (gh - 3.0) / 6.0
    hdr_y = gy + 3.0 + 5 * cell_h
    for j, col in enumerate(cols):
        cx = gx + j * cell_w
        ax.add_patch(mpatches.Rectangle((cx, hdr_y), cell_w, cell_h, facecolor="#ECEFF1",
                                        edgecolor="#90A4AE", lw=0.9, zorder=3))
        ax.text(cx + cell_w / 2, hdr_y + cell_h / 2, col, ha="center", va="center",
                fontsize=6.6, color="#546E7A", zorder=4)
    ax.text(gx - 0.4, hdr_y + cell_h / 2, "ratio", ha="right", va="center", fontsize=6.8,
            color="#546E7A", zorder=4)
    for i, row in enumerate(rows):
        for j in range(5):
            cx = gx + j * cell_w
            cy = gy + 3.0 + i * cell_h
            ax.add_patch(mpatches.Rectangle((cx, cy), cell_w, cell_h,
                                            facecolor=("#F9FBE7" if i == 2 else "none"),
                                            edgecolor="#90A4AE", lw=0.9, zorder=3))
        ax.text(gx - 0.4, cy + cell_h / 2, row, ha="right", va="center", fontsize=6.8,
                color=TEXT_C, zorder=4)
    _arrow(ax, (95, 27), (95, 23.9), GREY, style=(0, (4, 2)), lw=1.3)
    ax.text(95.6, 25.5, "argmin", fontsize=7.0, color=GREY, va="center", zorder=4)

    ax.text(59, 1.6,
            "Complexity: association O(U K) + candidate moves O(H x |C| x U x (K+2) x 5) simulations per slot;  "
            "offload argmin is per-slot optimal given post-move positions",
            ha="center", fontsize=7.8, color=GREY)

    save(fig, "pmeo_m_pipeline.png")


if __name__ == "__main__":
    architecture_multi_uav()
    pipeline_m()
