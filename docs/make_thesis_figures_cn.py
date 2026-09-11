#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制论文三张中文示意图：网络模型 / PMEO 流程 / LAETS 架构。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Arc
from matplotlib import font_manager
import numpy as np

# 中文字体
for name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Serif CJK SC"):
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
        if path and "DejaVu" not in path:
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

OUT = Path("/code/docs/GDRL/experiments/uav_leo_v2x_paper_final/figures")
OUT.mkdir(parents=True, exist_ok=True)

C_UAV = "#1565C0"
C_LEO = "#F9A825"
C_HOT = "#C62828"
C_ROAD = "#CFD8DC"
C_USER = "#37474F"
C_LOCAL = "#2E7D32"
C_DT = "#6A1B9A"
C_SYNC = "#D84315"
C_TEXT = "#212121"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)
    return p


def draw_network():
    """图1：多 UAV + 多热点网络模型。"""
    fig, ax = plt.subplots(figsize=(10, 8.2))
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1050)
    ax.set_aspect("equal")
    ax.axis("off")

    # 地面区域
    ax.add_patch(mpatches.Rectangle((40, 60), 920, 720, fill=False, ec="#90A4AE", lw=1.8, zorder=1))
    ax.text(50, 755, "地面服务区 2×2 km（道路网络）", fontsize=12, color=C_TEXT, fontweight="bold")

    # 道路
    for y in (180, 340, 500, 660):
        ax.plot([50, 950], [y, y], color=C_ROAD, lw=14, solid_capstyle="round", zorder=0)

    # 三个移动热点
    hotspots = [
        (320, 340, 95, "热点 A\n拥堵/事故"),
        (620, 500, 110, "热点 B\n主热点"),
        (780, 220, 80, "热点 C"),
    ]
    for hx, hy, hr, lab in hotspots:
        ax.add_patch(Circle((hx, hy), hr, facecolor="#FFCDD2", edgecolor=C_HOT,
                            lw=1.8, alpha=0.5, zorder=2))
        ax.annotate("", xy=(hx + 50, hy - 35), xytext=(hx - 10, hy + 10),
                    arrowprops=dict(arrowstyle="-|>", color=C_HOT, lw=2))
        ax.text(hx, hy + hr + 18, lab, ha="center", va="bottom", fontsize=10,
                color=C_HOT, fontweight="bold", zorder=6)

    ax.text(500, 100, "粉色圆：移动热点（高到达率）；圆外道路上仍有低概率任务（不全部挤在一块）",
            ha="center", fontsize=9.5, color="#546E7A")

    # 用户散点（背景稀疏 + 热点附近更密）
    rng = np.random.default_rng(7)
    users = []
    for y in (180, 340, 500, 660):
        xs = rng.uniform(80, 920, 10)
        for x in xs:
            users.append((x, y + rng.normal(0, 8)))
    # denser near hotspots
    for hx, hy, hr, _ in hotspots:
        for _ in range(8):
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, hr * 0.85)
            users.append((hx + r * np.cos(ang), hy + r * np.sin(ang)))
    for x, y in users:
        ax.plot(x, y, "s", color=C_USER, markersize=4.5, zorder=4)

    # 三架 UAV
    uavs = [(250, 820, "UAV 1"), (500, 860, "UAV 2"), (760, 820, "UAV 3")]
    for x, y, name in uavs:
        ax.add_patch(Circle((x, y), 28, facecolor=C_UAV, edgecolor="white", lw=2, zorder=5))
        ax.plot([x - 38, x - 18], [y + 22, y + 22], color=C_UAV, lw=3, zorder=4)
        ax.plot([x + 18, x + 38], [y + 22, y + 22], color=C_UAV, lw=3, zorder=4)
        ax.text(x, y - 48, name, ha="center", fontsize=11, color=C_UAV, fontweight="bold")

    # 关联示意线
    for (ux, uy, _), targets in zip(uavs, [
        [(200, 500), (280, 340)],
        [(520, 500), (620, 500), (480, 340)],
        [(780, 340), (820, 220)],
    ]):
        for tx, ty in targets:
            ax.plot([ux, tx], [uy - 28, ty], color=C_UAV, lw=1.0, alpha=0.55, zorder=3)

    # LEO
    for i, x in enumerate((280, 500, 720, 880)):
        ax.plot(x, 980, "^", color=C_LEO, markersize=16, markeredgecolor="#F57F17", zorder=5)
    ax.text(500, 1010, "LEO 星座（高算力边缘，经 UAV 回传）", ha="center", fontsize=12,
            color="#F57F17", fontweight="bold")
    for ux, uy, _ in uavs:
        ax.plot([ux, ux + 40], [uy + 28, 960], ls="--", color=C_LEO, lw=1.3, alpha=0.8)

    # 图例
    ax.add_patch(FancyBboxPatch((60, 20), 880, 55, boxstyle="round,pad=0.02",
                                fc="#FAFAFA", ec="#BDBDBD", lw=1))
    ax.plot(100, 48, "s", color=C_USER, markersize=7)
    ax.text(115, 48, "车辆用户（任务可本地/UAV/LEO）", va="center", fontsize=9.5)
    ax.add_patch(Circle((430, 48), 10, facecolor=C_UAV, ec="white"))
    ax.text(450, 48, "UAV 边缘节点", va="center", fontsize=9.5, color=C_UAV)
    ax.add_patch(Circle((640, 48), 12, facecolor="#FFCDD2", ec=C_HOT, alpha=0.7))
    ax.text(660, 48, "移动热点（可多个）", va="center", fontsize=9.5, color=C_HOT)

    ax.set_title("图：车辆—多UAV—LEO 三层网络（多热点、用户分散）", fontsize=14,
                 fontweight="bold", pad=8, color=C_TEXT)
    return save(fig, "fig_ch2_network_multi_cn.png")


def draw_pmeo():
    """图2：PMEO 两步流程。"""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="white", lw=1.5, zorder=3))
        ax.text(x + w / 2, y + h - 0.55, title, ha="center", va="top",
                fontsize=12, color="white", fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + 0.55, body, ha="center", va="bottom",
                fontsize=9.2, color="white", zorder=4, linespacing=1.35)

    box(2, 28, 22, 18, "① 观测",
        "用户位置/任务\n热点位置与速度\nUAV 位置、队列",
        "#455A64")
    box(30, 28, 26, 18, "② 启发式定轨迹",
        "目标 = 0.5×任务质心\n+ 0.5×预测热点\n受 vmax 限制一步飞过去\n（免训练规则）",
        C_UAV)
    box(62, 28, 28, 18, "③ 移动后精确卸载",
        "用新位置信道\n枚举 本地/UAV/LEO\n× 比例网格 ≈15 格\n选代价最小（免训练）",
        C_HOT)
    box(96, 28, 16, 18, "④ 执行",
        "移动\n+ 卸载",
        C_LOCAL)

    for x1, x2 in ((24, 30), (56, 62), (90, 96)):
        ax.annotate("", xy=(x2, 37), xytext=(x1, 37),
                    arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=2.2))

    # 强调顺序
    ax.add_patch(FancyBboxPatch((30, 8), 60, 14, boxstyle="round,pad=0.02",
                                fc="#FFF8E1", ec="#FFB300", lw=1.5))
    ax.text(60, 18.5, "关键：轨迹可先用规则定；卸载必须按「飞完后的新位置」计算",
            ha="center", va="center", fontsize=11, color="#E65100", fontweight="bold")
    ax.text(60, 12.2, "创新点不在枚举本身，而在决策顺序：同一轨迹下 post-move 优于 current-pos",
            ha="center", va="center", fontsize=9.5, color="#6D4C41")

    ax.text(57.5, 52, "PMEO 每时隙决策流程（训练免费 / training-free）",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)
    ax.text(57.5, 3, "代价 ≈ 时延 + we·能耗 + λ·丢弃；用户间可加 → 逐用户最优即该时隙网格最优",
            ha="center", fontsize=9, color="#607D8B")
    return save(fig, "fig_ch3_pmeo_pipeline_cn.png")


def draw_laets():
    """图3：LAETS 孪生同步架构。"""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec="white", lw=1.6, zorder=3))
        ax.text(x + w / 2, y + h - 0.7, title, ha="center", va="top",
                fontsize=11.5, color="white", fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + 0.7, body, ha="center", va="bottom",
                fontsize=9, color="white", zorder=4, linespacing=1.35)

    # 物理世界
    ax.add_patch(FancyBboxPatch((3, 8), 34, 54, boxstyle="round,pad=0.02",
                                fc="#E3F2FD", ec="#90CAF9", lw=1.5, zorder=1))
    ax.text(20, 58, "物理世界", ha="center", fontsize=13, fontweight="bold", color=C_UAV)
    box(7, 42, 26, 12, "用户 / 热点 / 任务", "真实到达与移动", C_LOCAL)
    box(7, 26, 26, 12, "UAV / LEO", "真实位置与队列", C_UAV)
    box(7, 10, 26, 12, "真实链路与能耗", "按真实状态结算", "#546E7A")

    # 数字孪生
    ax.add_patch(FancyBboxPatch((43, 8), 36, 54, boxstyle="round,pad=0.02",
                                fc="#F3E5F5", ec="#CE93D8", lw=1.5, zorder=1))
    ax.text(61, 58, "数字孪生层", ha="center", fontsize=13, fontweight="bold", color=C_DT)
    box(47, 42, 28, 12, "状态副本", "上次同步后的镜像", C_DT)
    box(47, 26, 28, 12, "状态外推预测器", "位置外推 + 任务冻结", "#8E24AA")
    box(47, 10, 28, 12, "LAETS 同步控制", "看聚合负载差\n超阈值才全量同步", C_SYNC)

    # 决策
    ax.add_patch(FancyBboxPatch((85, 18), 32, 34, boxstyle="round,pad=0.02",
                                fc="#E8F5E9", ec="#A5D6A7", lw=1.5, zorder=1))
    ax.text(101, 48, "决策模块", ha="center", fontsize=13, fontweight="bold", color=C_LOCAL)
    box(89, 28, 24, 16, "PMEO / Eco", "在孪生状态上\n做轨迹+卸载决策", "#00695C")

    # 箭头
    ax.annotate("同步 / 探测\n（事件触发）", xy=(43, 16), xytext=(37, 16),
                fontsize=9, color=C_SYNC, ha="right", va="center",
                arrowprops=dict(arrowstyle="<->", color=C_SYNC, lw=2))
    ax.annotate("用孪生状态决策", xy=(85, 36), xytext=(79, 36),
                fontsize=9, color=C_DT, ha="right", va="center",
                arrowprops=dict(arrowstyle="-|>", color=C_DT, lw=2))
    ax.annotate("动作下发执行", xy=(37, 48), xytext=(85, 48),
                fontsize=9, color=C_LOCAL, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=C_LOCAL, lw=1.8, ls="--"))

    ax.text(60, 66, "LAETS：不是第三种卸载算法，而是「何时刷新孪生」",
            ha="center", fontsize=14, fontweight="bold", color=C_TEXT)
    ax.text(60, 2.5, "因果链：孪生新鲜 → 状态准 → PMEO 决策顺序收益才成立；平静期少刷，突发起止才刷",
            ha="center", fontsize=9.5, color="#607D8B")
    return save(fig, "fig_ch4_laets_architecture_cn.png")


if __name__ == "__main__":
    paths = [draw_network(), draw_pmeo(), draw_laets()]
    print("done", len(paths))
