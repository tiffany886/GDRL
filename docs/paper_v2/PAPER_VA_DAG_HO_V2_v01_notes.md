# VA-DAG-HO v2 — Paper Draft (v0.1, 2026-09-07)

> 语言：正文 English（IEEE 风格，两栏草稿）；中文批注/待办在 `>` 与 `<!-- -->` 中。
> 数据冻结：`configs/main_v2.yaml`（R=800, λ_high=0.2, N=40, K=3, L=2, T=50）。
> 图表：`disaster_va_dag_ho/results/figures_v2/*.png`；全部数字来自 `results/main_table_v2_summary.csv`、`results/ablation_v2_summary.csv`、`results/v2_p1_stats_report.txt`。
> 目标：IoT-J / 中等会议（Go 低配）；TVT/TWC 需补 P3 规模（见 §VIII）。
> LaTeX 三表：`docs/paper_v2/tables_v2.tex`（表自动从冻结 CSV 导出）。

---

## Title (draft options)
1. **Dependency-Aware Task Embedding under UAV Hard Coverage and Intermittent LEO Visibility for Post-Disaster Offloading**
2. Coverage- and Visibility-Constrained DAG Scheduling over UAV–LEO for Post-Disaster Rescue Applications

<!-- 选 1；关键词里同时给 offloading 与 UAV-LEO 以便检索 -->

## Abstract (draft)

In post-disaster areas, terrestrial infrastructure is unavailable, so a few UAVs must cover scattered terminals and relay their computation-intensive tasks to low-Earth-orbit (LEO) satellites that are only intermittently visible. We study how to place the nodes of dependency-constrained (DAG) rescue applications on {UAV, relay-LEO} when a terminal can associate only while it lies inside a UAV's hard coverage disk. We do not learn trajectories. Instead, a fixed strip patrol keeps the UAVs over the rescue hotspots, a coverage-aware greedy association is allowed to leave terminals unassociated, and a schedule-embedded heuristic (ScheduleDAG) places DAG nodes in topological order while a decision-time visibility-window prune forbids LEO placements that cannot finish inside one contiguous window. On a 40-terminal, 3-UAV, 2-LEO simulator with spatially inhomogeneous arrivals, the proposed scheme keeps ~25% of terminal-slots unassociated, reaches a 0.731 success rate with an effective latency of 3.79 s, and beats a flat RL offloading policy by +0.13 in success (p<0.01, paired, 10 seeds) at a 36% lower effective latency; disabling the window prune preserves the success rate but significantly increases delay (+0.17 s, p=0.014). An ablation that treats dependent tasks as independent packages inflates the success rate to 0.818 while committing every completed application on average 3.3 times before its inputs exist, confirming that the DAG order is what makes the reported completions physically meaningful.

*Index Terms* — post-disaster offloading; UAV; LEO satellite; DAG scheduling; visible window; dependency-aware scheduling; MAPPO.

<!-- 数字口径都来自 p1_stats_report / main_table_v2。投稿前再压到 ~200 words。 -->

## I. Introduction

**背景与动机（首段模板，含文献位）**
After an earthquake or flood, the access backbone is often destroyed while terminals cluster at rescue points [REF: UAV-aided emergency networks, EURASIP JWCN 2018]. A few UAVs can be dispatched as aerial base stations [REF: UAV-BS deployment survey]; however, both their coverage and their energy are limited, and the backhaul to a ground core network may not exist. Low-Earth-orbit satellites provide a survivable backhaul, but a LEO appears only in short, periodic visibility windows [REF]. Tasks spawned by rescue applications are not independent: a "detect-fuse-decide" pipeline on one request is a directed acyclic graph whose nodes must execute in topological order. This paper asks a scheduling question: with no terrestrial core, no charging, fixed (not learned) UAV motion, and hard coverage disks, how should the nodes of arriving DAG applications be placed on the UAVs and the relay LEOs?

**问题陈述（三个难点）**
1. *Hard coverage* — a terminal can be served only when its horizontal distance to some UAV is ≤ R. Outside R there is no access at all, so a subset of terminals is periodically unreachable (`assignment = -1`). Prior UAV–LEO works [REF; REF] usually keep every terminal "connectable" with a small rate; that avoids the problem we want to face: some work physically cannot be uploaded while it is not covered.
2. *Intermittent visibility* — relay + service of a node must fit inside a single contiguous LEO window; outside a window the capacity is zero (hard constraint). Optimizing as if LEO were always-on commits work that can never complete (cross-window failures).
3. *Dependencies* — treating DAG nodes as independent packages raises the success rate (0.818 vs 0.731 in our data) but the extra "successes" start before their inputs exist; they are not valid completions in a real system.

**本文方案（贡献三条，与 SPEC §0 对齐）**
- **C1 (embedding scheduling).** We keep the trajectory *rule-based* (a strip patrol designed to sweep the hotspot bands) and embed the DAG scheduling into the simulator loop: coverage-aware association, topological-order ready set, {UAV, relay-LEO, wait} placement with deadline margin and per-slot capacity.
- **C2 (visible-window prune).** At decision time we only place a LEO node if its relay and service interval fit into one predicted contiguous window; a cross-window placement is forbidden instead of being committed and doomed. The prune keeps success comparable (0.744 vs 0.731, ns) while cutting delay significantly (p=0.014, 10 seeds).
- **C3 (structure over flat RL).** Under the same fixed patrol trajectory, embedding scheduling beats an end-to-end flat RL that learns only the offload decisions (B4 MAPPO-flat / B5 SAC-flat): success +0.13..0.16 (p<0.01) and effective latency reduced by ~36%, and the advantage is trajectory-robust (also +0.17 under a chase mover). We deliberately do *not* claim learned trajectory as a contribution.

**与已有工作的不同（每段 1–2 句 + REF）**
- 学轨迹的 MARL 卸载 [REF]：本文轨迹固定，把「学习」放在卸载决策；对比扁平 RL 只学卸载。
- 常在线 LEO 假设 [REF]：本文窗外容量=0，且决策期窗剪枝。
- 软连接全关联的 UAV 覆盖 [REF]：本文硬覆盖，出现真实连不上（frac_unassociated ≈ 0.25）。
- 任务独立卸载 [REF]：本文 DAG 拓扑序；消融证明独立包处理是乐观虚高。

## II. System Model and Problem Formulation

### A. Area, entities, arrivals
- 2×2 km square; `N=40` static/slow terminals (v_max=2 m/s, 50% static), `K=3` UAVs at altitude 100 m with v_max=20 m/s, `L=2` LEOs at 550 km; slot 1 s; horizon T=50.
- Arrivals: spatial inhomogeneous Poisson; two hotspots (radius 300 m at [500,150] and [1500,1850]) with λ_high=0.2 req/s inside and λ_low=0.03 outside; terminal concurrency ≤ 2; deadline 12 s. <!-- 数字与 main_v2.yaml 一致 -->

### B. Coverage and association
UAV k covers terminal n at time t iff horizontal distance ||x_n − p_k|| ≤ R (R=800 m in the frozen config). Let a(t) ∈ {-1} ∪ {0..K-1} be the greedy assignment: sort (n,k) pairs with r_nk>0 by rate, enforce at most ceil(N/K) terminals per UAV, and allow a(t)=-1 for uncovered terminals. Metrics: frac_unassociated and fail_uncovered (apps that could not progress because their terminal was out of coverage and missed the deadline). A pending app whose terminal is unassociated cannot start or push new nodes in that slot.

### C. DAG tasks
Each request is a DAG sampled from three templates (chain-light, fork-join, chain-heavy; weights 0.45/0.30/0.25), with node i needing in_bits=sum of parent out_bits and cycles (scaled 6.0x). Node must run after its parents finish (topological order).

### D. UAV–LEO channel and visibility
Terminal-UAV and UAV-LEO rates from 3GPP-style LoS/NLoS model [REF]; UAV-LEO usable rate gated by visibility and capped at C_l=10 Mb/s; outside the window, capacity is 0. Window schedule: LEO l visible in periodic windows (period 22 s, duty 0.4, phases 0/11 s so windows interleave). <!-- 参数表：Table I 放 main_v2.yaml 关键量 -->

### E. Energy
UAV SoC with hover power 80 W, speed-cubed drag term (drag_coeff 0.004, speed³), compute power 150 W, comm power 20 W; no charging; low-SoC penalty. Flight and comm/compute energy are charged separately.

### F. Objective
Maximize completed apps within deadlines under coverage and window constraints while keeping delay and energy low:
- primary: success rate (done/arrived);
- secondary: conditional completion delay (successful samples) and effective latency `eff_lat = (Σ done_delay + apps_failed·deadline)/arrived` — we report eff_lat because conditional delay alone is biased by hard tasks failing on weak methods (selection bias, §VII).
<!-- 公式用 LaTeX 在正式稿里展开；这里先给口径 -->

## III. Proposed Scheme: VA-DAG-HO

### A. Fixed patrol (T-patrol)
Each UAV owns a corridor strip; it patrols back and forth inside it (period ~40 s). The strips are placed to sweep the two hotspot bands, so no task-position prior is needed online. Alternatives used only as baselines: hover (B3), backlog-chase (B2), random (B1). <!-- 巡逻条带几何在图里画 -->

### B. ScheduleDAG (decision-embedded heuristic)
Per slot, given the new UAV positions and the current visibility:
1. **Association** (coverage-aware greedy, §II-B).
2. **Ready set**: node i of app a is ready if placed-none and, under DAG-order, all parents done.
3. **Candidate ordering** (urgency key): (deadline-margin class, deadline, heavier-first): `(rem ≤ deadline_margin ? 0 : (heavy ? 1 : 2), deadline_s, -cycles)`.
4. **Placement** on {UAV_k, LEO_l} or wait: UAV when uplink fits and deadline feasible; LEO only if *the relay interval plus the service interval fits inside one contiguous visibility window of l*, and capacity C_l free; otherwise wait or reject.
5. **Bookkeeping**: busy times, per-node finish; app completes when its sink finishes before deadline.

> 形式化伪码 Alg.1 放在正式稿；这里先给算法结构与「被验证的因果链」。

### C. Visible-window prune (VW-prune) vs no-prune (B6)
The prune forbids placing a LEO node when no single window can contain relay+service. The no-prune variant (B6) decides with an always-on nominal schedule and only discovers at execution that the interval crosses an outage — those commits are *doomed*, the app fails at deadline, and the attempt is counted as invalid (decision-time block Ours: 745 invalid attempts vs B6: 264 doomed releases). Delay consequence: B6's completed tasks are delayed by +0.17 s (p=0.014, paired) — the prune removes doomed placements that otherwise jam the LEO busy queue.

### D. Baselines
| ID | description | role |
|---|---|---|
| B1-random | random walk + same scheduler | weak trajectory |
| B2-chase | backlog-center chase + same scheduler | strong trajectory prior |
| B3-hover | hover at fixed uniform positions | weak coverage |
| **Ours** | **T-patrol + hard-cover association + ScheduleDAG (VW-prune + DAG-order)** | proposed |
| B4-flat | T-patrol + MAPPO (flat) learns offload only | flat RL |
| B5-SACflat | T-patrol + SAC-flat learns offload only (off-policy) | flat RL |
| B6-noVWprune | T-patrol + scheduler without decision-time window prune | ablation C2 |
Ablations: w/o VW-prune (=B6), w/o DAG-order (independent packages), w/o hard-cover (soft-connection v1), w/o embed (=B4).

## IV. Experimental Setup

- Environment/simulator `disaster_va_dag_ho` (pytest suite 45 tests, all green); physics reused from the shared `uav_leo_experiment/physics.py`.
- Frozen config Table I (main_v2.yaml). Grid: R∈{600,800,900} × λ_high∈{0.1,0.2} for sensitivity; center (R=800, λ_high=0.2) for statistics.
- Seeds: methods deterministic w.r.t. scene seed; RL baselines train on train_seed=100+s and are evaluated on scene_seed=s to enable paired tests.
- Training budget: B4 250 ep/seed (~53–61 s), B5 SAC-flat 400 ep (~144 s/seed); B4 additionally run at 500 ep as the budget check.
- Statistic: one-sided paired Wilcoxon signed-rank on scene-seed pairs; n=10 at the center cell for the headline claims.
<!-- 复现命令：REPRODUCE.md v2 章；p1_all.sh <scope> -->

## V. Results

### A. Main table (R=800, λ_high=0.2, 5 seeds)

| method | success | cond. delay (s) | eff. lat. (s) | UAV energy (J) | invalid LEO | frac unassoc. |
|---|---|---|---|---|---|---|
| B1-random | 0.680±0.060 | 1.50±0.25 | 4.14 | 16.46k±0.68k | 486±267 | 0.245±0.062 |
| B2-chase | 0.745±0.042 | 1.90±0.24 | 3.84 | 19.08k±0.25k | 571±302 | 0.229±0.066 |
| B3-hover | 0.646±0.056 | 1.47±0.15 | 4.28 | 15.33k±0.45k | 468±277 | 0.245±0.062 |
| **Ours** | **0.731±0.060** | 1.89±0.24 | **3.79** | 21.24k±0.92k | 745±414 | 0.246±0.042 |
| B6-noVWprune | 0.744±0.033 | **2.08±0.09** | 4.07 | 22.28k±0.65k | 264±109 | 0.246±0.042 |
| B4-flat | 0.571±0.044 | 3.54±0.15 | 5.84 | 24.66k±2.09k | 160±59 | 0.251±0.003 |
| B5-SACflat | 0.597±0.024 | 3.11±0.24 | 5.43 | 22.31k±2.58k | 216±46 | 0.255±0.000 |

Reads: (i) Ours ≈ B2-chase in success (0.731 vs 0.745, ns at n=10) with comparable effective latency — embedding is *not* buying success over a strong task-aware mover; the patrol is chosen because it needs no task prior (§VII). (ii) Flat RL offload-only is substantially worse in success (−0.13..−0.16) and effective latency (+~2 s). (iii) Ours energy is 5.9 kJ higher than hover (movement + serving heavy apps) but lower than B4/B5/B6. <!-- 数字对齐 Table II -->

<!-- figure: fig_v2_main.png -->

### B. Statistical significance and budget (center cell)

| claim | Δ | n | Wilcoxon p |
|---|---|---|---|
| Ours − hover success | +0.0949 | 10 | 0.0010 |
| Ours − B4 success | +0.1306 | 10 | 0.0010 |
| Ours − chase success | +0.0069 | 10 | 1.000 (ns) |
| Ours − chase energy | −2130 J | 10 | 0.0098 (Ours higher) |
| B6 − Ours cond. delay | +0.172 s | 10 | 0.0137 |
| B4 250→500 ep success | −0.001 | 5 | 0.81 (ns) |

Budget conclusion: B4 does not improve with more training (0.598→0.597) — the gap vs embedding is structural, not an under-trained artifact. <!-- fig_v2_budget.png -->

### C. Sensitivity ("region of validity", not tuning)

Gap Ours−hover / Ours−B4 across R×λ_high (5 seeds):
- R=600: Ours−hover +0.09 (p≈0.06), Ours−B4 +0.06–0.08 (p=0.03 at λ0.2)
- **R=800 (center)**: Ours−hover +0.086 (p=0.03 n=5 / p=0.001 n=10), Ours−B4 +0.13
- R=900: Ours−hover +0.02 (ns, hover catches up because coverage is easy → the problem degrades to v1 soft-connection), Ours−B4 +0.12–0.14
- Known weak cell: R=800, λ_high=0.1 — Ours−B4 +0.047 (ns). Interpretation: at low hotspot arrival the flat RL's load is light enough to close the gap; our rescue scenario is the high-λ burst case. If a reviewer presses, we can add 10 seeds at this cell (≈9 min).

The frozen operating point R=800/λ=0.2 is a mid-region point of frac_unassociated ≈ 0.25, not an endpoint picked for the best number. <!-- fig_v2_sensitivity.png -->

### D. Ablations (5 seeds)

| ablation | success | cond. delay (s) | eff. lat. (s) | energy (J) | invalid LEO | dep violations |
|---|---|---|---|---|---|---|
| Ours (ref.) | 0.731 | 1.89 | 3.79 | 21238 | 745 | 0 |
| w/o VW-prune (=B6) | 0.744 | 2.08 | 4.07 | 22276 | 264 | 0 |
| w/o DAG-order | **0.818** | 1.58 | 3.00 | 24133 | 341 | **276.2±10.0** |
| w/o hard-cover | 0.875 | 2.14 | 2.86 | 25108 | 1208 | 0 |
| w/o embed (=B4) | 0.571 | 3.54 | 5.84 | 24665 | 160 | 0 |

Key claims:
1. *DAG-order matters for validity, not for headline numbers.* w/o DAG-order "completes" apps by starting children before their parents' outputs exist: 276.2±10.0 dependency violations per episode ≈ 3.3 per completed app. A real system would compute garbage (a downstream node receives no input). 0.818 is an upper bound of independent-package processing, not a comparable success rate. Legal schedules keep the counter at zero.
2. *Hard coverage is what makes the problem hard.* Removing it (soft connection) raises success to 0.875 and drops frac_unassociated to 0 — coverage never binds anymore, which is exactly v1's regime; the interesting regime is the one in the main table (unassociated 0.25).
3. *VW-prune is a delay/quality mechanism.* Same success, +0.17 s delay, and higher energy when disabled.
<!-- fig_v2_ablation.png, fig_v2_depviol.png, fig_v2_delay_cdf.png, fig_v2_frac_unassoc_ts.png, fig_v2_hotspot_success.png, fig_v2_coverage_snapshot.png -->

### E. Mechanism evidence (replayed deterministic episodes, seeds 0-4 pooled)

1. *Where the difficulty is: the hotspots.* Split arrivals by whether the terminal
   lies inside a hotspot (radius 300 m, centres at the two rescue bands). Success
   inside hotspots is 0.515 (hover) vs **0.760 (Ours)** with 2.18 s delay; outside,
   hover already reaches 0.732 and Ours 0.704. The patrol strips sweep the hotspot
   bands, which is exactly what hover (fixed uniform positions) cannot do.
   <!-- fig_v2_hotspot_success.png -->
2. *Coverage over time.* frac_unassociated stays around 0.25 across the whole
   episode for every mover (mean ± std over seeds) - disconnection is structural,
   not a transient; hover and patrol differ little here.
   <!-- fig_v2_frac_unassoc_ts.png -->
3. *Selection bias made visible.* The CDF of successful-app delays shows hover's
   mass at shorter delays; its eff_lat is worse because ~35% of its tasks fail
   (they would have been the long ones). Success must be read together with
   eff_lat, never with conditional delay alone.
   <!-- fig_v2_delay_cdf.png -->
4. *Coverage snapshot.* Figure fig_v2_coverage_snapshot.png shows the three UAVs on
   their patrol strips with the 800 m disks and the two hotspot bands, and ~25% of
   terminals outside every disk (x markers) at a mid-episode slot (seed 2).
   <!-- fig_v2_coverage_snapshot.png -->

## VI. Discussion: Chase, Coverage Radius, and Assumptions

**Why chase achieves ≈ Ours success.** Chase follows the live backlog centroid; it is a strong, task-aware trajectory that needs instantaneous load information, whereas T-patrol needs no task prior and fits search-and-reconnaissance workflows. We do not claim patrol is better than chase; we report both. Under a *chase* mover, embed-vs-flat is +0.174 success and −2.1 s effective latency (paired, p=0.031), matching patrol's +0.16 — the scheduling contribution is trajectory-robust.

**Coverage radius.** Disk-coverage UAV models with hundreds-of-meters radii are standard for post-disaster UAV-BS scenarios [REF]; R=800 m over a 2 km area leaves ~25% of terminal-slots uncovered and is a mid-region working point (Fig. sensitivity), and the direction of all conclusions is stable across R∈[600,900].

**Assumptions that bound the claims.** (i) no direct terminal–LEO link and no cloud/edge outside the disaster cell: we model the typical surviving backhaul (relay via UAV) [REF]; direct access is future work (Appendix). (ii) no charging, energy budget 50 kJ/UAV. (iii) known-ish deterministic window schedule (ephemeris-predictable); we use the same predicted windows in the prune. (iv) a single-cell 2×2 km scenario; scale to N=60 or multiple cells is left to future work (P3 gate).

## VII. Threat-to-validity notes (for authors; removed in camera-ready)

1. Conditional delay is biased: hover's 1.47 s vs Ours 1.89 s is not a win — hover simply fails the long tasks. Report success × eff_lat jointly; we do.
2. All RL comparisons pair the same scene seeds; training seeds are offset (100+s) to avoid accidental scene leakage.
3. The 0.744 success of B6 vs 0.731 of Ours is ns; never phrase the prune as "improves success".
4. w/o DAG-order success is not comparable (see §V-D). The reproducibility ledger (run_log.csv) fixes config hashes for every run.

## VIII. Conclusion and Next Steps

We presented VA-DAG-HO, a fixed-patrol, hard-coverage, visibility-window-aware DAG scheduler for post-disaster UAV–LEO offloading. With 40 terminals, 3 UAVs and 2 intermittent LEOs, embedding the scheduling beats flat RL offloading by +0.13 success (p<0.01, paired) and ~36% effective latency, and decision-time window pruning yields a significant delay gain without hurting success. Ablation evidence shows both hard coverage (frac_unassociated ≈0.25) and DAG-order enforcement (0 violations under legal scheduling) are what make these numbers physically meaningful.

<!-- 下一阶段（对应 P2/P3 闸门）：
- [done] P2 精简机制图 4 张（figures_v2/fig_v2_{delay_cdf,frac_unassoc_ts,hotspot_success,coverage_snapshot}.png；数据源 scripts/mechanism_figs.py）
- B4/B5 收敛曲线图（run_curves.py 按冻结 config 重训 1 seed）
- 补引文：UAV-BS 覆盖建模、LEO 可见窗、DAG 卸载、MARL offloading
- 若冲 TVT/TWC：N∈{30,40,60} 规模扫 + chase 显著性话术定稿 -->
