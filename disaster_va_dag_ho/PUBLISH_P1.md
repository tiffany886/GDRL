# PUBLISH_P1.md — v2 投稿加固：P0 → P1-lite 记录（2026-09-07）

> 目标：把 v2 验收从「调参凑出」升级为「敏感性成立区间 + 配对显著 + 可辩护」。
> 冻结基线：`configs/main_v2.yaml` 未改（`git diff --stat configs/main_v2.yaml` 为空）。
> 记账：`results/run_log.csv`（argv/git_rev/cfg_sha/wall）。判据：No-Go = 中心格
> Ours−hover 或 Ours−B4 不显著，或 B4 加预算后追上 embed。

## P0 冻结（done）
- 记账 `scripts/run_log.py` → `results/run_log.csv`；一键 `scripts/p1_all.sh <scope>`。
- 敏感性 runner `scripts/run_sens_v2.py`：行级带 `cell/R/lambda_high/scene_seed`，
  B4 按 train_seed=100+s 训练、同 scene seed s 评估（可配对）。
- 统计 `scripts/p1_stats.py`：配对单边 Wilcoxon；有效时延
  `eff_lat=(done_delay_sum + apps_failed×deadline_s)/apps_arrived`（deadline_s=12）。
- 校验：center-smoke 与冻结主表逐位一致（Ours 0.710526… / hover 0.568807…）。
- pytest 45 passed（每轮 p1_all 门禁复核）。

## 敏感性 6 格（grid-lite，seeds 0–4）+ 中心格 10 seeds

| cell (R-λ_high) | Ours succ | hover | B4-flat | Ours−hover | Ours−B4 | frac≈ |
|---|---|---|---|---|---|---|
| R600-l0.1 | 0.556 | 0.466 | 0.477 | +0.091 (p=0.0625, n5) | +0.079 (p=0.0625) | 0.45 |
| R600-l0.2 | 0.491 | 0.405 | 0.431 | +0.087 (p=0.0625) | +0.061 (p=0.0312) | 0.45 |
| R800-l0.1 | 0.723 | 0.667 | 0.676 | +0.055 (p=0.0625) | +0.047 (p=0.2188) | 0.245 |
| **R800-l0.2 中心** | **0.731** | 0.646 | 0.598 | **+0.086 (p=0.0312, n5)** | **+0.133 (p=0.0312)** | 0.246 |
| R900-l0.1 | 0.812 | 0.796 | 0.693 | +0.017 (ns) | +0.120 (p=0.0312) | 0.17 |
| R900-l0.2 | 0.813 | 0.789 | 0.676 | +0.023 (ns) | +0.137 (p=0.0312) | 0.17 |

**中心格 10 seeds（c10，R800-l0.2）**
- Ours 0.736±0.059 | hover 0.641±0.075 | chase 0.729±0.079 | B4-flat 0.606±0.088
- Ours−hover **+0.0949, p=0.0010**（n=10）；Ours−B4 **+0.1306, p=0.0010**
- eff_lat：Ours 3.71 s < hover 4.17 s < B4 5.36 s

**成立区间（结论）**
- 硬覆盖生效区 frac_unassociated ∈ [0.15, 0.45] ≈ R ∈ [600, 800]：Ours−hover ≥ +0.06
  （n5 下 p=0.03–0.06，n10 中心格显著 p=0.001）；R=900（frac≈0.15）覆盖太弱，
  hover 追平 → 场景退化向 v1 软连接，正好佐证「硬覆盖是前提」。
- Ours−B4 ≥ +0.12 在 R800-l0.2 与 R900 显著；R600 两格 +0.06–0.08（n5 下部分 p=0.06）。
- **主表工作点 R=800 / λ_high=0.2（frac≈0.245）是「中区」而非端点**：不是挑出来的
  最大值，写正文时按此表述。
- 已知缺口：R800-l0.1 格 Ours−B4 差仅 +0.047 不显著（低热点到达下 flat RL 表现较好）
  ——按「救援突发=高热点到达」叙事说明成立场景；如审稿压力大可再扩 10 seeds 复核。

## 统计与 B4 预算
- B6−Ours 时延（det-10，n=10）：**+0.172 s，p=0.0137**；Ours−B6 成功率差 +0.0013
  （ns）→ 正文口径 =「相近成功率下时延显著更低」。
- B4 加预算：中心格 250ep→500ep 成功率 0.598→0.597（配对差 mean −0.001，p=0.81）：
  **加预算无质变**，flat RL 已到平台（论文引用收敛论证即可；曲线图待补）。

## chase 叙事（红队关键，W4 决策）
- 中心格 10 seeds：chase 成功率 0.729 ≈ Ours 0.736（双侧 p=1.0）；时延 +0.08 s ns；
  **能耗 Ours 比 chase 高 2130 J（p=0.0098）**——chase 是更强的任务感知轨迹。
- chase-mover 下 embed-vs-flat（5 seeds 配对）：chase+embed（=B2）0.745 vs
  chase+B4-flat 0.571 → **+0.174（p=0.0312）**，时延 −1.8 s、eff_lat −2.1 s
  （p=0.0312）。与 patrol 下 +0.16 一致 → **嵌入优势不挑轨迹**。
- **决策（方案 a）**：正文承认 chase 是高信息需求强启发式（实时积压质心），主方法默认
  T-patrol 的理由 = 巡逻不需任务分布先验、贴合灾后侦察/搜索流程；同时报告
  chase-mover 对照证明 embed≫flat 对轨迹鲁棒。不用 chase 成功率/能耗主张「巡逻更优」。
  若导师要求更强轨迹主张，再讨论主方法改用 chase（走方案 b），本轮不改表。

## R≈800 物理话术（草稿，论文阶段再核引文）
- 灾后 UAV-BS 文献普遍以「圆盘覆盖」建模 UAV 覆盖（如 UAV-aided emergency networks,
  EURASIP JWCN 2018；UAV-BS disk-cover deployment 2024），覆盖半径取数百米量级。
- 本文 2×2 km 场景 R=800 m 对应中等覆盖密度（~25% 终端时隙失联），且敏感性证明
  R∈[600,900]、λ_high∈[0.1,0.2] 内方向一致、R=800 是区间内工作点而非唯一解。

## W4：w/o DAG-order「假成功」量化（本轮补完）

改动（不改变任何调度语义，只加计数与指标）：
- `scheduler/schedule_dag.py`：`_commit` 里若「孩子节点在其前驱 done 前被提交」则
  `dep_violations += 1` 并把 `app.violated=True`；`StepStats.dep_violations` 给每步差值。
- `env/env.py` metrics 增 `dep_violations`；`_stats`/trainer/run_sens 键同步；
  原 `dag_order=True` 的合法调度路径不受影响（`_ready_nodes` 门控保证前驱先 done）。
- 测试改为断言该语义：no-DAG-order 单链上孩子先于父节点被提交 → 计 1 次违例；
  随机 episode 计数 >0；合法行保持 0。单测 45 passed。

重跑 `ablation-v2`（5 seeds，`results/ablation_v2_{raw,summary}.csv`）：

| 消融行 | dep_violations/集 | apps_done/集 | 违例/完成 app | 成功率 |
|--------|------------------|--------------|---------------|--------|
| w/o DAG-order | **276.2 ± 10.0** | 84.6 ± 3.1 | **3.27 ± 0.12** | 0.8182±0.0433 |
| w/o VW-prune（=B6） | 0（对照） | 72.0 ± 7.3 | 0 | 0.7436±0.0325 |
| w/o hard-cover | 0（对照） | 94.0 ± 8.1 | 0 | 0.8748±0.0246 |
| w/o embed（=B4） | 0（对照） | 53.3 ± 4.2 | 0 | 0.5705±0.0437 |

**红线答复（审稿口径）：** w/o DAG-order 的 0.818 成功率把依赖链当独立包：每个
非源节点的 `in_bits`（=前驱 `out_bits`）在其前驱计算完成**之前**即被提交执行，
全 episode 平均 276 次、约 **3.3 次/每个完成 app** 属「输入尚不存在就开始算」的
物理不可能完成——0.818 是拆掉拓扑约束后的乐观上界，不是真实系统可达到的完成率。
合法调度（B1–B6、Ours）`dep_violations == 0`（REPRODUCE §7.4 已有断言）。正文写法：
「保留 DAG-order 时依赖满足与完成语义合法；去掉后每完成 app 平均引入 3.3 次
依赖未就绪的提交，成功率虚高 +0.087（0.818 vs 0.731），不作比较结论。」

## 未完成（下一轮）
- B4 收敛曲线图（train_curves 记录）与敏感性/时延 CDF 机制图（P2 精简版）。
- 主表 REPRODUCE §7 增补引用本文件；P4 红队剩余条逐条归档。
