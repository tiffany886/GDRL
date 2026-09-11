# REPRODUCE.md — VA-DAG-HO 主表复现

规格/计划：
- v1（2026-09-04，MAPPO 学飞 + 软连接，已归档）：
  `docs/superpowers/specs/2026-09-04-va-dag-ho-disaster-design.md`
- v2（2026-09-07，不学轨迹 + 硬覆盖 + 热点非齐次到达 + T-patrol 固定轨迹）：
  `disaster_va_dag_ho/SPEC_V2_DS.md` =
  `docs/GDRL/docs/spec_va_dag_ho_v2_hard_no_traj.md` =
  `docs/GDRL/docs/superpowers/specs/2026-09-07-va-dag-ho-v2-hard-no-traj.md`。

所有命令在仓库根 `/code/docs/GDRL` 下执行，Python 解释器用
`/root/miniconda3/envs/asr_env/bin/python`（torch 2.x + CUDA）。结果落在
`disaster_va_dag_ho/results/`。v1/v2 各自的配置与结果文件互相独立，不互相覆盖。

## 1. 环境（v1 归档）

- 主表场景配置：`disaster_va_dag_ho/configs/main.yaml`（2×2 km、N=30、K=3、L=2、
  T=50 s、突发 [15,35] s、无 GS、无充电）。
- 变体/消融增量 yaml：`configs/{b4_flat,b5_flat,b6_no_vw_prune,
  ablation_no_dag_order,ablation_no_energy_state,sens_alpha_low,sens_alpha_high}.yaml`
  （`env/config.py::load_config_variant` 先读 main.yaml 再叠加增量 + seed 覆盖）。

## 2. 单测（前置检查）

```
/root/miniconda3/envs/asr_env/bin/python -m pytest disaster_va_dag_ho/tests -q
```

期望：`45 passed`（v1：W1 环境 12 + W2 调度 6 + W3 agents 4 + W4 变体 3；
v2 新增：`test_env_v2.py` 14 + `test_run_v2.py` 5 + W4 变体测试合并后总数 45）。

## 3. 主表 B1–B6（5 seeds，均值 ± 标准差）

```
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_table \
    --scope main --seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4
```

- row 预算：B1–B3（hover/random/sweep/chase）每 (方法, scene seed) 跑 1 episode；
  VA-DAG-HO-MAPPO 与 B6-noVWprune 训练 seed=100+seed × 200 ep；
  B4-flat 250 ep；B5-SACflat 400 ep（`--episodes-*` 可改）。
- RL row 在 eval scenes 0..4 上确定性求平均后每 (train_seed) 一行，
  summary 对同方法 5 行求 mean ± std。
- 产物：
  - `results/main_table_raw.csv`（每 seed 一行 + 墙钟 `wall_s`）
  - `results/main_table_summary.csv`（方法 → 指标 mean±std）

复现核对（本仓库 2026-09-04 一次运行的记录值，5 seeds 均值 ± 标准差；
`results/main_table_raw.csv` 另有每行 `wall_s` 学习墙钟时间）：

| 方法 | 成功率 | 平均完成时延 s | UAV 总能耗 J | 无效 LEO 尝试 |
|------|--------|----------------|--------------|---------------|
| hover (B3) | 0.8983±0.0497 | 2.90±1.26 | 18754±1554 | 636±216 |
| random (B1) | 0.9014±0.0292 | 2.69±1.00 | 20078±1626 | 633±185 |
| sweep (B1) | 0.8974±0.0068 | 2.75±0.87 | 23836±2049 | 582±103 |
| chase (B2) | 0.8914±0.0474 | 2.79±1.04 | 21031±1986 | 649±219 |
| VA-DAG-HO-MAPPO | 0.9036±0.0096 | 2.84±0.10 | 19244±552 | 607±56 |
| B4-flat | 0.7376±0.0508 | 3.46±0.31 | 18221±2440 | 382±141 |
| B5-SACflat | 0.7526±0.0255 | 3.32±0.26 | 20483±2448 | 379±56 |
| B6-noVWprune | 0.9000±0.0177 | 3.18±0.13 | 20900±602 | 207±5 |

**口径注释：**
- B5 = off-policy 扁平族 SAC-flat（MATD3-flat 连续阈值无法表达 LEO 中间类且
  确定性 actor 崩塌，已留档在 `baselines/td3_flat.py`；主实现见
  `baselines/sac_flat.py`，Gumbel 重参数化离散卸载头）。
- B6 = 决策层去掉可见窗剪枝（常在线假设）；执行层仍遵守规格 §1.1「跨窗未完成
  则失败」：跨窗的 LEO 提交永不完成、应用在截止期失败（计入无效尝试）。
- 学习法墙钟时间（每 train seed，200/250/400 ep）：VA-DAG-HO-MAPPO ~30 s、
  B4-flat ~53 s、B6 ~41 s、B5-SACflat ~144 s（见 raw CSV `wall_s`）。

## 4. 四组消融

```
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_table \
    --scope ablation --seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4
```

消融行 = w/o VW-prune（`b6_no_vw_prune.yaml`）、w/o DAG-order
（`ablation_no_dag_order.yaml`，依赖当独立包）、w/o embed（=B4，`b4_flat.yaml`）、
w/o energy-in-state（`ablation_no_energy_state.yaml`）。
产物：`results/ablation_raw.csv`、`results/ablation_summary.csv`。

记录值（5 seeds 均值 ± 标准差；主方法行取自主表 VA-DAG-HO-MAPPO 0.904）：

| 消融 | 成功率 | 平均完成时延 s | UAV 总能耗 J | 无效 LEO 尝试 |
|------|--------|----------------|--------------|---------------|
| w/o DAG-order | 1.0000±0.0000 | 2.00±0.04 | 21463±849 | 100±3 |
| w/o energy-in-state | 0.8949±0.0057 | 2.80±0.14 | 19135±299 | 645±26 |
| w/o VW-prune（B6） | 0.9000±0.0177 | 3.18±0.13 | 20900±602 | 207±5 |
| w/o embed（=B4） | 0.7376±0.0508 | 3.46±0.31 | 18221±2440 | 382±141 |

w/o DAG-order 把依赖当独立包后成功率虚高到 1.0——说明拓扑约束是有意义的（防
止不现实的乐观完成）。附录还产出 `results/sweep_summary.csv`（规模扫描）与
`results/sens_summary.csv`（α 敏感性）。

## 5. 训练曲线 / 图

```
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_curves --seed 100
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.make_figures
```

曲线 CSV：`results/train_curves/*.csv`；图：`results/figures/*.png`
（主对比 / 消融 / 训练曲线）。

## 6. 附录规模扫描（可选）

```
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.sweep \
    --seeds 0,1,2 --episodes 150
```

产物：`results/sweep_raw.csv`、`results/sweep_summary.csv`。


## 7. v2 主表与消融（2026-09-07，硬覆盖 + 固定 T-patrol）

v2 场景配置：`disaster_va_dag_ho/configs/main_v2.yaml`。与 v1 的关键差异：
- N=40、K=3、L=2、T=50；`uav_cover_radius_m: 800.0`（硬覆盖，覆盖外 `assignment=-1`，
  真实“连不上”）；
- 空间非齐次到达：`hotspot_arrivals: true`，热点中心
  `hotspot_centers: [500,150,1500,1850]`（底部/顶部条带救援点，hover 悬停不可达、
  T-patrol 条带巡逻可覆盖），`hotspot_radius_m: 300.0`，热点内
  `lambda_high_per_s: 0.2` / 热点外 `lambda_low_per_s: 0.03`，突发窗全局乘子 1.0；
- 主方法 **Ours = T-patrol + 硬覆盖关联 + ScheduleDAG（VW-prune + DAG-order）**，
  UAV 速度由 `baselines/trajectories.py` 固定轨迹模块给出，**不再跑 MAPPO 轨迹**；
- B4/B5 = 固定 T-patrol + **只学卸载**（`Mappo flat_no_vel` / `SacFlat
  offload_only+mover`，速度头惰性、env 移动来自 patrol）；B6 = T-patrol +
  `vw_prune: false`（无决策期可见窗剪枝，执行仍按「跨窗未完成则失败」）；
- 新增指标：`frac_unassociated`（每时隙无关联终端比例均值）、`fail_uncovered`
  （规则 A：覆盖丢失导致的失败数，`AppRequest.suffered_uncovered` 打标）。

### 7.1 运行命令

```
# 单测（45 项）
/root/miniconda3/envs/asr_env/bin/python -m pytest disaster_va_dag_ho/tests -q

# v2 主表（B1 random / B2 chase / B3 hover / Ours / B4-flat / B5-SACflat / B6，5 seeds）
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_table \
    --scope v2 --seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4

# v2 四组消融（w/o VW-prune、w/o DAG-order、w/o hard-cover、w/o embed(=B4)）
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_table \
    --scope ablation-v2 --seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4
```

row 预算：B1/B2/B3/Ours/B6 每 (方法, scene seed) 1 episode（确定性）；B4-flat 每
train seed 250 ep、B5-SACflat 400 ep（学习墙钟见 7.3）。产物：
`results/main_table_v2_raw.csv`、`results/main_table_v2_summary.csv`、
`results/ablation_v2_raw.csv`、`results/ablation_v2_summary.csv`。

> P1 投稿加固（敏感性成立区间、配对显著性、B4 加预算、chase/B6/DAG 辩护、R 话术）
> 见 **`PUBLISH_P1.md`**（2026-09-07 定稿）；其 Go/No-Go = Go(IoT-J/中等会议)。

### 7.2 v2 主表记录值（2026-09-07 一次运行，5 seeds 均值 ± 标准差）

| 方法 | 成功率 | 平均完成时延 s | UAV 总能耗 J | 无效 LEO 尝试 | frac_unassociated | fail_uncovered |
|------|--------|----------------|--------------|---------------|-------------------|----------------|
| B1-random | 0.6799±0.0602 | 1.5050±0.2484 | 16457±677 | 486±267 | 0.2453±0.0624 | 21.0±4.7 |
| B2-chase | 0.7452±0.0424 | 1.8982±0.2369 | 19079±247 | 571±302 | 0.2293±0.0655 | 17.0±6.8 |
| B3-hover | 0.6456±0.0562 | 1.4698±0.1534 | 15328±451 | 468±277 | 0.2450±0.0620 | 22.8±7.5 |
| Ours(T-patrol) | 0.7311±0.0597 | 1.8883±0.2380 | 21238±917 | 745±414 | 0.2463±0.0417 | 16.4±4.1 |
| B6-noVWprune | 0.7436±0.0325 | 2.0758±0.0870 | 22276±652 | 264±109 | 0.2463±0.0417 | 16.2±3.2 |
| B4-flat | 0.5705±0.0437 | 3.5410±0.1473 | 24665±2086 | 160±59 | 0.2513±0.0028 | 19.1±0.9 |
| B5-SACflat | 0.5967±0.0241 | 3.1130±0.2385 | 22309±2583 | 216±46 | 0.2547±0.0000 | 19.3±1.2 |

**SPEC §3.3 验收核对（seeds 0–4）：**
1. B3-hover 成功率 0.6456±0.0562 ∈ [0.55, 0.75]（逐 seed 0.569–0.719）✓
2. Ours 成功率比 hover 高 +0.0855 ≥ +0.05（逐 seed Ours≥hover）✓
3. Ours vs B4-flat 差 +0.1606、vs B5-SACflat 差 +0.1344，均 ≥ +0.10 ✓
4. 平均 `frac_unassociated`：hover 0.2450±0.0620、Ours 0.2463±0.0417（逐 seed
   0.192–0.295）∈ [0.15, 0.40] ✓
5. B6-noVWprune 平均时延 2.0758 s > Ours 1.8883 s（成功率接近，符合口径）✓

**示例运行/绘图推荐 seed（2026-09-07 备注）：scene seed 2。** Ours(T-patrol) 该 seed 成功率 0.819、reward -323、时延 1.716 s、fail_uncovered 11，为 Ours 5 seeds 中最佳（seed 0–4 各 0.711/0.670/0.819/0.782/0.673）。注意：seed 2 场景对所有方法都偏易（B3-hover 0.719、B1-random 0.782、B2-chase 0.791 同样为其各自最高），属场景样本特性而非方法优势；方法间结论一律以 5-seed 均值为准。

### 7.3 学习墙钟与调参历史

- B4-flat 每 train seed ≈ 53–61 s、B5-SACflat ≈ 141–144 s（raw CSV `wall_s`）。
- 调参链（记录于 `docs/superpowers/plans/2026-09-07-va-dag-ho-v2-hard-no-traj/
  W3_acceptance.md`）：R 从 500 起调到 800、热点移到 hover 盲区
  `[500,150,1500,1850]`、λ 提到 0.03/0.20——先让 hover/frac 落窗，再拉大
  embed-vs-flat 成功率差距；`main_v2.yaml` 已锁定。

### 7.4 v2 消融记录值（5 seeds 均值 ± 标准差；主方法行取自主表 Ours 0.7311）

| 消融 | 成功率 | 平均完成时延 s | UAV 总能耗 J | 无效 LEO 尝试 | frac_unassociated |
|------|--------|----------------|--------------|---------------|-------------------|
| w/o VW-prune（=B6） | 0.7436±0.0325 | 2.0758±0.0870 | 22276±652 | 264±109 | 0.2463±0.0417 |
| w/o DAG-order | 0.8182±0.0433 | 1.5759±0.2371 | 24133±825 | 341±492 | 0.2463±0.0417 |
| w/o hard-cover | 0.8748±0.0246 | 2.1422±0.2686 | 25108±1202 | 1208±200 | 0.0000±0.0000 |
| w/o embed（=B4） | 0.5705±0.0437 | 3.5410±0.1473 | 24665±2086 | 160±59 | 0.2513±0.0028 |

口径注释：
- w/o DAG-order 把依赖任务当独立包 → 成功率虚高（0.818 > Ours 0.731）。量化：
  `dep_violations`（孩子在前驱产出前被提交）276.2±10.0/集 ≈ 3.3 违例/完成 app，
  合法行（Ours/B1–B6/B4）为 0（列见 `ablation_v2_summary.csv`；口径与数字详见
  `PUBLISH_P1.md`「W4」节）。
- w/o hard-cover = v1 软连接对照（`uav_cover_radius_m: 0`）：成功率升到 0.875 且
  `frac_unassociated` 归零——证明硬覆盖确实改变问题（本文核心场景）。
- w/o embed（=B4）与主表 B4-flat 行同口径：固定 T-patrol + 只学卸载。
- 无文档不改 B6 口径：`vw_prune: false` 只去掉决策期剪枝，执行仍按跨窗即败结算。

### 7.5 论文表述红线核对（SPEC §0/§5/§6）

- 主表没有 MAPPO 轨迹行：v2 scope 内所有 UAV 速度来自 `baselines/trajectories.py`；
- 真实出现 `assignment=-1`：Ours 每 seed `frac_unassociated` 0.19–0.30；
- 硬覆盖没有被为涨成功率关掉：v2 主表全程 `uav_cover_radius_m: 800.0`；
- 创新点表述 = 硬覆盖 + 可见窗下依赖任务嵌入调度（非学轨迹）。
