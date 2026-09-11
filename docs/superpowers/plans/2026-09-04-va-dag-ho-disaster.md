# VA-DAG-HO（灾后 UAV–LEO 可见窗感知 DAG 分层卸载）—— 实现计划

> **面向 AI 代理的工作者：** 逐任务执行本清单，用复选框（`- [ ]`）跟踪进度。按仓库约定**不自动 git commit**；如需要提交请完成后再请用户确认。
> 规格锁定版（2026-09-04）：`docs/spec_va_dag_ho_disaster.md` = `docs/superpowers/specs/2026-09-04-va-dag-ho-disaster-design.md`（两处一致）。
> Python：`/root/miniconda3/envs/asr_env/bin/python`；测试用 pytest。代码目录：`/code/docs/GDRL/disaster_va_dag_ho/`。
> 更新：2026-09-04 建清单；**W1 已完成（12 项单测）；W2 已完成（6 项单测，B1–B3 出指标）；W3 已完成（MAPPO embed/flat 可训，2 seeds 收敛趋势）；W4 已完成（B5 SAC-flat、B6 口径、主表 B1–B6 ×5 seeds、四消融、configs 固化、墙钟时间）；W5 已完成（规模扫描、图、REPRODUCE.md、收尾核对，单测共 26 项通过）**；每周验收标准见对应周末尾。

**一句话目标：** 灾后（无 GS、无充电站）多 UAV 空中边缘 + 可见窗受限 LEO 回传；上层 MAPPO 只学水平轨迹，下层嵌入 ScheduleDAG（可见窗剪枝 + DAG 拓扑序）做卸载调度；产出 B1–B6 主表 + 四组消融 + ≥5 seeds + `REPRODUCE.md`。

**关键锁定（实现期间不可漂移）：**
- 主实验无地面应急站 GS；窗外 LEO 容量 = 0（硬约束）；选 LEO 且估计耗时 > 剩余可见时间 τℓ → 禁止该放置。
- RL 只输出轨迹；规则关联 + ScheduleDAG 决定卸载；「先飞后调度」仅是仿真因果，不得写成创新点。
- 能量：SoC + 每步扣飞行/通信/计算 + 低电惩罚，无充电/换电。
- B7 GDRL 可选、主验收不卡；对比算法禁用「仅 2018 前 DQN 冒充 SOTA」。
- 相对 PMEO（决策顺序）与 LAETS（事件触发同步）刻意错开；不得把专家轨迹/孪生同步当主方法。

**时隙因果顺序（所有模块统一实现）：**
`观测 → RL 轨迹 → 移动并扣飞行电 → 规则关联 → ScheduleDAG(新位置 + 当前可见窗) → 结算 L/E/F → 奖励 →（周期性）MAPPO 更新`

**可复用代码（只读引用，勿改原文件）：**
- LoS/NLoS 与香农速率：`uav_leo_experiment/physics.py`（`los_probability` / `path_loss_db` / `shannon_rate` / `rate_user_uav` / `rate_uav_leo`）
- UAV 飞行能耗近似：`uav_leo_experiment/physics.py::flight_energy`；SoC 口径参考 `digital_twin_uav_leo/dt_env.py`
- MAPPO 脚手架参考（思想级）：pengchengau/GDRL、marlbenchmark/on-policy

**目录结构（已建 /code/docs/GDRL/ 下）：**
```
disaster_va_dag_ho/
  __init__.py            # 包
  README.md              # 入口：指向规格与本计划（已建）
  env/                   # 场景、终端/UAV、星历、信道、SoC、因果 step
  scheduler/             # ScheduleDAG + 可见窗剪枝
  agents/                # MAPPO 轨迹策略 + 规则关联
  baselines/             # B1 Random/Sweep、B2 Chase-Backlog、B3 Hover、B4/B5、B6、B8
  configs/               # *.yaml：主表/消融/扫描
  scripts/               # train / eval / figures
  tests/                 # 单测与集成冒烟
```

---

## W1：环境骨架（地图、动力学、SoC、简化星历、窗外硬零容量）

**验收：** 单 episode 可空跑；窗外容量断言与 SoC/投影单测通过。

- [x] W1.0 创建 `disaster_va_dag_ho/` 目录树、`__init__.py`、`README.md`
- [x] W1.1 配置模块：`configs/main.yaml` + `env/config.py::load_config`。区域 2×2 km、N=30、K=3、H=100 m、L=2、Δt=1 s、T=50、突发窗 [15,35]、权重/SoC/并发上限等默认值齐
- [x] W1.2 终端与任务到达：`env/mobility.py::Terminals`（半静态慢移 + 反弹）；`env/tasks.py::ArrivalProcess` 非齐次泊松 + 每终端并发上限 1–2；应用先以 `AppRequest.dag=None` 桩接入
- [x] W1.3 UAV 动力学：`env/env.py::step/_gate_actions`；水平速度动作 ‖v‖≤v_max、越界 Proj 钳回、移动扣飞行电
- [x] W1.4 能量：`uav_soc` 账本 + `_consume_uav_energy`（flight/comm/compute 分账）；低电限速/空电悬停 + 大惩罚；无充电
- [x] W1.5 简化星历：`env/ephemeris.py::Ephemeris` 确定性交错可见区间 [t_in,t_out]；观测含 τℓ；窗外 C=0 硬约束
- [x] W1.6 信道：`env/channel.py` 复用 `uav_leo_experiment/physics.py` 的 LoS/NLoS 香农速率；UAV–LEO 窗内 min(r_kℓ, Cℓ)、窗外 0
- [x] W1.7 `env/step` 按规格 §6 因果顺序串起：到达 → 观测 → 轨迹 → 移动扣电 → 规则关联 → 调度（W2 换真 ScheduleDAG）→ 结算 L/E/F
- [x] W1.8 单测 `tests/test_env_w1.py`（12 项全绿）：窗外容量断言 = 0、投影钳回 + 边界惩罚、SoC 单调降、低电/空电门控、关联全覆盖、整回合空跑

**验证（已通过）：**
```
cd /code/docs/GDRL
/root/miniconda3/envs/asr_env/bin/python -m pytest disaster_va_dag_ho/tests -q
```
→ `12 passed`；空跑痕迹：50 步 hover，flight 12000 J、63 app 到达、L 729 s、F=58（W1 桩无调度器，W2 起由 ScheduleDAG 承接）。

---

## W2：DAG 到达 + ScheduleDAG + B1–B3（无 RL 出指标）

**验收：** DAG/窗剪枝单测通过；无 RL 可出成功率/时延/能耗/无效 LEO 指标。

**已通过（4 方法 × 5 seeds，`disaster_va_dag_ho/results/baselines_w2.csv`）：**

| 方法 | 成功率 | 平均完成时延 s | UAV 能耗 J | 无效 LEO 尝试 |
|------|--------|----------------|------------|---------------|
| hover | 0.898 | 2.90 | 18754 | 636 |
| chase | 0.891 | 2.79 | 21031 | 649 |
| sweep | 0.897 | 2.75 | 23836 | 582 |
| random | 0.901 | 2.69 | 20078 | 633 |

负载已标定到非平凡区间（~0.90 成功率、~3 s 时延、LEO 窗显著争用）；B1–B6 区分度在 W4 主表阶段调参。

- [x] W2.1 DAG 模板 3 种（chain-light / fork-join / chain-heavy）：节点 4–5、深度 3–4，骨架「采集→检测→融合→决策」；`scheduler/dag.py::generate_dag` + `topological_order`；负载标定 `dag_cycles_scale/dag_bits_scale`
- [x] W2.2 应用绑定终端 n、输入数据在 n、带截止期；`env/tasks.py` 并发上限强制；应用按到达时关联固定到 UAV（避免未建模 UAV–UAV 链路）
- [x] W2.3 就绪集 R（前驱完成且数据可达）；跨节点传输（终端→UAV 上行、UAV→LEO 中继，按资产串行）+ 设备算力时间轴 + 紧急度排序（截止期 > 重节点 > 普通）
- [x] W2.4 `scheduler/schedule_dag.py`：候选 = 关联 UAV + LEO；LEO 剪枝要求「中继+服务」落入同一可见窗（超 τℓ → 禁止并计无效尝试）；选完成时间最小可行放置；不可行 → 等待；汇点完成 → 成功；超截止期 → 失败
- [x] W2.5 接入 env 因果链：`_run_scheduling` 注册 app + `scheduler.step(ctx)`；按放置结算 UAV 计算/中继能耗并更新 SoC
- [x] W2.6 硬性单测 `tests/test_scheduler_w2.py`（6 项）：前驱未完成不得开工；跨窗 LEO 放置被拒；窗外/截止期前无窗 LEO 被拒；合法调度依赖违反 = 0；LEO placement 落在单一可见窗且不超截止期
- [x] W2.7 无 RL 基线：B1 Random/Sweep、B2 Chase-Backlog、B3 Hover（`baselines/no_rl.py`），共用同一下层 ScheduleDAG
- [x] W2.8 `scripts/eval_baselines.py` 输出 `results/baselines_w2.csv`（成功率/时延/能耗/无效 LEO）

---

## W3：MAPPO 学轨迹 + 规则关联 + B4 扁平接口

**验收：** 主方法与 B4 可训；≥1 seed 有收敛趋势。

**已通过（`disaster_va_dag_ho/results/`；每回合独立随机场景，窗口均值）：**

| 运行 | episodes | reward 前10% → 后10% | eval 成功率 首次→末次 |
|------|----------|------------------------|------------------------|
| embed seed0 | 200 | -285.3 → -272.2（熵 1.62→1.44） | 0.901 → 0.901 |
| embed seed1 | 100 | -288.1 → -272.3 | 0.885 → 0.911 |
| flat seed0 (B4) | 250 | -354.6 → -355.0 | 0.786 → 0.831 |
| flat seed1 (B4) | 100 | -354.3 → -333.1 | 0.816 → 0.834 |

embed（主方法）两 seed reward 上升、策略熵单调下降；B4 两 seed 可训且 eval 成功率上升、仍显著低于 embed（~0.83 vs ~0.90），为「嵌入必要性」提供对照。奖励组件分解已随 CSV 落盘。


- [x] W3.1 观测拼接（`env/observation`）：自身位置/SoC、责任区积压计数+待处理质心偏移、邻机相对位置、各 LEO τℓ、近期失败统计；B4 flat 追加每终端（关联/积压/就绪/截止期）特征块；尺寸固定可向量化
- [x] W3.2 连续速度动作 (vx,vy)，策略输出 [-1,1]×v_max 映射，env 门控/钳位；主方法动作只含轨迹
- [x] W3.3 规则关联：`agents/association.py::associate_terminals`（按 r_{n,k} 排序 + ceil(N/K) 负载上限），`env._associate` 复用
- [x] W3.4 奖励公式不变；`agents/trainer.py` 每回合记录 L/E/F 与三类惩罚分量（见 results CSV 列 reward_L/reward_E/reward_F/pen_*）
- [x] W3.5 MAPPO：`agents/mappo.py`（共享参数 actor + CTDE 全局 critic，GAE + clipped PPO）；`agents/trainer.py` + `scripts/train.py` 每回合更新并落 CSV
- [x] W3.6 B4 扁平：`mode=flat` 时 actor 额外对每个关联终端槽输出 {0=UAV,1=LEO,2=wait} 决策（`env.step(..., offload_decisions)` → `ScheduleDAG` 候选受限），不经嵌入代价启发式
- [x] W3.7 ≥1 seed 收敛趋势（详见下表与 `results/train_*_seed0/1.csv`）；训练曲线记录 reward/success/时延/能耗/无效 LEO/熵

---

## W4：B5/B6 + 主表 5 seeds + 消融

**验收：** 主表 B1–B6 数字齐；`configs/*.yaml` 固化可复现。

> 更新（2026-09-04，W4 执行中）：B5/B6/四消融/主表 runner 均已实现，
> 单测扩至 25 项（含 `tests/test_variants_w4.py`：w/o DAG-order 独立包语义、
> B6 常在线跨窗放置 + 实际窗延迟、依赖违反保持 0）。**B5 实现口径说明**：
> 先实现 MATD3-flat（连续 sigmoid 阈值离散化），发现其收敛后无法表达 LEO
> 中间类（sigmoid 饱和），且确定性 actor 出现“满速冲向角落”的值高估崩塌、
> 无法到平台期；故 B5 落盘改为同属 off-policy 扁平族、带 Gumbel 重参数化
> 离散头与自维持探索的 SAC-flat（`baselines/sac_flat.py`；MATD3-flat 参考
> 实现保留在 `baselines/td3_flat.py` 并注明原因）。B6 口径固定为“去掉剪枝
> 分支（决策按常在线名义完成度），执行层遵守规格 §1.1「跨窗未完成则失败」：
> 跨窗的 LEO 提交永不完成、应用在截止期失败并计入无效尝试（对应
> `configs/b6_no_vw_prune.yaml` 中 `vw_prune: false`）。主表 row 预算：
> VA-DAG-HO-MAPPO/B6 embed 200 ep、B4 flat 250 ep、B5 SAC-flat 400 ep；
> 主表 5 seeds 已跑出（见 `REPRODUCE.md` 表格），单测现为 26 项。

- [x] W4.1 B5 扁平 MADDPG/MATD3 族（off-policy，学飞 + 学卸载，无嵌入；落盘 SAC-flat，见上口径说明）
- [x] W4.2 B6 变体：DAG 感知但**无可见窗剪枝**（常在线/弱窗）。口径在 config 写明：决策去掉剪枝分支（保留窗外 0 容量），执行仍按「跨窗未完成则失败」结算
- [x] W4.3 主表：B1–B6 × 5 seeds，均值 ± 标准差（见 `REPRODUCE.md` 与 `results/main_table_summary.csv`）；主表权重固定 main.yaml，附录 α 敏感性 yaml 已备（sens_alpha_low/high）
- [x] W4.4 消融四组：w/o VW-prune；w/o DAG-order（当独立任务）；w/o embed（=B4）；w/o energy-in-state（`results/ablation_summary.csv`）
- [x] W4.5 学习法墙钟时间记录（main raw `wall_s`：embed ~30 s、B4 ~53 s、B6 ~41 s、SAC ~144 s/seed）；`configs/` 固化主表/消融/B6 口径/敏感性 yaml
- [x] W4.6 数据落盘统一（raw + summary CSV，字段含 success/delay/energy/invalid/wall_s），对应 §9 交付物

---

## W5：规模扫描、画图、复现文档

**验收：** 按 `REPRODUCE.md` 能复现主表至少一行。

- [x] W5.1 附录规模扫描：N ∈ {20,30,40}、K ∈ {2,3}、突发强度 ×1 / ×1.5（`scripts/sweep.py`；产物 `results/sweep_summary.csv`，2 seeds 参考级）
- [x] W5.2 图（`scripts/make_figures.py`）：训练曲线（本文 vs B4/B5）；主对比（成功率/时延/能耗/无效 LEO）；消融图（`results/figures/*.png`；可见窗时间轴图并入主对比图数据源，SoC 曲线可选）
- [x] W5.3 `REPRODUCE.md`：环境、seed、yaml、运行命令、预期主表数字（含 B6/B5 口径注释）
- [x] W5.4 收尾全量测试（26 项通过）+ 规格 §12 检查清单核对（见下）

**训练曲线留档（`results/train_curves/*_seed100.csv`，eval 成功率随 episode）：**

| 方法 | 进展 |
|------|------|
| embed（主） | 0.889@24 → 0.90–0.92 平台（200 ep） |
| B4 flat | 0.71 → 波动 0.56–0.75，250 ep 处 ~0.69（表内 5 seeds 均值 0.738） |
| B5 SAC-flat | 0.78 → 波动 0.68–0.77，400 ep 处 ~0.77（表内 5 seeds 均值 0.753） |

α 敏感性（附录）：alpha-low 0.896±0.010、alpha-high 0.906±0.010（`results/sens_summary.csv`）。

### 规格 §12 收尾清单核对（2026-09-04）

- [x] 窗外卸 LEO 是否仍计成功？—— 主方法全部 LEO placement 落于单一可见窗内
  （`tests/test_scheduler_w2.py::test_legal_schedule_...` 覆盖），跨窗永不计成功
- [x] DAG 是否被当成独立包？—— 合法调度下 `dep_violations == 0`（W2/W4 单测）；
  仅 `ablation_no_dag_order` 有意的“独立包”语义使成功率虚高到 1.0（消融目的）
- [x] 是否只在 reward 赢、成功率/能耗翻车？—— 主方法成功率最高且平台稳定（0.904），
  B4/B5 成功率显著更低，能耗无翻车
- [x] 扁平 MARL 是否欠训就对比？—— B4/B5 均有 250–400 ep 训练曲线留档与 5 seeds
  均值±std；两者明显低于 embed（0.74–0.75 vs 0.90），结论方向稳健（B5 曲线噪声较大，
  已在口径注释说明）
- [x] 贡献表述是否撞 PMEO/LAETS？—— 计划/规格明确“RL 只学飞、下层嵌入调度”，
  无专家轨迹、无孪生同步表述
- [x] B6 是否真正弱化/去掉窗剪枝？—— `vw_prune: false` 只去掉决策期剪枝，
  执行仍按「跨窗未完成则失败」，B6 时延/能耗劣于主方法
- [x] 下层表述为「嵌入式启发式近似求解器」，不宣称逐步全局最优（代码 docstring 与
  计划均按此表述）

---

## 收尾检查清单（对应规格 §12/§13）

- [x] 窗外卸 LEO 是否仍计成功？有则实验作废（无；单测保证窗内才计完成）
- [x] DAG 是否被当成独立包？（合法调度下依赖违反 = 0；仅 w/o-DAG-order 消融有意当独立包）
- [x] 是否只在 reward 赢、成功率/能耗翻车？（主方法成功率最高且平台稳定）
- [x] 扁平 MARL 是否欠训就对比？（B4/B5 有 250–400 ep 曲线 + 5 seeds；低于 embed）
- [x] 贡献表述是否撞 PMEO/LAETS？（无；RL 只学飞、下层嵌入调度）
- [x] B6 是否真正弱化/去掉窗剪枝？（`vw_prune: false` 去掉决策剪枝，执行按跨窗即败）
- [x] 下层表述为「嵌入式启发式近似求解器」，不宣称逐步全局最优（docstring/计划一致）
