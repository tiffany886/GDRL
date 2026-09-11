# 数字孪生驱动的 UAV-LEO 边缘计算 —— 设计文档（一周探索版）

> 日期：2026-08-28 · 状态：设计已批准 · 目标：一周内产出可向导师汇报的指标结果
> 定位：方向验证 + 初步机制研究（非完整论文）；与现有 PMEO / 决策顺序故事串联

## 1. 背景与目标

- 现状：`uav_leo_experiment/` 已有成熟 UAV-LEO-MEC 仿真器（物理模型已修复、PMEO 免训练
  决策器、配对检验/敏感度扫描/画图全套工具）。
- 动机：轨迹+能耗/延迟优化的论文已饱和；导师无方向偏好，要求"哪个效果好就用哪个"。
- 选择：**数字孪生（DT）驱动的 UAV-LEO 边缘计算**。2025 年 TVT / Ad Hoc Networks /
  Computer Networks 大量发表 DT+MEC 论文，热点真实；且最大程度复用现有资产，一周可出指标。
- 目标（一周）：量化"孪生同步周期-性能"权衡，提出**误差触发的自适应同步**机制，
  输出 1 张权衡曲线 + 1 张自适应同步对比表 + 汇报 README。

## 2. 系统定义

### 2.1 物理世界
现有仿真器 `env.py` 的 step 即物理世界：
- 用户沿路匀速运动（速度 reset 时固定），热点沿路移动（带反弹）；
- 任务每时隙按热点邻近度高斯核生成（`proximity = exp(-d²/2r²)`，到达概率 + 大小提升）；
- 无任务用户 `task_bits=0`。

### 2.2 数字孪生层（新增）
`TwinnedWorld` 抽象，参数 `(τ, predictor, ε, sync_mode)`：
- **同步**：每 `τ` 个时隙从物理世界拷贝完整状态（user_pos/vel、hotspot_pos/vel、
  task_bits、cycles_per_bit、backlogs、leo_pos/leo_backlog、t）。
- **间隙外推**（predictor ∈ {freeze, linear, linear+noise}）：
  - `freeze`：位置/任务全部冻结在最后同步值；
  - `linear`：用户位置 `p + v·Δt`（v 冻结为同步值）；热点用 `predict_hotspot`（含道路反弹，
    复用 `mpc_traj.py` 现有函数）；任务冻结；
  - `linear+noise`：线性外推基础上，对用户/热点位置注入高斯噪声
    `N(0, ε·scale)`（scale = 同步间隙内的预期移动距离），模拟感知/动态误差。
- **决策视图**：决策器读到的 obs = 孪生状态，但 **UAV 位置用物理真值**（无人机自定位
  GPS/IMU 已知，误差只在感知侧），这是符合现实的关键设计。
- **误差估计**（供自适应同步）：每 `m` 个时隙做一次"轻量探测"——随机抽查 `n_probe` 个
  用户/热点的真实位置，估计当前预测误差 `‖p_pred - p_true‖`。

### 2.3 决策器复用
- 单 UAV：`postmove_exact` / `current_exact` / `mpc_traj_h3`（`baselines.py`/`mpc_traj.py`）；
- 多 UAV：`PmeoMEcoPolicy`（`multi_uav.py`，论文主线）。
- 全部只改输入（喂孪生 obs），不改决策器内部逻辑 —— 隔离"孪生质量"影响。

## 3. 实验矩阵

### RQ1 同步周期-性能权衡
- 场景：单 UAV hard/stress（40 集，seed 73+episode 协议）；多 UAV k3（5 seeds × 10 集）
- 变量：`τ ∈ {1,2,3,5,10}`，predictor=linear，ε=0
- 输出：τ vs reward / 平均时延 / 能耗 / 任务级完成率 / 卸载决策一致率

### RQ2 预测模型与误差鲁棒性
- 单 UAV hard：predictor ∈ {freeze, linear, linear+noise(0.05, 0.1, 0.2)}，τ=3
- 输出：误差水平 vs 性能下降斜率；freeze vs linear 的差距（外推是否值得）

### RQ3 决策器鲁棒性对比
- 单 UAV hard：PMEO(post-move) vs current-position vs MPC-H3，τ ∈ {1,3,5,10}
- 问题：post-move 精确卸载依赖位置精确性，谁对孪生误差最敏感？

### RQ4 自适应同步（亮点）
- 单 UAV hard + 多 UAV k3：`sync_mode ∈ {fixed(τ), adaptive}`，adaptive 参数
  `m=2, n_probe=3, δ ∈ {5,10,20 m}`，对比基准为同平均同步间隔的 fixed τ
  （由 RQ1 曲线插值得到）
- 输出：同等平均同步开销下 adaptive vs fixed 的性能差；或同等性能下同步次数节省比例

### 附加指标
- **卸载决策一致率**：孪生决策 vs 完美信息决策（用物理真值重算 exact offload argmin）
  的一致比例，量化误差传导路径。
- **同步开销**：`总开销 = 同步次数 × c_sync`（c_sync 为参数化单次同步成本），
  用于 RQ4 的"同开销"比较。

## 4. 文件结构

```
/code/docs/GDRL/digital_twin_uav_leo/
  README.md            # 方向说明 + 结果摘要（导师汇报入口）
  dt_env.py            # TwinnedWorld：sync / predict / get_obs / probe_error
  predictors.py        # freeze / linear / linear+noise
  adaptive_sync.py     # 误差阈值触发同步逻辑
  run_experiments.py   # 一键跑 RQ1-4 矩阵，输出 CSV
  make_figures.py      # 权衡曲线 + 自适应对比图
  results/             # CSV + summary.md
```

依赖：仅 numpy + 现有 `uav_leo_experiment` 包（matplotlib 用于画图，已装）。

## 5. 一周时间线

| 天 | 交付 |
|---|---|
| D1 | `dt_env.py` + `predictors.py`，单 UAV hard τ 扫描跑通 |
| D2 | 单 UAV stress + 多 UAV k3 τ 扫描（RQ1） |
| D3 | RQ2 误差鲁棒性 + RQ3 决策器对比 |
| D4 | RQ4 自适应同步 + 决策一致率 |
| D5 | 图 + 表 + `results/summary.md` |
| D6 | README 汇报稿 + 复跑核对 |
| D7 | 缓冲（导师汇报 PPT 用图/表） |

## 6. 风险与诚实边界

- 一周版是"机制量化 + 自适应同步机制"，无重大理论贡献；汇报定位为方向验证。
- 用户速度恒定 → linear 外推几乎完美，误差主要靠 ε 注入和任务冻结制造，需在 README
  说明"误差注入模拟真实感知/动态误差"。
- 任务冻结高估负载 → RQ2 中 freeze 即包含该效应，作为对比基线。
- 若 RQ4 自适应无优势，如实报告并给 fixed τ 最优值作为工程结论（不硬凹）。

## 7. 复用资产清单

- `scenario_spec.MY_SCENARIOS`：权威场景参数
- `physics.simulate_task_multi` / `env.py`：物理世界
- `multi_uav.associate_users` / `exact_offload_multi` / `PmeoMEcoPolicy`：决策器
- `mpc_traj.predict_hotspot`：热点反弹预测
- 现有 40 集 / 5-seed 评估协议与 CSV 格式约定
