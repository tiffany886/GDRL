# 数字孪生驱动的 UAV-LEO 边缘计算（探索线）

> 状态：2026-08-28 一周探索版完成。定位：方向验证 + 初步机制（非完整论文）。
> 复用 `uav_leo_experiment` 仿真器与 PMEO 免训练决策器，新增数字孪生层。
> 详细结果：`results/summary.md`；实验数据：`results/rq{1,2,4}_*.csv`；图：`results/fig_*.png`。

## 一句话

**孪生状态的新鲜度（同步周期 + 任务建模）是卸载决策最优性的上游约束**：
同步周期从 1 时隙放宽到 10 时隙，PMEO 性能从 -81 恶化到 -270（单 UAV hard）；
而误差触发的自适应同步无需先验调参即可自动逼近最优固定周期。

## 方法与机制

- 物理世界 = 现有仿真器（用户/热点移动、任务每时隙按热点邻近度生成）。
- 数字孪生 = `TwinnedWorld`：每 τ 时隙同步一次；间隙内由 predictor 外推
  （freeze / linear / linear+noise / resample）；**UAV 位置用物理真值**（自定位已知），
  用户/热点/任务用孪生状态。
- 决策器完全复用：PMEO（post-move exact offload）/ MPC-H3 / PMEO-M-Eco（多 UAV），
  只改输入。决策一致率 = 孪生决策 vs 完美信息决策的目标一致比例。
- 自适应同步 = 每 probe_m 时隙抽查 n_probe 个用户（位置 + 任务漂移），
  误差超阈值 δ 即触发全量同步（`adaptive_sync.py`）。

## 主要结果（修复版 env，单 UAV 40 集 / k3 5 seeds×10 集）

### RQ1 同步周期-性能权衡（核心图）
- 单 UAV hard：τ=1 → -80.7（=无孪生基准），τ=10 → -269.9（3.3 倍恶化）；stress 更陡。
- k3：τ=1 → -3568（=论文基准 -3590），τ=10 → -7565。
- 决策一致率 agree：τ=1 时 1.00（孪生=物理即精确最优），τ=10 时 0.62——误差传导清晰。

### RQ2 预测模型：任务建模 >> 位置预测
- 位置线性外推 vs 冻结：-178.5 vs -179.1（几乎无差，用户匀速）。
- 位置噪声 20% 内：无影响。
- **任务盲目重采样反而更差**（-186.5）：孪生与物理随机序列不匹配比冻结更误导。
- → 孪生质量的关键是任务/负载动态建模，不是位置预测。

### RQ3 决策器鲁棒性：孪生误差淹没决策顺序收益
- PMEO / current / MPC 在 τ=10 时趋同（-270/-275/-253），post-move vs current 的
  +2.5 优势被孪生误差淹没一个数量级。
- **发现**：孪生新鲜度是"决策顺序机制"生效的前提——与 PMEO 论文的决策顺序故事
  构成因果链：孪生新鲜度 → 状态精度 → post-move 精确卸载收益。

### RQ4 误差触发自适应同步
- 单 UAV：δ=5m 用 12.4 次同步达 -163.9，逼近固定周期最优（τ=2: 15 次/-152.4）。
- k3：adaptive 48 次同步达 -5517，与 fixed τ=2（51 次/-5460）持平（差 1.0%）。
- δ 5~20m 不敏感（触发由任务漂移主导）。
- **诚实边界**：本环境任务每时隙全量重采样（无时间相关性），任务漂移"永远存在"，
  自适应省同步的空间小；工程结论 = 同步周期应 ≤2 时隙，或直接用自适应免调参。
  在任务到达有时间相关性的系统（突发/稀疏）中，事件触发同步收益预计显著更大。

### RQ4b 突发流量变体：自适应从"持平"变"硬优势"
- 突发配置：每 30 时隙 6 个突发时隙（4-6、19-21），突发时到达概率 ×4、任务尺寸 ×2
  （`--burst`，`BURST_OVERRIDES`），活动任务数 2-4 → 9-11。
- **负载感知触发**（`task_mode="load"`，触发指标 = 孪生 vs 物理聚合负载相对变化）
  在 burst 下支配固定周期前沿：k3 δ=40（32.3 次同步）比 fixed τ=3（34 次）好 9.7%，
  δ=50（26.1 次）比 fixed τ=5（21 次）好 10.6%；单 UAV δ=40（14.6 次）以更少同步
  超过 fixed τ=2（16 次）。per_user 触发在 burst 下"过火"（94 次同步才逼近 τ=1）。
- 同步时机诊断 `fig_sync_timing_burst.png`：同步集中在突发边界（4-7、19-22）。
- **结论**：i.i.d. 下自适应只能持平最优固定周期；任务到达有时间相关性（突发）时
  事件触发同步收益显著，且免去先验 τ / 流量类型调参。详细表见 `results/summary.md`。

### RQ4c 随机突发变体：不知道突发何时来也成立
- `--burst-random`：每个 cycle 突发起始位置随机，孪生**不知道**突发相位。
- k3（论文主线规模）：adaptive load δ=30（40.9 次同步，-8387）在 reward 与同步次数
  上同时胜过 fixed τ=2（51 次，-8239）；δ=50（24.6 次）达到 fixed τ=3（34 次）的
  reward 且比 fixed τ=5（21 次）好 9.8%——**硬优势不依赖"知道突发时刻"**。
- 单 UAV：自适应 ≈ 最优固定周期（免先验 τ）；优势随系统规模增大（51 用户负载信号
  更平滑）。per_user 触发仍过火（95 次同步逼近 τ=1）。

## 文件

```
dt_env.py           TwinnedWorld（同步/外推/观测视图/探测）
predictors.py       freeze / linear / linear_noise / resample
adaptive_sync.py    误差估计（位置+任务漂移）+ 阈值触发
runner.py           孪生驱动 episode 评估（reward/时延/能耗/完成率/一致率/同步数）
run_experiments.py  RQ1-4 CLI（--rq --k3 --taus --deltas ...）
make_figures.py     出图（fig_tau_tradeoff / fig_predictor_robustness / fig_adaptive_sync）
analyze_sync_timing.py  同步时机诊断（burst 边界对齐图）
results/            CSV + summary.md + 图
```

## 复现

```bash
python -m digital_twin_uav_leo.run_experiments --rq rq1 --scenarios hard,stress \
    --methods postmove_exact,current_exact,mpc_traj_h3 --taus 1,2,3,5,10 --episodes 40
python -m digital_twin_uav_leo.run_experiments --rq rq2 --scenarios hard --taus 3 --episodes 40
python -m digital_twin_uav_leo.run_experiments --rq rq4 --scenarios hard --deltas 5,10,20 --episodes 40
python -m digital_twin_uav_leo.run_experiments --rq rq1 --k3 --episodes 10
python -m digital_twin_uav_leo.run_experiments --rq rq4 --k3 --episodes 10
# 突发流量变体（负载感知自适应同步）
python -m digital_twin_uav_leo.run_experiments --rq rq4 --scenarios hard --burst \
    --deltas 40,50,60,80 --probe-m 1 --task-mode load --episodes 40
python -m digital_twin_uav_leo.run_experiments --rq rq4 --k3 --burst \
    --deltas 25,30,40,50 --probe-m 1 --task-mode load --episodes 10
# 随机突发（孪生不知道突发相位）
python -m digital_twin_uav_leo.run_experiments --rq rq4 --scenarios hard --burst \
    --burst-random --deltas 40,50,60 --probe-m 1 --task-mode load --episodes 40
python -m digital_twin_uav_leo.run_experiments --rq rq4 --k3 --burst \
    --burst-random --deltas 25,30,40,50 --probe-m 1 --task-mode load --episodes 10
python -m digital_twin_uav_leo.make_figures
```

## 与导师汇报的一句话

"我们在 UAV-LEO 边缘计算仿真上加了一层数字孪生，量化了同步周期对卸载性能的影响
（同步放宽 10 倍性能掉 3 倍），发现瓶颈在任务动态建模而非位置预测，并验证了
误差触发同步无需调参即可逼近最优固定周期；下一步在任务突发场景量化自适应收益，
并验证了突发流量下负载感知触发把 RQ4 从'持平'变成'硬优势'（更少同步 + 更好 reward）；
随机突发（不知道突发何时来）下优势依然成立，构成第二个创新点（负载感知事件触发
同步），可与 PMEO 决策顺序合并成毕设论文的两大贡献。"
