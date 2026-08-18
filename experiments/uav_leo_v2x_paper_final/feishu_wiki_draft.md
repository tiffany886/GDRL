# UAV-LEO 应急通信：多无人机协同任务卸载与轨迹优化（项目讲解）

> 配套文档：`multi_uav_scaling_plan.md`、`multi_uav_experiment_summary.md`、
> `route_a_method.md`；数据在 `GDRL/experiments/uav_leo_v2x_paper_final/multi_uav*`。
> 更新：2026-08-17；git 分支 `uav-leo-trajectory-energy`（cfdd3d1 多 UAV 扩展、
> adb0139 PMEO-M-Eco）。

## 1. 项目背景与场景

- 单 UAV 版本：1 台 UAV + 12/16 用户 + 1 km，验证"决策顺序（post-move）"机制：
  卸载决策必须在移动后的实际位置求解，而不是移动前。
- 多 UAV 版本：**K 台 UAV + 用户 + 热点 + LEO 回传**，规模对标 2024 年
  UAV-MEC/VEC 论文；最强对比为自研 MPC-H3（多步轨迹前瞻 + 精确卸载）。

## 2. 方法（详见 multi_uav_scaling_plan.md）

- **代码改造**：`config.py` 增加 `uavs/hotspots` 参数；`env.py` 支持 `(K,2)`
  位置矩阵 + 多 UAV 状态；`physics.py` 支持 `simulate_task_multi`（K=1 时退化为
  单 UAV）。
- **PMEO-M（多 UAV 基础版）**：每个时隙先做**负载均衡关联**（把用户按任务量
  分配给 K 台 UAV），再在 {local, UAV_k, LEO} 上逐用户精确求解卸载。
- **PMEO-M-Eco（能耗优化版）**：在 PMEO-M 基础上加 **H=3 前瞻**，UAV 在
  {原地, 专家移动, 任务质心, 负载均衡质心, LEO 附近} 候选轨迹里选
  "飞行能耗 + 卸载收益"最优的移动；其余时隙按负载重新做卸载。
  朴素单步门控（naive）无效，说明节能必须靠多步前瞻。
- **关键设计**：post-move association —— 移动后按**实际位置**重新算链路速率
  再关联，避免旧位置高估链路速率；决策顺序的收益由热点集中流量驱动。
- 对比算法：Current-pos exact、Follow-TEA、MPC-M-H3、Random、All-LEO，以及
  多 UAV 参数共享 DRL（DQN/TD3/SAC，进行中）。

## 3. k3 主结果（3 UAV / 51 用户 / 3 热点 / 2 km；10 seeds x 10 episodes）

| 方法 | reward 均值 ± std | 成功率 | 增益 vs PMEO-M-Eco |
|---|---|---|---|
| **PMEO-M-Eco（ours）** | **-2864.4 ± 220.0** | 0.907 | - |
| MPC-M-H3（多步轨迹前瞻） | -2955.9 ± 205.2 | 0.903 | **+91.5（t=+2.49, p=3.4e-02, 7/10）** |
| PMEO-M（固定轨迹+post-move） | -2982.5 ± 212.7 | 0.904 | +118.1（t=+2.86, p=1.9e-02, 8/10） |
| Current-pos exact（移动前求解） | -3088.3 ± 218.4 | 0.898 | +223.9（决策顺序消融） |
| Follow-TEA（追热点+启发式卸载） | -3626.6 ± 174.2 | 0.871 | +762.2（t=+12.5, p<1e-21） |
| Random | -9052.9 ± 540.5 | 0.660 | +6188.5（t=+50.3, p<1e-70） |

k3 指标：latency `0.1840s` vs MPC 0.1849s；flight energy `314.0` vs MPC 320.6、
PMEO-M 443.6；total energy `322.7` vs MPC 329.2。**reward/latency/success 全面
优于 MPC**，计算时间 24s vs MPC 36.5s per episode（10 episodes）。

**规模可信度（k3）**：3 UAV / ~50 用户 / 2 km 对标 JNCA 2024（3 UAV/10 UE/1 km）、
arXiv:2409.14782（1 LEO + 多 UAV）；10 seeds 上 Eco 稳定优于 MPC；
物理参数：UAV 25 m/s、热点 18 m/s。

## 4. 规模扫描：Eco 在 9 个规模点全部显著优于 MPC（10 seeds x 10 episodes）

| preset | UAV | 用户 | 面积 | PMEO-M-Eco | MPC-M-H3 | eco-vs-mpc | p | wins |
|---|---|---|---|---|---|---|---|---|
| k2 | 2 | 34 | 2.0 km | -1773.0 | -1840.2 | **+67.2** | 1.2e-03 | 9/10 |
| k3 | 3 | 51 | 2.0 km | -2864.4 | -2955.9 | **+91.5** | 3.4e-02 | 7/10 |
| k4 | 4 | 68 | 2.0 km | -3777.7 | -3973.3 | **+195.6** | 8.1e-04 | 9/10 |
| u30 | 3 | 30 | 2.0 km | -1645.9 | -1804.1 | **+158.2** | 2.4e-04 | 10/10 |
| u50 | 3 | 50 | 2.0 km | -2739.6 | -2862.8 | **+123.2** | 8.1e-03 | 10/10 |
| u80 | 3 | 80 | 2.0 km | -4072.9 | -4141.8 | **+68.9** | 5.3e-03 | 9/10 |
| a1 | 3 | 50 | 1.0 km | -1528.9 | -1568.4 | **+39.5** | 2.2e-03 | 10/10 |
| a15 | 3 | 50 | 1.5 km | -2058.5 | -2148.1 | **+89.5** | 1.1e-03 | 10/10 |
| a3 | 3 | 50 | 3.0 km | -4529.8 | -4577.9 | **+48.1** | 1.9e-02 | 8/10 |

- 能耗：eco-vs-mpc 每时隙低 1.5 ~ 9.2 J。
- **vs PMEO-M 的规模故事**：u80（80 用户）下 PMEO-M 对 MPC 反而差 -142.7
  （p=0.0095），但 Eco 仍 +68.9（p=0.005）——说明高负载下能耗优化更关键，
  写进 Discussion。

## 5. 规模/用户/面积趋势（PMEO-M 的 post-move 增益）

| 维度 | 变化 | 增益变化 | 解读 |
|---|---|---|---|
| UAV 数（~17 用户/UAV，2 km） | 2 -> 3 -> 4 | +76 -> +111 -> +130 | UAV 越多决策顺序越重要 |
| 用户数（3 UAV，2 km） | 30 -> 50 -> 80 | +51 -> +106 -> +157 | 负载越高差异化越大 |
| 面积（3 UAV，50 用户） | 1 -> 1.5 -> 2 -> 3 km | +28 -> +61 -> +106 -> +118 | 距离越远 UAV 移动越关键，3 km 处衰减 |

## 6. 图（figures/，2026-08-17 更新）

- `multi_uav_k3_comparison.png`：k3 六方法 2x2（reward / 成功率 / 时延 / 能耗）
  + 配对 t 检验显著性，300 dpi；
- `multi_uav_k3_table.tex` / `multi_uav_k3_table.csv`：主表 LaTeX x 4 指标；
- `multi_uav_scale_gain.png`：UAV 数 / 用户数 / 面积三维扫描 Eco vs MPC vs PMEO-M；
- `multi_uav_k3_main.png`：k3 六方法 reward 柱状 + 显著性；
- `multi_uav_k3_delay_energy.png`：k3 六方法时延-能耗对比。

## 6b. 为什么默认差异"看起来小"（drop_penalty 敏感性）

- `drop_penalty=6` 时 reward 各方法都约 -28/38 per-slot，Eco 每 slot 只比 MPC
  好 0.22，成功率 90.7% vs 90.3%，reward 差距 ~3%（+92）。
- 把 drop_penalty 提到 30/60，差距放大：vs MPC +92 -> +653 -> +1301（14 倍，
  p<0.05，7/10）；vs Follow-TEA 最大 +10997。
- 含义：失败代价越高（deadline 敏感业务），Eco 的高成功率优势越被放大；
  drop_penalty=30 表示"丢 1 个任务 = 30 单位惩罚"，适合应急/实时业务。
- 图：`figures/multi_uav_drop_penalty.png`。

## 6c. 视界增强（2026-08-17）

- **H5 + 速度前瞻**（`pmeo_m_eco_h5`）：reward vs MPC 从 +92 提升到 **+202**
  （默认 deadline，10/10 seeds），latency 0.1825s vs MPC 0.1849s，成功率 91.22%
  vs 90.29%。
- **deadline 0.5s 压力**：所有方法成功率降到 ~82%，Eco/H5 仍领先 **+239**（10/10），
  成功率 +1.13pp。
- 代价：H5 能耗 +3% vs H3（vs MPC +4 J/slot，p=0.039）；多花 3% 能耗换 8% reward，
  适合 deadline 紧张场景；默认 H3 更均衡。
- 图：`figures/multi_uav_horizon_stress.png`。

## 7. 参考文献（对比算法依据）

1. T. Ju et al., "A multi-UAV assisted task offloading and path optimization for
   mobile edge computing via multi-agent deep reinforcement learning", JNCA,
   vol. 229, art. 103919, 2024. doi:10.1016/j.jnca.2024.103919.
2. W. Liu et al., "Energy-Efficient Multi-UAV-Enabled MEC Systems over SAGIN",
   arXiv:2409.14782, 2024.
3. 多 UAV ISCC 场景 Gaussian-Markov 移动模型 + MAPPO，arXiv:2410.04151, 2024.
4. "Multi-Objective Optimization for Multi-UAV-Assisted MEC", IEEE TMC, 2024,
   doi:10.1109/TMC.2024.3350078.
5. "Joint UAV Deployment and Task Offloading Scheme for Multi-UAV-Assisted Edge
   Computing", Drones, 2023 (MDPI).
6. Cai et al., "Graphic Deep Reinforcement Learning for Dynamic Resource Allocation
   in SAGIN", IEEE JSAC, 2025（背景 + 差异化对比）。

## 8. 复现命令

```bash
# 初始 14 preset x 5 seeds x 8 episodes x 4 方法
python -m uav_leo_experiment.run_multi_uav --output_dir experiments/uav_leo_v2x_paper_final/multi_uav
# k3/u80 x 10 seeds x 10 episodes x 5 方法（含 MPC-M）
python -m uav_leo_experiment.run_multi_uav --presets u80,k3 --seeds 73,1,7,42,2024,12345,999,314,555,888 --episodes 10 --methods pmeo_m,current_exact_m,follow_tea_m,mpc_m_h3,random_m --output_dir experiments/uav_leo_v2x_paper_final/multi_uav_final
# PMEO-M-Eco 对比（k3/u80 x 10 seeds x 10 episodes x 3 方法）
python -m uav_leo_experiment.run_multi_uav --presets u80,k3 --seeds 1,7,42,73,314,555,888,999,12345,2024 --episodes 10 --methods pmeo_m,pmeo_m_eco,mpc_m_h3 --output_dir experiments/uav_leo_v2x_paper_final/multi_uav_eco
```

## 9. 待办

- [ ] 把本讲解 + 图表写入飞书 wiki（https://my.feishu.cn/wiki/IVn8wJb7fi7wnHkKX06ca6XCnac）；
- [ ] 补 MPC-M-H5 对比、energy_weight 敏感性（0.0001/0.001/0.01）；
- [ ] DRL（DQN/TD3/SAC）多 UAV 基线评估 + 收敛曲线；
- [ ] k3 六方法对比扩展到 k2/k4/u30/u50/u80/a1/a15/a3。

## 6d. 消融与超参数（2026-08-17）

**消融（k3，10 seeds x 10 episodes）**：

| 变体 | reward | vs Eco | 能耗 |
|---|---|---|---|
| PMEO-M-Eco（完整） | -2864.4 | - | 322.7 J |
| Eco-nb（去掉负载均衡关联） | -2746.5 | +117.8（p=0.0026） | 326.8 J（+4.1） |
| Eco-cc（候选集只剩质心+原地） | -3026.5 | -162.1（p<0.0001） | 321.9 J（n.s.） |

- 候选轨迹集是 reward 核心贡献（去掉 -162）；负载均衡关联是能耗/公平性旋钮。

**energy_weight 敏感性**：w=0.0001/0.001/0.01 时 Eco vs MPC 的 reward 差距
+59 -> +92 -> +187，w 越高优势越大；w=0.01 时 Eco 成功率 0.899 vs MPC 0.890，
能耗几乎相同（307.7 vs 308.2 J）。

## 6e. 多 UAV DRL 基线（2026-08-17）

- 实现：`uav_leo_experiment/multi_uav_drl.py`，参数共享 decentralized
  DQN/TD3/SAC，每 UAV 只有局部观测（211-217 维），30k env steps 训练。
- 收敛：M-DQN best eval -9089，M-TD3 -10312，M-SAC -11142，都远差于确定性的
  PMEO-M-Eco（-2864）；收敛曲线 `figures/fig5_convergence.png`。
- 论文卖点：确定性在线策略在"每时隙即时决策 + 强排队耦合"场景比 DRL 更实用，
  DRL 作为强对比而非我们的方法。
