# Route A — 论文重定位：PMEO（Post-Move Exact Offloading）

> 目标：把论文贡献从"图强化学习 GDRL"改写为"决策顺序敏感的轨迹-卸载联合优化"。
> 所有数字来自 `paper_results.md` 与 `routeA_analysis.md`（40 集 × 5 seed 评估）。

## 1. 方法

**PMEO（Post-Move Exact Offloading）**：免训练的确定性方法，每时隙两步：

1. **需求预测轨迹（expert move）**：UAV 朝"任务加权用户质心 + 一个时隙后的热点位置"的
   混合目标移动（速度上限内），即 demand-predictive 专家轨迹；
2. **移动后精确卸载（post-move exact offload）**：以**移动后的实际 UAV 位置**对每个用户
   在 {local, UAV, LEO} × 比例网格上取使
   `latency + energy_weight·energy + drop_penalty·dropped` 最小的卸载决策
   （时隙内逐用户可加，故逐用户 argmin 即全局最优）。

复杂度：每时隙 O(U × 15) 次链路/计算仿真，无训练、无前向规划。

## 2. 核心贡献（三个可写点）

### C1. 决策顺序（decision-order）发现（机制表 3b）
- 在**移动前**位置求解卸载（`current_exact`）会使用"不会真正服务该时隙"的旧位置，
  系统性高估链路速率 → 卸载决策次优。
- 实验：post-move 比 current-position 好 `+2.2`（hard）/ `+4.7`（stress），
  5 seed 池化配对 t 检验 p<0.0001。
- 论文命题（可证明）：
  - **命题 1（不劣性）**：对任意轨迹 {p_t}，若 O_pre 与 O_post 分别是时隙 t 在 p_{t-1}
    与 p_t 处求解的卸载决策，则 `cost(p_t, O_post) ≤ cost(p_t, O_pre)`。
    （O_pre 在同一可行域内，故恒成立。）
  - **命题 2（差距下界/上界）**：令速率函数对距离满足 Lipschitz 条件
    `|r(p)-r(p')| ≤ L_r·‖p-p'‖`，则决策顺序带来的收益
    `cost(p_t,O_pre) - cost(p_t,O_post) ≤ L_c·‖p_t - p_{t-1}‖`，
    其中 L_c 由路径损耗指数、任务大小与延迟/丢弃惩罚决定。
    即：移动量越大、链路对距离越敏感，post-move 收益越大——与热点场景实验结果一致。

### C2. 学习不必要（消融 3 与主表）
- GDRL（残差 PPO 训练 40k 步，~5 分钟 CPU）与免训练的 PMEO 结果几乎完全一致：
  hard `-85.2` vs `-85.2`，stress `-132.4` vs `-132.6`（配对 p=0.09/0.50，均不显著）。
- 消融：去掉轨迹学习（Δ≈0）、去掉精确卸载（Δ≈0）→ 学习组件无贡献。
- 结论/卖点：**在卸载层做对决策顺序，比任何学习都更有效**；PMEO 免训练、零部署成本。

### C3. MPC 长视界无益（主表）
- MPC-H3 `-85.6/-134.8`；MPC-H10 `-93.9/-147.8`；MPC-H30 `-131.7/-209.2`（均更差）。
- 原因：MPC 的候选轨迹固定 + 未来任务采样噪声随视界累积 → 长视界反而选择更差动作。
- 结论：对本文设定，**轨迹前瞻不是杠杆，决策顺序才是**；同时 PMEO 计算量比 MPC-H3
  低一个量级以上（无 lookahead、无采样）。

## 3. 诚实边界（写进论文 Discussion）

- **能量**：本论文 `energy_weight=0.001` 时能量几乎不影响目标，PMEO 与 MPC 的 reward
  打平但 PMEO 能耗更高（148 vs 124，纯飞行能耗差异）。把能量定价后（weight≥0.01），
  MPC 反超——说明**节能轨迹规划确实需要前瞻**。
- **PMEO-E（naive 能量门控）失败**：只在"收益>飞行能耗"时移动的 myopic 门控在
  任意能量权重下都远差于 MPC（hard -128.6 vs -85.6）→ 证明"节能不能靠单步门控"，
  需要多步前瞻。这个负结果反而强化 MPC 轨迹层的必要性。
- 因此论文最终方法建议表述为：**PMEO 负责卸载层（决策顺序），轨迹层可选
  （默认专家轨迹；能量敏感场景换 MPC 轨迹）**。

## 4. 主结果（paper_results.md，40 集 × 2 场景 + 5 seed）

| 方法 | hard | stress | vs PMEO (5-seed pooled p) |
|---|---|---|---|
| **PMEO (ours, 免训练)** | **-85.2** | **-132.4** | — |
| GDRL (residual PPO, 训练) | -85.2 | -132.6 | 0.09 / 0.50（不显著） |
| MPC-H3 | -85.6 | -134.8 | 0.0013 / 0.0001 |
| MPC-H10 / H30 | -93.9 / -131.7 | -147.8 / -209.2 | <0.01 / <0.0001 |
| Current-pos exact / Predict-TEA | -87.4 | -137.0 | <0.0001 |
| Follow-TEA | -93.2 | -140.1 | <0.0001 |
| PPO (BC+KL) / GAT-PPO / Transformer-PPO | -106.7 / -120.0 / -117.7 | -165.7 / -184.2 / -178.4 | <0.0001 |
| P-D3QN / DQN / SAC / TD3 / Random | -294 ~ -339 | -357 ~ -436 | <0.0001 |

## 5. 投稿路线与剩余工作

- 目标期刊（中档）：Ad Hoc Networks、Computer Networks、Physical Communication、
  IEEE Access、Vehicular Communications、EURASIP JWCN。
- P2（写作前必做）：
  - [ ] 系统架构图 + 算法伪代码（PMEO 两步流程）；
  - [ ] 命题 1/2 的完整证明与数值验证（不同移动量/路径损耗下的收益曲线）；
  - [ ] 英文全文（Introduction 强调"卸载决策顺序被长期忽视"；Related Work 覆盖
        UAV-MEC / 车联网 / LEO 边缘计算 2023-2025 文献；Experiments 用现有表格）；
  - [ ] 能量 Discussion 段落（第 3 节内容）；
  - [ ] 复现：一键 `run_routeA_all.bat`（主表 + 敏感性 + 多种子）。