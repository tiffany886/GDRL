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

> 两阶段“轨迹规划 → 卸载决策”的分离思路参考了 [6][8]（UAV 轨迹优化与
> 卸载/资源分配联合设计的最新工作）；本文的差异点是强调**决策顺序**：
> 卸载必须在移动后的实际位置求解，而不是移动前。

## 2. 核心贡献（三个可写点）

### C1. 决策顺序（decision-order）发现（机制表 3b）
- 在**移动前**位置求解卸载（`current_exact`）会使用"不会真正服务该时隙"的旧位置，
  系统性高估链路速率 → 卸载决策次优。
- 实验：post-move 比 current-position 好 `+2.2`（hard）/ `+4.7`（stress），
  5 seed 池化配对 t 检验 p<0.0001。
- 热点消融（`routeA_analysis.md` 敏感性表）：把热点流量改为均匀到达后，post-move
  优势缩至 `+0.6`（-91.2 vs -91.8），且 MPC-H3 靠能耗节约反超（-89.9）→
  决策顺序收益由**热点集中流量**驱动，与命题 2 中"移动量/距离敏感度决定收益"一致；
  该消融也说明论文场景（热点）是 PMEO 贡献成立的必要条件。
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
  - [x] 系统架构图 + 算法伪代码（PMEO 两步流程）——见下方 Figure 1/Figure 2；
  - [ ] 命题 1/2 的完整证明与数值验证（不同移动量/路径损耗下的收益曲线）；
  - [ ] 英文全文（Introduction 强调"卸载决策顺序被长期忽视"；Related Work 覆盖
        UAV-MEC / 车联网 / LEO 边缘计算 2023-2025 文献；Experiments 用现有表格）；
  - [ ] 能量 Discussion 段落（第 3 节内容）；
  - [ ] 复现：一键 `run_routeA_all.bat`（主表 + 敏感性 + 多种子）。

![Figure 1: 系统架构图（UAV-LEO 边缘卸载场景，UAV 按专家轨迹移动，卸载可选 local/UAV/LEO）](figures/architecture.png)

![Figure 2: PMEO 两步流程（需求预测轨迹 → 移动后精确卸载）](figures/pmeo_pipeline.png)

## 6. 对比算法与参考文献映射

| 本文中的算法 | 论文中的角色 | 参考/来源 |
|---|---|---|
| PMEO（Post-Move Exact Offloading） | 本文提出方法（免训练） | 自研；轨迹-卸载分离思路参考 [6][8] |
| PMEO-E（能量门控变体） | 消融 / 负结果 | 自研（naive 单步能量门控） |
| TEA 系列（Predict-TEA / Follow-TEA / Current-pos exact） | 对比 / 消融基线 | 自研启发式（专家轨迹 + 精确卸载） |
| GDRL（residual PPO） | 本文训练版 / 主基线 | 源于 [7] 的图强化学习思想；轨迹部分参考 [6] |
| PPO (BC+KL) | 强化学习基线 | 标准 PPO（Schulman et al., 2017），行为克隆 + KL 约束 |
| GAT-PPO | 前沿基线 | GAT 编码参考 [3]（UAV-aided MEC 多智能体 DRL）；图结构卸载参考 [4] |
| Transformer-PPO | 前沿基线 | [5]（Transformer-based DRL offloading, VTC 2025-Spring） |
| P-D3QN | 前沿基线 | [1]（prioritized experience-based double dueling DQN）；ICV/UAV-MEC 辅助参考 [2] |
| D3QN 组件（dueling / double Q / PER） | 实现细节 | 标准技术，组合方式见 [1][2] |
| DQN / SAC / TD3 / DDQN | 标准基线 | 标准算法（Mnih et al.; Haarnoja et al.; Fujimoto et al.; van Hasselt et al.） |
| Random | 下界基线 | 随机动作 |
| MPC-H3 / H10 / H30 | 轨迹层对比 | [8]（MPC 轨迹规划 + 卸载优化）；不同视界消融为本实验扩展 |
| 综述 / 背景 | Related Work | [9] |

## 7. 多 UAV 扩展：PMEO-M 与 PMEO-M-Eco（2026-08-17）

把 PMEO 从单 UAV 扩展到 **K 台 UAV**：PMEO-M（multi-UAV）在每个时隙做两件事：

- **关联（association）**：按任务负载把用户均衡分配给 K 台 UAV（post-move association，
  基于移动后的实际位置重新计算链路速率）；
- **卸载目标扩展**：{local, UAV_1..UAV_K, LEO} × 比例网格，逐用户取最小
  `latency + energy_weight*energy + drop_penalty*dropped`。

**PMEO-M-Eco（能耗优化版）**：在 PMEO-M 基础上加 H=3 前瞻 + 负载均衡关联，
在 {原地, 专家移动, 任务质心, 负载均衡质心, LEO 附近} 的候选轨迹集中选择
"飞行能耗 + 卸载收益"综合最优的移动（每小时隙评估一次移动，其余时隙按负载
重新做卸载）。相比 PMEO-M（固定飞行能耗 443.6 J/slot），Eco 能省 30-130 J/slot，
接近 MPC 的飞行能耗，同时保留 post-move 卸载的决策顺序优势。

**多规模验证（10 seeds × 10 episodes，9 个规模点）**：PMEO-M-Eco 的 reward 在全部
9 个规模点都显著优于 MPC-M-H3（eco vs mpc = +39.5 ~ +195.6，paired p<0.05，
win-count 7/10 ~ 10/10），同时每时隙能耗低 1.5 ~ 9.2 J，计算时间约
24s vs 36.5s/episode（k3）。

**k3 主结果**：reward `-2864.4 ± 220` vs MPC `-2955.9`（p=0.034），
成功率 90.7% vs 90.3%，能耗 322.7 J vs 329.2 J；u80 规模 Eco +68.9（p=0.005，9/10）。

**机制解释（哪个模块贡献了增益）**

| 组件 | 作用 | 证据 |
|---|---|---|
| PMEO-M-Eco（H=3 前瞻 + 负载均衡关联） | 在保证卸载质量前提下降低飞行能耗 | 能耗低于 PMEO-M 约 30% |
| MPC-M-H3（多步轨迹前瞻） | 轨迹层更强基线 | 能耗与 Eco 接近，但 reward 略差 |
| PMEO-M（无 Eco 的固定轨迹） | 卸载决策顺序贡献 | reward 与 Eco 相当，但能耗高出 ~130 J |
| Current-pos exact（移动前求解卸载） | 决策顺序消融 | 比 post-move 差 +105.8（p<1e-5） |
### References

[1] J. Chi, X. Zhou, F. Xiao, Y. Lim, and T. Qiu, “Task Offloading via Prioritized Experience-Based Double Dueling DQN in Edge-Assisted IIoT,” *IEEE Transactions on Mobile Computing*, vol. 23, no. 12, pp. 14575-14591, Dec. 2024. DOI: 10.1109/TMC.2024.3452502.

[2] C. Li, K. Jiang, Y. Zhang, L. Jiang, Y. Luo, and S. Wan, “Deep Reinforcement Learning-based Mining Task Offloading Scheme for Intelligent Connected Vehicles in UAV-aided MEC,” *ACM Transactions on Design Automation of Electronic Systems*, vol. 29, no. 3, 2024. DOI: 10.1145/3653451.

[3] M. Kim, H. Lee, S. Hwang, M. Debbah, and I. Lee, “Cooperative Multiagent Deep Reinforcement Learning Methods for UAV-Aided Mobile Edge Computing Networks,” *IEEE Internet of Things Journal*, vol. 11, no. 23, pp. 38040-38053, Dec. 2024. DOI: 10.1109/JIOT.2024.3447090.

[4] I. Ullah and Y.-H. Han, “Optimizing vehicular edge computing: graph-based double-DQN approaches for intelligent task offloading,” *The Journal of Supercomputing*, vol. 81, no. 1, 2025. DOI: 10.1007/s11227-024-06599-4.

[5] Y. Xie, F. Zhang, Y. Fu, C. Xu, and T. Q. S. Quek, “Towards Task Number Adaptive Offloading in MEC Systems: A Transformer-based DRL Approach,” in *Proc. IEEE VTC 2025-Spring*, Oslo, Norway, 2025. DOI: 10.1109/VTC2025-Spring65109.2025.11174688.

[6] X. Wu, L. Liang, W. Wen, Z. Huang, X. Liu, and Y. Jia, “DRL-Based Trajectory Optimization and Computation-Aware Resource Allocation for UAV-Assisted Edge Computing Networks,” *IEEE Internet of Things Journal*, vol. 12, no. 20, pp. 43540-43558, Oct. 2025. DOI: 10.1109/JIOT.2025.3597502.

[7] Y. Cai, P. Cheng, Z. Chen, W. Xiang, B. Vucetic, and Y. Li, “Graphic Deep Reinforcement Learning for Dynamic Resource Allocation in Space-Air-Ground Integrated Networks,” *IEEE Journal on Selected Areas in Communications*, vol. 43, no. 1, pp. 334-349, Jan. 2025.

[8] Y. Zhang, Z. Kuang, Y. Feng, and F. Hou, “Task Offloading and Trajectory Optimization for Secure Communications in Dynamic User Multi-UAV MEC Systems,” *IEEE Transactions on Mobile Computing*, vol. 23, no. 12, pp. 14427-14440, 2024. DOI: 10.1109/TMC.2024.3442909.

[9] T. Baidya, A. Nabi, and S. Moh, “Trajectory-Aware Offloading Decision in UAV-Aided Edge Computing: A Comprehensive Survey,” *Sensors*, vol. 24, no. 6, art. 1837, 2024. DOI: 10.3390/s24061837.
