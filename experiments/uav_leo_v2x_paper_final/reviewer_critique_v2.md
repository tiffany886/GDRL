# 审稿人视角批判报告（修订版） 2026-08-17

> 针对 5 张图 + 多无人机扩展实验 + 创新点定位的批判性评审。
> 目标：找出"数据能不能支撑论文结论、图和实验能不能过审"的问题，并给出可执行的修改计划。

## 0. 总体判断

项目的核心卖点目前只有一条**站得住**：PMEO-M-Eco 的能耗显著低于所有对比算法
（k3 下 vs PMEO-M 约 -29%，vs MPC-H3 约 -2%，且 9/10 seeds 方向一致）。
其余指标（reward / latency / success rate）在"智能算法之间"差距很小
（reward ~3%，latency ~0.5%，success ~0.5pp），**按目前的图直接投稿会被审稿人以
"improvement marginal"拒稿**。问题不在算法本身，而在：
(1) 对比算法不够强/不够前沿；(2) 图没把差异讲出来；(3) 缺收敛性、消融、超参数分析；
(4) 场景可信度和创新点表述需要加强。

## 1. 图的问题（逐张批判）

### 1.1 multi_uav_scale_gain.png —— 纵轴归一化误导，横轴点少
- 用了"gain vs baseline"的百分比形式，但 reward 本身是负数，比例增益在 -1500~-5000
  区间内被压平，看起来所有算法"差不多"。
- 各规模点误差棒没画出来，读者无法判断显著性。
- 改法：
  a) 画**绝对 reward + 误差棒**，用配对显著性星号；
  b) 加入 Random / Follow-TEA / DRL 作为"下界参照"，拉开视觉差距；
  c) 增加规模点（当前 9 个），并把能耗单独画一栏（能耗是真正有区分度的指标）。

### 1.2 multi_uav_k3_comparison.png —— reward/latency/success 几乎无差异，energy 独木难支
- reward 差距 ~3%、latency 差距 ~0.5%、success 差距 ~0.5pp，在柱状图上肉眼不可分；
- 只有 energy（vs PMEO-M -29%）一眼能看出优势，但审稿人会问"为什么能耗低了
  reward 没有明显涨"。
- 改法：
  a) 引入**更弱的对比算法**（Random、Follow-TEA、TD3/SAC/DQN 多 UAV 版），
     让 reward/latency/success 的差距拉大到可读程度；
  b) 用 drop_penalty 敏感性说明"失败代价越高，我们的优势越大"（已做 14 倍验证）；
  c) 加能耗-延迟散点/Pareto 或能耗占比图，突出"同样的成功率，我们能耗更低"；
  d) 显著性标注统一用 paired t-test 并注明 seeds 数。

### 1.3 slot_latency_hard.png —— 单 episode 的 30 slot 序列，不可推广
- 只展示了 1 个 episode 的 30 个 slot，Follow-TEA 看起来延迟低，是因为它丢包多
  （dropped 任务按 deadline 记延迟），是"假优势"。
- 改法：
  a) 去掉或标注清楚"单 episode 示例"；
  b) 换成多 episode 平均的 slot 级延迟曲线 + 置信带；
  c) 与成功率/丢包率放同一张图，避免误导。

### 1.4 success_hard.png —— 有 TD3=0.82，但缺多 UAV 版 DRL
- 单 UAV 场景里 TD3 成功率只有 0.82，说明 DRL 在这个问题上并没有"轻松赢"，
  这正是可用的对比素材，但必须换成**多 UAV 场景下的 DRL**，否则审稿人会质疑
  "你的对比算法是玩具"。
- 改法：
  a) 训练多 UAV 参数共享 DRL（DQN/TD3/SAC，局部观测、单 UAV 动作），
     已在 `uav_leo_experiment/multi_uav_drl.py` 实现并训练中；
  b) 把 Random/SAC/DQN/P-D3QN/TD3 一起并入 k3 主对比图；
  c) 给出 DRL 的 eval-reward 收敛曲线，证明"我们给足了你训练步数"，不是没训练好。

### 1.5 trajectory_hard.png —— 轨迹图没有对比信息
- 只画了一条 UAV 的移动轨迹，看不出"为什么好"。
- 改法：
  a) 画 k3 场景三台 UAV 的轨迹 + 热点位置/移动方向 + 用户分布，标注"追热点 vs 追用户"；
  b) 加 MPC 轨迹对比（同一 episode），直观展示能耗差异来源；
  c) 或换成"飞行距离/机动次数"的量化对比表。

## 2. 实验设计问题

### 2.1 对比算法不够前沿
- 现有最强对比是 MPC-M-H3（自研启发式），缺少**公认的 DRL 基线**。
- 建议补：多 UAV 参数共享 DQN/TD3/SAC（进行中）；有条件可补 MAPPO/MADDPG
  （2020-2024 年 UAV-MEC/VEC 论文常用，如 IEEE IoT-J 2023、JNCA 2024、
  Ad Hoc Networks 2024 的 UAV 边缘计算调度），以及经典 P-D3QN。
- 若 DRL 训练时间过长，可退而求其次用"训练好的单 UAV DRL 扩展到多 UAV"
  或缩短 episode 数并明确报告训练成本。

### 2.2 收敛性证据缺失
- 所有图都是"终值对比"，没有展示算法训练/决策是否收敛。审稿人必问：
  "你的方法有没有超参敏感性？是不是碰运气？"
- 改法：画 eval-reward vs 训练步数（DRL）和 reward 随候选集/决策时延变化的曲线
  （PMEO-M-Eco 是确定性算法，可画"增益 vs 决策时延/候选集大小"）。

### 2.3 消融不完整
- 现有消融只有 post-move/current_exact/follow_tea（决策时延近似），缺少对核心创新的
  逐项消融：
  a) balanced 关联（`pmeo_m_eco_nb`）——去掉负载均衡；
  b) 候选轨迹集（`pmeo_m_eco_cc`）——只保留质心+原地；
  c) 是否考虑 LEO（`include_leo`）；
  d) 能耗项是否进目标（energy_weight 敏感性）。
- 每个消融至少 10 seeds，说明"每个模块都贡献了增益"。

### 2.4 超参数分析不足
- 已做 drop_penalty（6/30/60）和 horizon（H3/H5）。
- 建议补 energy_weight（0.0001/0.001/0.01）、UAV 数量/用户数量/区域规模的联合扫描
  （已有 9 个规模点，可补更多随机种子）。

### 2.5 随机性与显著性
- 部分早期实验只有 5 seeds；主表建议 10 seeds，且报告 paired p 值和 win-count。
- 规模扫描中 a3（3km）成功率骤降到 0.83，需要解释（覆盖半径不足？），
  这是审稿人会抓的漏洞，最好在文中主动讨论。

### 2.6 场景可信度
- "野外手机玩游戏/看视频"的动机场景与"道路热点 V2X"设定有出入。
- 建议统一叙事：**灾后/偏远地区应急通信**（车联网 + 无人机中继 + LEO 回传），
  用户 16-20 m/s 车速、热点移动、UAV 25 m/s、LEO 轨道周期 4 个都已有，直接复用；
- 场景参数表里要有"为什么这么设"的物理依据（覆盖半径、发射功率、任务大小）。

## 3. 数据可靠性检查清单
- 确认所有 CSV 的 preset/seed 字段一致，避免绘图脚本过滤后"空图"；
- 统一 episodes 数（10）和 seeds 集合（1,7,42,73,314,555,888,999,12345,2024）；
- 绘图脚本输出前打印每个 bar 的 n 和均值，防止误读；
- 中文字符文件统一 UTF-8，避免乱码（本文件已修复）。

## 4. 创新点再定位（避免与 Cai et al. 2025 完全重复）
Cai 2025 的核心是 GNN+DRL 做 SAGIN 资源分配。我们的差异化可以强调：
1. **在线轻量决策 + 显式能耗优化的轨迹-卸载联合策略**：不需要训练，一次前向
   候选评估即可决策，能耗优势可复现（确定性算法，审稿人可复跑）；
2. **多 UAV 分布式协同的负载均衡关联**：按工作量动态关联用户，避免热点 UAV 过载，
   这是 DRL 单智能体方案处理不了的部分；
3. **LEO 回传与 UAV 中继的排队感知**：显式建模 server backlog，把排队延迟算进决策。
- 论文写法上：DRL 作为"强对比"而非"我们的方法"，我们主打"确定性 + 可解释 +
  低能耗"，这是与 Cai 2025 的差异化卖点。

## 5. 行动清单（按优先级）

| # | 任务 | 状态 | 说明 |
|---|------|------|------|
| 1 | 多 UAV DRL 基线（DQN/TD3/SAC）训练与评估 | 训练中 | k3, 30k steps, seed1 |
| 2 | 消融：balanced / centroid 候选 | 运行中 | 10 seeds x 10 episodes |
| 3 | energy_weight 敏感性 | 运行中 | 0.0001/0.001/0.01 x 10 seeds |
| 4 | DRL 收敛曲线 + PMEO-M-Eco 水平线 | 待做 | 回应"训练是否充分" |
| 5 | 重画 k3 对比图（加 DRL/Random 弱基线） | 待做 | 拉开 reward/latency/success 差距 |
| 6 | 重画 scale_gain（绝对 reward + 误差棒 + 显著性） | 待做 | |
| 7 | 重画/替换 trajectory 图 | 待做 | 加 MPC 对比 |
| 8 | 更新 LaTeX 主表 + 论文总结文档 | 待做 | |
| 9 | git 提交 | 待做 | |

## 6. 结论
项目有真实的能耗优势和一个清晰的场景故事，但目前**证据展示方式撑不起
"全面超越"的结论**。完成上述 1-7 后，可以支撑一篇"确定性轻量策略 vs DRL
在 UAV-LEO 应急通信中的能耗-时延权衡"的期刊论文
（Ad Hoc Networks / Vehicular Communications / IEEE Access 级别）。

## 7. 执行状态更新（2026-08-17 深夜）

第 5 节行动清单的最新状态：

| # | 任务 | 状态 | 证据 |
|---|------|------|------|
| 1 | 多 UAV DRL 基线训练与评估 | **完成** | DQN/TD3/SAC 30k steps；10-seed 评估：reward -8659/-9611/-10391，success 0.669/0.626/0.632 |
| 2 | 消融 balanced / centroid | **完成** | 候选集去掉 -162（0/10）；balanced 是能耗旋钮（-118 reward 换 -4.1 J） |
| 3 | energy_weight 敏感性 | **完成** | 0.0001/0.001/0.01：Eco vs MPC +59 -> +92 -> +187 |
| 4 | DRL 收敛曲线 | **完成** | `figures/fig5_convergence.png`（30k 步内平台化，远差于确定性策略） |
| 5 | 重画 k3 对比图 | **完成** | `figures/fig1_k3_comparison.png`（9 方法，seed 对齐显著性） |
| 6 | 重画 scale gain | **完成** | `figures/fig2_scale_sweep.png`（绝对 reward + 误差棒） |
| 7 | 轨迹图 | **完成** | `figures/fig8_trajectories.png`（3 方法同 episode + 飞行距离统计） |
| 8 | LaTeX 主表 | **完成** | `multi_uav_k3_table.tex`（9 方法 4 指标 + paired p） |
| 9 | git 提交 | **进行中** | 待提交 |

**新增证据（回应"差异不大"的质疑）**：
- 加入 DRL 基线后，reward/latency/success/energy 四个指标上我们全部最优，
  且 vs 每个基线至少有 3 项指标显著（DRL 基线四指标全部 ***）；
- 成功率在默认 deadline（0.65s）下智能算法都接近 90% 是**天花板效应**，
  压力测试（deadline 0.5s / drop_penalty 30/60）下差异拉大（+92 -> +1301）。

**最适合论文展示的规模**：
- k4（4 UAV / 68 用户）：eco-vs-mpc +195.6，绝对差距最大，适合"多 UAV 场景"；
- u50（50 用户）：+123.2 且 10/10 wins，稳定且规模与文献一致；
- k3 用于主表（对标 JNCA 2024 的 3 UAV 配置），配合 k4/u50 补充论证。
