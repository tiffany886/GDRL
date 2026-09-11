# 学习篇：结果怎么说 · 对比图怎么讲

> 用途：对着实验图练「三句话结果口径」。配合主通读篇。
> 飞书本篇对应新建文档之一。
> Reward = 负代价，**越大越好**。

# 0. 先记住总口径（背下来）

1. **相对很多 DRL**：明显更好，而且**不用训练**。
2. **相对 Current-pos（旧位置卸载）**：更好 → 证明「顺序」有用。
3. **相对 MPC**：单机**接近**（约 98%–99% 量级，别吹全面第一）；**多机 Eco 往往更好**（主战场）。

禁令：

> 不要说「所有方法里全面第一」。单机上短视界 MPC 常略强或持平。

# 1. 相对 DRL：要不要「照着对比图说」？

**要。** 但不是瞎指条形图，而是固定话术 + 指对的图。

## 推荐怎么说（念这三句）

> 「这张对比里，PPO / DQN 系 / SAC / TD3 等要训练的方法，reward 明显更差。  
> 我们 PMEO / Eco 是免训练的规则飞 + 枚举卸。  
> 所以主张不是『又训了一个更强网络』，而是『把顺序做对，零训练也能打过常见 DRL』。」

## 单机：对着这张说「大胜 DRL、逼近 MPC」

![单机 hard 代价条形图](../experiments/uav_leo_v2x_paper_final/figures/reward_hard.png)

**图里看什么：** 这里若是「代价」则越低越好（= −reward）。DRL 一截较差；PMEO 与短视界 MPC、GDRL 挤在最优区。

**怎么讲：**

- 左边/较差那一截：Random、弱启发式、很多 DRL
- 最优区：PMEO、post-move、MPC-H3/H10、GDRL（GDRL≈专家+精确卸，学习增益≈0）

**别踩坑：** 不要指着最优区说「我们全面第一」；要说「我们在最优区，且零训练」。

## 多机：对着这张说「主结果」

![k3 多机四指标](../experiments/uav_leo_v2x_paper_final/figures/fig1_k3_comparison.png)

**图里看什么：** (a) reward (b) 成功率 (c) 时延 (d) 能耗。红色 Eco 往往最好或并列最好；多机 DRL 差一截。

**怎么讲：**

> 「多机是我们相对 MPC 也能赢的地方。Eco 在四个指标上压过要训练的多机 DRL，也压过短视界 MPC。」

# 2. 相对 Current-pos：讲创新点 1

![决策顺序增益](../experiments/uav_leo_v2x_paper_final/figures/scale_decision_order.png)

**固定话术：**

> 「同一条轨迹，只改一件事：卸载按旧位置算还是按飞完后位置算。  
> post-move 稳定更好，且区域越大增益越明显。  
> 这说明贡献不是换了一个更花的网络，而是决策顺序。」

数字备忘（修复版配对）：hard ≈ +2.5，stress ≈ +4.0，p 极小。

# 3. 相对 MPC：分单机 / 多机两套词

## 单机

> 「逼近短视界 MPC（约 98%–99% 量级），但零训练、每时隙更轻。  
> 不宣称单机全面超过 MPC。」

## 多机

![规模扫描 Eco vs MPC](../experiments/uav_leo_v2x_paper_final/figures/fig2_scale_sweep.png)

> 「多机 Eco 在多数规模点上 reward 优于 MPC；区域特别大时大家一起变难，差距会缩小——这是诚实边界。」

## 导师问「那为何不用 MPC」

备答：

- 多机：Eco 效果更好或持平，时隙计算也不更重（粗账 Eco 约 11 ms，MPC-M-H3 约 17 ms）
- 单机极致节能：能量权重大时 MPC 可能更省 → 产品上可「卸用 PMEO，轨迹可选 MPC」
- DRL：要训很久，训完仍常更差

# 4. 多机消融 / 能量权重（加分题）

![消融](../experiments/uav_leo_v2x_paper_final/figures/fig3_ablation.png)

讲法：去掉均衡关联或候选轨迹变窄，性能会变 → Eco 模块不是摆设。

![能量权重](../experiments/uav_leo_v2x_paper_final/figures/fig4_energy_weight.png)

讲法：主实验 $w_e$ 不大；调大后多机 Eco 仍站得住；单机高 $w_e$ 承认 MPC 更强。

# 5. LAETS 三张图（创新点 2，先混个脸熟）

若决策看数字孪生：状态太旧 → PMEO 顺序优势会被淹没。

![同步周期权衡](../digital_twin_uav_leo/results/fig_combined_k3_tau.png)

讲法：$\tau$ 从 1 放到 10，reward 掉很多。

![自适应支配固定周期](../digital_twin_uav_leo/results/fig_combined_k3_adaptive.png)

讲法：横轴同步次数、纵轴 reward；负载感知点在固定周期曲线上方。

![突发边界触发](../digital_twin_uav_leo/results/fig_sync_timing_burst.png)

讲法：粉区突发，红三角多在突发起止附近 → 该刷才刷。

# 6. 口头 8 分钟建议顺序

1. 场景图（三层 + 热点）
2. PMEO 流程图（规则飞 + 新位置卸）
3. **多机主对比图**（赢 DRL + 赢/平 MPC）
4. 顺序增益图（创新点 1）
5. 单机条形图（补充：大胜 DRL、逼近 MPC）
6. LAETS 一张（创新点 2 一句话）

# 7. 小作业（发我批改）

用自己的话各写 1～2 句：

1. 相对很多 DRL，你怎么说？（可提「对照图」）
2. 相对旧位置卸载，你怎么说？
3. 相对 MPC，单机怎么说、多机怎么说？
