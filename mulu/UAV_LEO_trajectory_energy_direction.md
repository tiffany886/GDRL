# 用户-UAV-卫星任务卸载与轨迹能耗方向整理

## 当前项目相对原论文的扩展

现在相对原本论文算法，项目主要增加了三类内容：

1. 能耗项：reward 中加入 energy penalty，结果指标中增加 `energy_mean`、`energy_total`、`energy_p95`。
2. 多类型节点：从原始 UE-LEO-HAPS，扩展到 hybrid：UE-gNB-UAV-LEO-MEC。
3. 任务部分卸载：从本地/全卸载，扩展为带 `offload_ratio` 的部分卸载。

## 相关论文方向

### 1. 最贴近老师建议的方向

**A Learning-Based Stochastic Game for Energy Efficient Optimization of UAV Trajectory and Task Offloading in Space/Aerial Edge Computing**, IEEE TVT 2025。

该方向包含 IoT 用户、UAV、LEO 卫星，联合优化 UAV 轨迹、任务卸载、带宽分配和能效。其目标是最大化 UAV 和 LEO 的长期能效，并允许 UAV 处理部分任务、其余任务转发到 LEO。

链接：https://doi.org/10.1109/TVT.2025.3540964

### 2. 部分卸载 + UAV 轨迹 + 能耗

**Joint optimization task offloading and trajectory control for UAV-assisted MEC**, 2023。

该工作没有卫星，但包含移动用户、障碍物、部分卸载、总能耗、总时延和 UAV 轨迹，用改进 DDPG 求解。适合参考部分卸载和轨迹能耗建模。

链接：https://www.sciencedirect.com/science/article/pii/S0045790623003403

### 3. UAV-MEC 能效 + 轨迹规划 + PPO

**Energy-efficient task offloading and trajectory planning in UAV-enabled MEC networks**, Computer Networks 2023。

该工作重点是最大化能效，联合优化用户发射功率、计算频率、UAV 功率、带宽和 UAV 轨迹，使用 EE-PPO。

链接：https://www.sciencedirect.com/science/article/pii/S1389128623003857

### 4. UAV + LEO 增强任务卸载

**A novel energy-efficient and cost-effective task offloading approach for UAV-enabled MEC with LEO enhancement**, 2024。

三层架构包括远程用户/IoRT、UAV-MEC、LEO。结果声称能耗降低 14.73%，时延降低 23.13%。

链接：https://www.sciencedirect.com/science/article/pii/S1569190X24001321

### 5. UAV 轨迹 + 任务卸载 DDPG

**Joint Trajectory Optimization and Task Offloading for UAV-Assisted MEC**, PIMRC 2023。

该工作用 DDPG 优化 UAV 飞行角度、速度、任务调度和发射功率。

链接：https://doi.org/10.1109/PIMRC56721.2023.10293959

## 公开源码情况

目前没有找到特别可靠、完整对应“用户-UAV-LEO-部分卸载-轨迹能耗”的公开源码。公开代码更多集中在两类：

1. UAV 轨迹/路径规划代码，偏机器人轨迹，不包含任务卸载。
2. UAV-MEC/任务卸载论文代码，通常不开源完整代码，或只给 MATLAB/CVX 优化脚本，不适合直接接入当前 Python GDRL 框架。

可参考代码资源：

- GCOPTER，UAV 轨迹优化通用代码：https://github.com/ZJU-FAST-Lab/GCOPTER
- quadrotor，UAV 路径规划/控制 MATLAB 代码：https://github.com/yrlu/quadrotor
- EnergyAwareMCPP，能耗感知多 UAV 覆盖路径规划：https://github.com/ctu-mrs/EnergyAwareMCPP
- UAV_Obstacle_Avoiding_DRL，UAV DRL 避障/轨迹学习，含 DDPG/TD3/PPO/SAC：https://github.com/ZYunfeii/UAV_Obstacle_Avoiding_DRL

这些代码不能直接复现实验，但可以参考 UAV 动作空间、轨迹约束和飞行能耗建模。

## 当前实验是否还合适

当前版本不是完全不合适，但复杂度加得太多，主线不够清楚。

目前同时加入了：

- 能耗
- gNB
- UAV
- LEO
- MEC
- 部分卸载
- GAT/GDRL
- SAC
- AMN 动作映射

这会导致核心创新点不够聚焦。老师容易追问：主要贡献到底是多节点异构网络、能耗建模、部分卸载、还是 UAV 轨迹优化？此外，当前 `hybrid_small_v2` 已经出现 AMN 决策塌缩，说明复杂 hybrid 系统还没有稳定。

更合适的方案是收缩为：

> 用户产生任务，可选择本地处理、卸载到 UAV，或经 UAV 转发到 LEO。UAV 具备移动轨迹和飞行能耗，算法联合优化任务部分卸载比例、UAV 轨迹、计算资源分配和能耗-时延权衡。

这样创新点更集中：

1. UAV 轨迹影响通信质量。
2. UAV 飞行能耗影响任务调度。
3. 部分任务可在 UAV 处理，剩余任务转发 LEO。
4. GDRL/SAC 用于联合连续-离散决策。

## 建议汇报口径

当前已完成能耗、部分卸载和多节点扩展，但多节点 hybrid 版本复杂度较高，训练中发现动作映射塌缩。根据老师建议，下一步准备将场景收缩为用户-UAV-LEO 三层模型，重点研究 UAV 轨迹能耗约束下的部分卸载问题。这个方向已有 TVT 2025 和 Computer Networks 2023 等相关工作支撑，问题更聚焦，也更容易形成可解释实验对比。

## 结论

继续堆 gNB/MEC 不如收缩到“用户-UAV-LEO”。该方向和老师建议一致，实验目标更明确，也更容易解释算法为什么有效。
