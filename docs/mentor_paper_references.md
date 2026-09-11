# 导师汇报参考文献清单

更新时间：2026-09-11

## 1. 先给结论

当前最适合向导师汇报的是新的 **VA-DAG-HO** 方向：灾后场景、UAV-LEO 网络、硬覆盖、LEO 间歇可见、DAG 依赖任务和调度嵌入。建议不要把所有论文都讲一遍，而是选 5 篇组成一条清晰的研究脉络：

1. Liu et al., IEEE JSAC 2023：说明 DAG 依赖任务和 deadline violation 是重要问题。
2. Chen et al., IEEE TPDS 2024：说明 dependent/parallel tasks 的实时卸载已经进入高水平并行分布式系统研究。
3. Chen et al., IEEE TMC 2024：说明 UAV-LEO 边缘计算和多用户卸载是有价值的系统背景。
4. Li et al., IEEE TCOM 2023：说明 optimization-embedding 是与你当前“把结构化调度嵌入服务循环”最接近的方法思想。
5. Liu et al., IEEE COMST 2018：作为 SAGIN 总体背景和研究动机来源。

汇报时可以将 VA-DAG-HO 的定位说成：**已有工作分别研究了 SAGIN/UAV-LEO 卸载、DAG 任务调度和强化学习，但较少同时处理固定 UAV 巡航、硬覆盖、LEO 连续可见窗口以及物理可完成的 DAG 执行顺序；本文研究这个交叉约束下的调度问题。**

## 2. 最值得主讲的论文

以下分区按 LetPub 可查到的 **2025 年 3 月中科院升级版**记录；分区会随版本和学科口径变化，正式汇报时建议说“以最新学校认定版本为准”。引用数为 OpenAlex 在 2026-09-11 左右的检索值，只用于判断影响力，不作为正式评价依据。

| 优先级 | 文献 | 期刊/年份 | 分区或含金量 | 与当前工作的关系 |
|---|---|---|---|---|
| A1 | Liu et al., “Dependent Task Scheduling and Offloading for Minimizing Deadline Violation Ratio in Mobile Edge Computing” | IEEE JSAC, 2023 | 计算机科学 2 区；电信学 3 区；领域顶级期刊 | 直接支撑 DAG 依赖、截止期和 deadline violation 研究动机，可用来解释为什么不能把 DAG 当成独立任务包。 DOI: `10.1109/JSAC.2022.3233532` |
| A2 | Chen et al., “Real-Time Offloading for Dependent and Parallel Tasks in Cloud-Edge Environments Using Deep Reinforcement Learning” | IEEE TPDS, 2024 | 计算机科学 2 区；CCF-A 期刊；Top 口径较强 | 直接支撑 dependent/parallel task 的实时卸载建模，是当前论文最重要的 DAG 对照文献之一。 DOI: `10.1109/TPDS.2023.3349177` |
| A3 | Chen et al., “Multi-User Task Offloading in UAV-Assisted LEO Satellite Edge Computing: A Game-Theoretic Approach” | IEEE TMC, 2024 | 计算机科学 2 区；移动计算领域高水平期刊；Top 口径较强 | 直接支撑 UAV-LEO 多用户边缘卸载背景；可以用来对比本文的硬覆盖、连续可见窗口和 DAG 约束。 DOI: `10.1109/TMC.2024.3465591` |
| A4 | Li et al., “Computing Over the Sky: Joint UAV Trajectory and Task Offloading Scheme Based on Optimization-Embedding Multi-Agent Deep Reinforcement Learning” | IEEE TCOM, 2023 | 电子与电气/电信口径较强；计算机科学大类 3 区 | 与“优化模块嵌入 MARL/服务循环”的方法思想最接近，但它重点学习 UAV 轨迹；当前 VA-DAG-HO 则固定轨迹并把 DAG 调度嵌入 serving loop。 DOI: `10.1109/TCOMM.2023.3331029` |
| A5 | Liu et al., “Space-Air-Ground Integrated Network: A Survey” | IEEE Communications Surveys & Tutorials, 2018 | 计算机科学 1 区；高被引综述 | 支撑 SAGIN 的系统背景、空间-空中-地面协同和研究挑战。检索引用约 1275 次。 DOI: `10.1109/COMST.2018.2841996` |
| A6 | Huang et al., “Joint Offloading and Resource Allocation for Hybrid Cloud and Edge Computing in SAGINs: A Decision Assisted Hybrid Action Space Deep Reinforcement Learning Approach” | IEEE JSAC, 2024 | 计算机科学 2 区；Top 口径较强 | 支撑 SAGIN 中混合动作空间、卸载和资源分配问题；适合作为学习型基线和“连续/离散联合决策”对照。 DOI: `10.1109/JSAC.2024.3365899` |
| A7 | Ji et al., “Cooperative Multi-Agent Deep Reinforcement Learning for Computation Offloading in Digital Twin Satellite Edge Networks” | IEEE JSAC, 2023 | 计算机科学 2 区；Top 口径较强 | 可用于说明数字孪生卫星边缘网络的另一条研究路线，并突出当前工作刻意聚焦于硬覆盖、间歇可见和 DAG 物理可执行性。 DOI: `10.1109/JSAC.2023.3313595` |

## 3. 核心推荐论文逐篇拆解

下面的“创新点、建模场景、解决问题的方法”是根据 DOI 对应论文的公开摘要和题目整理的汇报版归纳，不是论文原文逐字翻译。正式汇报时可以用来理解论文逻辑，引用论文时仍应以原文为准。

### A1. Liu et al., IEEE JSAC, 2023

**中文标题：** 面向最小化移动边缘计算网络截止期违约率的依赖任务调度与卸载。

**创新点：**

- 将具有任务依赖关系的应用建模为有向无环图（DAG），研究在线任务到达和多种时延约束下的整体可靠性问题。
- 以截止期违约率（Deadline Violation Ratio，DVR）为核心目标，而不是只优化已完成任务的平均时延。
- 引入任务迁移与任务合并机制，并设计多优先级排序指标处理不同依赖关系和执行紧迫程度。
- 将组合调度问题交给基于 DDPG 的深度强化学习策略求解，公开摘要报告其相较基准方案的 DVR 改善。

**建模场景：** 多时隙 MEC 系统中，计算应用持续在线到达；每个应用由相互依赖的任务组成，未来到达信息未知。任务可以在边缘节点之间迁移或合并执行，同时受到计算资源、任务依赖和截止期约束。

**解决问题的方法：** 先利用 DAG 描述依赖关系，再通过任务迁移、任务合并和多优先级排序产生候选执行顺序，最后使用 DDPG 学习长期调度策略，目标是在有限时间范围内尽可能降低应用截止期违约率。

**对 VA-DAG-HO 的借鉴：** 支撑“任务成功率和 deadline 违约比单纯平均时延更重要”的论点；当前工作进一步将 UAV 硬覆盖和 LEO 连续可见窗口加入同一调度闭环。

### A2. Chen et al., IEEE TPDS, 2024

**中文标题：** 基于深度强化学习的云边环境依赖任务与并行任务实时卸载。

**创新点：**

- 提出依赖感知任务卸载方法 DODQ（Dependency-aware task Offloading with Deep Q-networks）。
- 将任务应用显式建模为 DAG，使算法能够处理任务依赖，而不是预先把任务调度顺序固定下来。
- 在 DQN 中加入对并行性的处理，使多个无前驱依赖的任务能够被同时考虑。
- 面向动态云边资源和实时决策需求，学习快速生成卸载决策，以适应资源状态变化。

**建模场景：** 移动设备产生计算密集型应用，任务可以上传到云或边缘服务器执行。云边节点的可用资源会动态变化，应用内部同时存在前驱依赖和可并行执行的任务。

**解决问题的方法：** 首先用 DAG 表示应用结构，维护任务依赖和并行关系；然后定制 DQN 作为卸载决策模型，在动态环境中学习任务放置方案，不再依赖人工预设的固定优先级或调度顺序。

**对 VA-DAG-HO 的借鉴：** 支撑 ready set、拓扑序和并行节点处理；当前工作将动作进一步约束为 UAV 本地、UAV-relay-LEO 或等待，并加入连续可见窗口校验。

### A3. Chen et al., IEEE TMC, 2024

**中文标题：** UAV 辅助 LEO 卫星边缘计算中的多用户任务卸载：一种博弈论方法。

**创新点：**

- 研究复杂地形下 UAV 与 LEO 协同提供广域覆盖时，多用户竞争有限卫星边缘资源的问题。
- 证明联合任务卸载优化问题是 NP-hard，并将其重构为 LUTO-Game 博弈模型。
- 分析 LUTO-Game 至少存在一个纳什均衡，为多用户自组织卸载提供理论基础。
- 提出联合 UAV-LEO 任务卸载（JULTO）算法，并分析其最坏情况下的策略性能。

**建模场景：** 多个移动用户设备处于 UAV 辅助的 LEO 卫星边缘计算网络中。用户通过 UAV 接入系统，共同竞争有限的 UAV/LEO 计算和通信资源，需要满足资源约束和时间约束。

**解决问题的方法：** 先将多用户联合卸载问题转化为非合作博弈，再基于博弈均衡设计 JULTO 联合策略，通过实验验证算法收敛性和性能。

**对 VA-DAG-HO 的借鉴：** 支撑 UAV-LEO 协同卸载的系统价值和资源竞争背景；当前工作不采用用户博弈，而是重点研究硬覆盖、间歇可见和 DAG 可执行顺序。

### A4. Li et al., IEEE TCOM, 2023

**中文标题：** 面向空中计算的基于优化嵌入多智能体深度强化学习的 UAV 轨迹与任务卸载联合设计。

**创新点：**

- 联合考虑 UAV 轨迹、任务卸载、能耗和服务公平性，而不是只优化单一卸载动作。
- 提出多智能体能效联合轨迹与计算卸载方案 MA-ETCO，使不同 UAV 根据服务需求自主作出控制决策。
- 提出优化嵌入多智能体深度强化学习算法 OMADRL：先利用优化模块处理混合整数非线性规划问题，再把优化结果作为学习策略的指导信息。
- 通过优化嵌入降低动作空间维度并改善训练收敛效率，解决纯强化学习在复杂混合动作空间中探索困难的问题。

**建模场景：** 多 UAV 辅助 MEC 网络服务多个用户，用户任务具有不同功能需求；UAV 电池容量有限，需要在移动位置、服务公平性、任务卸载和能耗之间进行联合权衡。

**解决问题的方法：** 将 UAV 轨迹和任务卸载联合建模为混合整数非线性优化问题；每个智能体根据需求进行决策，再利用优化结果生成引导指标，嵌入 MADRL 学习过程，最终联合输出轨迹控制和卸载策略。

**对 VA-DAG-HO 的借鉴：** 是与当前“结构化决策嵌入服务循环”最接近的思想来源；当前工作固定 UAV strip patrol，将研究变量收缩到 DAG 调度和 LEO 可见窗口，从而增强机制可解释性。

### A5. Liu et al., IEEE COMST, 2018

**中文标题：** 空天地一体化网络综述。

**创新点：**

- 从空间、空中和地面三个层次统一梳理 SAGIN，而不是只讨论某一个网络分段。
- 总结 SAGIN 的典型体系结构、协议设计、资源管理、性能分析和优化方法。
- 归纳异构性、自组织性、时变性以及三层资源不均衡带来的关键挑战。
- 给出未来技术方向，为后续 UAV-LEO、卫星边缘计算和跨层协同研究提供整体框架。

**建模场景：** 空间卫星系统、空中平台和地面通信网络组成的三层异构网络。各层具有不同的覆盖范围、链路特征、计算资源和时变性，需要共同承载通信和计算服务。

**解决问题的方法：** 这是一篇综述论文，不提出单一的卸载算法；其方法是建立 SAGIN 分类框架，按体系结构、协议、资源分配、性能分析和优化方向整理已有研究，并总结开放问题。

**对 VA-DAG-HO 的借鉴：** 用于说明为什么灾后场景需要 UAV 与 LEO 的协同架构；当前论文将综述中的宏观异构性具体化为覆盖失联、卫星可见窗口和 DAG 依赖三个可验证约束。

### A6. Huang et al., IEEE JSAC, 2024

**中文标题：** SAGIN 中混合云边计算的联合卸载与资源分配：一种基于决策辅助混合动作空间深度强化学习的方法。

**创新点：**

- 将卫星、云、UAV 和多接入边缘计算资源纳入统一的 SAGIN 混合云边场景。
- 将地面用户任务建模为 DAG，同时联合优化任务卸载和资源分配，以降低能耗和时延。
- 针对卸载选择中的离散动作与资源分配中的连续动作，设计混合动作空间 DRL 方法。
- 引入决策辅助机制减少不可用动作对训练过程的影响，提高学习过程的有效性。

**建模场景：** 多颗卫星、云服务器、UAV 和 MEC 服务器共同构成 SAGIN；地面用户产生 DAG 任务，并在多层计算基础设施之间进行任务卸载和资源分配。

**解决问题的方法：** 以能耗和时延为联合目标，将任务放置、卸载和资源分配统一为混合动作决策；通过决策辅助模块过滤或降低不可用动作的影响，再利用多智能体 DRL 学习联合策略。

**对 VA-DAG-HO 的借鉴：** 可作为“混合动作空间和 DAG 卸载”的高水平对照；当前工作不把所有资源分配交给黑盒 DRL，而是使用规则化的可见窗口剪枝和 DAG ready-set 保证动作物理可执行。

### A7. Ji et al., IEEE JSAC, 2023

**中文标题：** 数字孪生卫星边缘网络中的协作多智能体深度强化学习计算卸载。

**创新点：**

- 将数字孪生（Digital Twin，DT）引入卫星-地面协同网络，用数字孪生层辅助系统状态建模和决策。
- 研究地面用户任务在关联基站服务器、LEO 卫星和相邻服务器之间的部分卸载。
- 联合优化三层资源分配和任务拆分比例，目标是平衡系统总时延与能耗。
- 提出基于集中训练、分散执行（CTDE）的多智能体双延迟确定性策略梯度算法 MA-DATD3，以处理多卫星资源共享和协同决策。

**建模场景：** 数字孪生增强的卫星-地面协同网络中，地面用户通过基站接入，任务可以部分卸载到基站服务器、LEO 卫星或相邻服务器；多颗卫星之间共享有限资源。

**解决问题的方法：** 构造多层卸载优化问题，联合决定资源分配和任务拆分比例；采用 CTDE 训练多智能体策略，并用 MA-DATD3 处理卫星间的协作和资源竞争。

**对 VA-DAG-HO 的借鉴：** 可作为数字孪生和多智能体卸载路线的代表；当前工作明确不引入数字孪生层，而是聚焦灾后硬覆盖、LEO 间歇可见和 DAG 物理执行顺序，形成差异化边界。

## 4. UAV、LEO、SAGIN 系统背景

| 文献 | 期刊/年份 | 汇报用途 |
|---|---|---|
| Lin et al., “LEO Satellite and UAVs Assisted Mobile Edge Computing for Tactical Ad-Hoc Network: A Game Theory Approach” | IEEE IoT-J, 2023 | 说明 LEO 与 UAV 协同 MEC 的系统建模和资源竞争问题。 DOI: `10.1109/JIOT.2023.3299950` |
| Zhang et al., “Multiagent Reinforcement Learning-Based Orbital Edge Offloading in SAGIN Supporting Internet of Remote Things” | IEEE IoT-J, 2023 | 说明 SAGIN 中多智能体卸载的代表性路线。 DOI: `10.1109/JIOT.2023.3287737` |
| Zhao et al., “Multi-Agent Deep Reinforcement Learning for Task Offloading in UAV-Assisted Mobile Edge Computing” | IEEE TWC, 2022 | 说明 UAV-MEC 中 MARL 卸载的成熟基线，引用量较高；适合放在 related work，不建议作为当前论文唯一核心对照。 DOI: `10.1109/TWC.2022.3153316` |
| Hao et al., “Joint Task Offloading, Resource Allocation, and Trajectory Design for Multi-UAV Cooperative Edge Computing With Task Priority” | IEEE TMC, 2024 | 说明多 UAV、任务优先级、资源分配和轨迹联合设计；可用来解释当前工作为什么选择固定巡航以隔离调度问题。 DOI: `10.1109/TMC.2024.3350078` |
| McEnroe et al., “A Survey on the Convergence of Edge Computing and AI for UAVs: Opportunities and Challenges” | IEEE IoT-J, 2022 | 高质量综述，适合用于引言中的 UAV 边缘计算和 AI 研究趋势。 DOI: `10.1109/JIOT.2022.3176400` |

## 5. DAG、依赖和多跳卸载

| 文献 | 期刊/年份 | 与 VA-DAG-HO 的对应点 |
|---|---|---|
| Wang et al., “Dependent Task Offloading for Edge Computing based on Deep Reinforcement Learning” | IEEE Transactions on Computers, 2021 | 依赖任务与 DRL 卸载的直接参考；支持 ready set、前驱完成约束和依赖感知决策。 DOI: `10.1109/TC.2021.3131040` |
| Chai et al., “Joint Multi-Task Offloading and Resource Allocation for Mobile Edge Computing Systems in Satellite IoT” | IEEE TVT, 2023 | 连接卫星 IoT、任务卸载和资源分配；适合作为卫星边缘计算中的任务调度参考。 DOI: `10.1109/TVT.2023.3238771` |
| Sahni et al., “Multihop Offloading of Multiple DAG Tasks in Collaborative Edge Computing” | IEEE IoT-J, 2020 | 支撑 DAG 多跳卸载和协同边缘执行；与当前 UAV relay-LEO 链路有较好的概念对应。 DOI: `10.1109/JIOT.2020.3030926` |

## 6. 算法原理和实验基线

这些论文很有含金量，但主要是算法原文，不应把它们表述成“中科院一区期刊论文”。它们用于解释 MAPPO、MADDPG、TD3、SAC 为什么出现在方法或基线中。

| 文献 | 年份 | 用途 |
|---|---:|---|
| Yu et al., “The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games” | 2021 | MAPPO 的核心参考；当前实验中的 MAPPO/集中训练分散执行思路来源。 DOI: `10.48550/arXiv.2103.01955` |
| Lowe et al., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments” | 2017 | MADDPG 原始论文；多智能体集中式 critic 的经典来源。 DOI: `10.48550/arXiv.1706.02275` |
| Fujimoto et al., “Addressing Function Approximation Error in Actor-Critic Methods” | 2018 | TD3 原始论文；连续动作基线的经典来源。 DOI: `10.48550/arXiv.1802.09477` |
| Haarnoja et al., “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor” | 2018 | SAC 原始论文；连续动作、最大熵和 off-policy 基线的经典来源。 DOI: `10.48550/arXiv.1801.01290` |

## 7. 不建议作为“一区论文”介绍的条目

- IEEE Open Journal of the Communications Society：可作开放获取背景参考，但不建议向导师称为中科院一区。
- IEEE Internet of Things Journal：不同版本和学科口径变化明显；2025 年升级版在计算机科学大类显示为 1 区，但小类口径并不统一，汇报时应直接给出版本和学科，不要笼统说“绝对一区”。
- IEEE TVT、IEEE TCCN、IEEE Access：都是有价值的领域期刊，但不宜为了凑一区而统一包装成一区。
- arXiv 上的 MAPPO、MADDPG、TD3、SAC：方法影响力很高，但汇报时应称“经典算法原文/预印本或会议论文”，不要称为中科院一区期刊论文。

## 8. 给导师的 5 分钟汇报话术

可以按下面顺序讲：

> 第一类工作研究 UAV-LEO/SAGIN 中的卸载、资源分配和多智能体决策，例如 TMC、JSAC 和 IoT-J 的相关工作；第二类工作研究 DAG 或 dependent tasks 的执行顺序、截止期和实时卸载，例如 JSAC 和 TPDS 的工作。前者通常没有同时强调 DAG 的物理依赖和 LEO 的连续可见窗口，后者通常不处理灾后场景中的 UAV 硬覆盖和间歇连接。我的工作因此固定 UAV 巡航轨迹，把研究重点放在 coverage-aware association、DAG ready-node scheduling 和 visibility-window pruning 上，并用 success rate、effective latency、invalid LEO attempts 以及 DAG 顺序违规进行验证。

## 9. 汇报时的严谨表述

建议说：

> “分区按 2025 年 3 月中科院升级版查询，最终以学校和导师认可的最新分区表为准。”

不要说：

> “这篇一定是一区，所以我的工作也是一区水平。”

期刊分区只能说明发表平台和领域位置，导师更关心的是：问题是否真实、约束是否被正确建模、相对于最接近工作新增了什么、实验是否能证明新增机制确实有效。
