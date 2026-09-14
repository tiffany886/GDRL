# 面向 6G 数字孪生 SAGIN 的无线供能 UAV 定位量子机器学习
## （协作纳米卫星星座场景）

**Quantum Machine Learning for Wireless-Powered UAV Positioning in 6G Digital Twin SAGIN With Cooperative Nano-Satellite Constellations**

**作者**：Sasinda C. Prabhashana（学生会员，IEEE）、Minh-Hien T. Nguyen（会员，IEEE）、Vishal Sharma（高级会员，IEEE）、Thang X. Vu（高级会员，IEEE）、Berk Canberk（高级会员，IEEE）、Hyundong Shin（会士，IEEE）、Trung Q. Duong（会士，IEEE）

**出处**：IEEE Journal on Selected Areas in Communications（JSAC），Vol. 44，2026，pp. 5028–5042
**DOI**：10.1109/JSAC.2026.3700576

**作者单位与联系方式**：
- Sasinda C. Prabhashana：加拿大纽芬兰纪念大学工程与应用科学学院，St. John's, NL A1C 5S7（cwelhengodag@mun.ca）
- Minh-Hien T. Nguyen、Vishal Sharma：英国贝尔法斯特女王大学电子、电气工程与计算机科学学院，BT7 1NN Belfast（h.nguyen@qub.ac.uk；v.sharma@qub.ac.uk）
- Thang X. Vu：卢森堡大学安全、可靠与信任交叉研究中心，1855 Esch-sur-Alzette（thang.vu@uni.lu）
- Berk Canberk：英国爱丁堡龙比亚大学计算、工程与建筑环境学院，EH10 5DT Edinburgh（b.canberk@napier.ac.uk）
- Hyundong Shin：韩国庆熙大学电子与信息融合工程系，Gyeonggi-do 17104（hshin@khu.ac.kr）
- Trung Q. Duong（通讯作者）：加拿大纽芬兰纪念大学工程与应用科学学院；兼英国贝尔法斯特女王大学；兼韩国庆熙大学电子工程系（tduong@mun.ca）

**收稿与审稿轨迹**：2025 年 10 月 15 日收稿；2026 年 3 月 23 日修回；2026 年 5 月 30 日录用；2026 年 6 月 5 日在线发布；2026 年 6 月 17 日当前版本。

**资助信息**：本研究部分受加拿大卓越研究主席（CERC）计划资助（Grant CERC-2022-00109）；部分受加拿大自然科学与工程研究理事会（NSERC）发现基金资助（Grant RGPIN-2025-04941）；部分受 NSERC CREATE 计划资助（Grant 596205-2025）。Minh-Hien T. Nguyen 与 Vishal Sharma 的工作受英国 UKRI 与 Horizon Europe 计划下的 MISO 项目资助（"Autonomous Multi-Format In-Situ Observation Platform for Atmospheric Carbon Dioxide and Methane Monitoring in Permafrost & Wetlands"，Grant 10061165）。Thang X. Vu 的工作部分受卢森堡国家研究基金（FNR）资助（Grant FNR/C22/IS/17220888/RUTINE）。Hyundong Shin 的工作部分受韩国政府（MSIT）资助的韩国国家研究基金会（NRF）项目资助（Grant RS-2025-00556064）；部分受韩国科学技术信息通信部（MSIT）信息技术研究中心（ITRC）支持计划资助，由韩国信息通信技术规划与评价院（IITP）监管（Grant IITP-2025-RS-2021-II212046）。

---

## 摘要

能量高效的空天地一体化网络（SAGIN）对可持续通信至关重要。本研究提出一个能量感知的 SAGIN 框架：该框架采用由数字孪生技术增强的、搭载于无人机（UAV）上的移动边缘计算（MEC）平台，同时利用通过无线功率传输实现的 UAV 能量收集，以及一个带 MEC 设施的纳米卫星星座。我们把 UAV 轨迹规划、任务卸载、计算资源分配与卫星负载均衡的联合优化问题建模为一个混合整数非线性规划（MINLP）问题，目标是在满足能量与时延约束的前提下最小化加权系统代价。为求解这一复杂问题，我们提出了两种量子驱动的深度强化学习（QD-DRL）算法，即量子驱动的高性价比优势 Actor–Critic 算法（quantum-driven cost-effective advantage actor–critic, QD-CE-A2C）与量子驱动的高性价比近端策略优化算法（quantum-driven cost-effective proximal policy optimization, QD-CE-PPO）。这两种算法采用带可学习参数的角度编码与变分量子神经网络，以增强策略探索并加速收敛。仿真结果表明，所提出的 QD-DRL 方法具有更优的代价效率，并能确保在既定任务时长内为所有接入点提供有效服务。此外，与经典 DRL 基线相比，QD-DRL 方法获得了更高的累积回报与更快的收敛速度。因此，所提框架为未来 6G 使能的 SAGIN 中的高性价比资源管理提供了一种可扩展的智能化范式。

**关键词**——6G 网络、空天地一体化网络、卫星网络、数字孪生、量子神经网络、量子深度强化学习。

---

## I. 引言

过去几十年里，无线通信经历了持续的变革：从第一代模拟系统演进到第五代（5G）网络的高速数字时代 [1]。每一代技术都在数据传输、连接能力与服务质量方面带来了实质性的提升 [2]。到 5G 时代，关注点从以人为中心的通信转向了更广泛的应用，例如增强移动宽带、超可靠低时延通信（URLLC）以及大规模的机器间通信 [3]。这些能力为智慧城市、智能网联汽车、工业自动化与远程医疗等应用打开了大门 [2]。然而，5G 网络仍然存在若干不足：对于新兴的实时与沉浸式业务而言，其时延与吞吐仍显不足；对极端移动性的支持有限；全球覆盖范围有限 [4]。因此，为满足这些不断增长的需求并克服上述局限，学术界与产业界开始探索第六代（6G）无线网络的设计与架构。

此外，全球连接也是未来 6G 通信系统的关键目标之一 [5]。为此，6G 网络将以协同方式融合地面、空中、海上与卫星各组成部分 [6]。这种架构能够提供无缝覆盖，同时还能在偏远地区、深农村地带与高空空域之间实现实时数据交换 [4]。更进一步，这些能力是通过空天地一体化网络（SAGIN）实现的。SAGIN 作为核心架构，把地面与非地面基础设施汇聚到统一框架之中 [5]。它依赖卫星链路实现广域通信，依赖空中平台实现自适应中继与监测，依赖地面站实现高吞吐回传连接 [7]。这种多层框架增强了韧性，保障了数据的持续传输；此外，它还支撑态势感知，并使地理上分布的用户之间能够实时协同 [8]。因此，在分层架构中，既要管理能耗又要保证低时延，这一点至关重要。通过从网络内部收集能量，这类系统可以变得更加自维持、更加能量高效。无线功率传输（WPT）可以进一步支撑这一目标：无需外部电源即可为设备供电，从而使边缘节点能够收集能量并运行更长时间，而不必频繁更换电池 [9]。

近来，低地球轨道（LEO）卫星在全球通信网络中获得了极大关注。其密集部署可提供低时延、高容量的覆盖 [10]。然而，这种快速增长也带来了挑战：轨道面是有限的，同一轨道层内只能维持一定数量的卫星 [7]。因此，必须有效管理卫星资源以提升网络性能。此外，部署 LEO 卫星涉及高昂成本以及发射与维护方面的复杂挑战。作为一种切实可行的替代方案，立方星（CubeSat）已成为一种高性价比的解决方案。立方星是紧凑、标准化的卫星，可以大批量部署；其模块化设计与小巧体积使其非常适合快速验证、科学研究与通信网络扩展 [11]。特别地，作为立方星的一个子类，纳米卫星因其轻量化设计、可负担性与易部署性而受到关注；其可扩展性与对泛在连接的支持，使其非常适合现代天基网络 [11]。把移动边缘计算（MEC）引入卫星系统则带来了一次重大转变 [8]。MEC 使本地化的通信、计算与内容缓存成为可能，把边缘服务推向更靠近终端用户的位置，并降低对地面基础设施的依赖，从而减少回传流量、改善响应时间 [6]。这些改善对偏远地区时延敏感型应用至关重要。因此，卫星与 MEC 的融合能够在传统网络难以有效发挥作用的场景中，支撑快速、自适应且有韧性的服务 [8]。

进而，无人机（UAV）正成为 SAGIN 中不可或缺的一部分 [8]。它们可以在地面基础设施有限或不可用的区域提供覆盖。UAV 在高空运行，可与用户形成视距（LoS）通信，从而削弱地面障碍物造成的信号遮挡 [12]。此外，其机动性还允许动态重新部署位置以维持稳定连接 [13]。然而，UAV 在航程与能量方面受到约束，因此需要高效的轨迹规划来保证可靠链路与低时延 [14]。另一方面，数字孪生（DT）技术也被预期将在 6G 网络的发展中发挥关键作用 [15]。DT 是物理系统的高保真虚拟副本，通过持续的实时数据流与双向更新来维持 [16]。数字域与物理域之间的交互使预测性分析成为可能，同时也支撑更智能的决策与更高效的运行，在无线网络中尤其如此 [15]。

尽管深度强化学习（DRL）在无线网络资源分配中表现出显著性能，但它面临高维状态空间与动作空间带来的挑战 [14]；此外，它需要大量训练回合才能收敛。这些问题显著限制了它在大规模、异构 SAGIN 环境中的可扩展性 [8]。为缓解此类局限，量子计算（QC）成为一种有前景的解决方案。凭借利用并行性与指数级状态空间表示的能力，QC 能够更高效地处理复杂优化问题 [17]。与经典方法不同，QC 可以用更少的数据与训练回合取得显著的性能提升 [16]。特别地，量子线路允许把经典数据编码到量子态中，使 DRL 智能体能够探索更大的解空间，同时消耗更少的计算资源 [18]。

进一步而言，把 QC 融入 DRL 可以加速 DRL 智能体的学习能力。尽管有这些优势，当前 QC 的进展仍受硬件约束的限制。大多数实现依赖含噪中等规模量子（NISQ）设备，它们面临噪声敏感、量子比特数有限与退相干等问题，从而制约了大规模部署 [19]。虽然量子-经典混合学习在求解复杂优化与控制问题方面已展现出可观潜力，但在真实 NISQ 硬件上的实际实现仍然困难。现有量子设备受限于量子比特连通性受限、相干时间有限、门操作不完美与测量噪声，而这些都会影响参数化量子线路的可靠执行。此外，学习过程中估计期望值需要反复执行线路，这会加剧噪声与测量不确定性的影响 [20]、[21]。当线路深度增大时，量子线路的可训练性也会受到影响，因为优化会变得更困难、稳定的梯度更新更难维持。这些局限在强化学习场景中尤为突出，因为策略与价值的更新依赖反复且一致的线路执行。因此，基于仿真的模拟提供了一条切实可行的途径：在受控条件下研究量子-经典混合学习，同时把当前的 NISQ 局限纳入考虑。由此，为充分释放量子驱动学习框架的能力，推进量子驱动 DRL（QD-DRL）模型的研究至关重要。相应地，推进这一方向将有助于在未来无线通信网络中实现智能、可扩展且实时的资源管理 [15]、[17]、[18]、[19]、[21]。

### A. 相关工作

近年来，SAGIN 作为突破传统地面系统覆盖局限的一种切实手段而受到关注。通过把卫星、空中平台与地面网络连接起来，SAGIN 能够提供更大的服务范围与更可靠的连接 [4]、[5]、[7]、[8]、[10]。然而，如何以能量高效的方式对其进行管理仍是一项关键挑战。因此，这类网络可以设计为在网络内部收集能量。于是，WPT 可用于为边缘设备供电而无需外部电源，使网络更具自维持能力 [9]、[12]、[13]、[14]、[22]、[23]。文献 [22] 提出了一个 WPT 辅助的联邦学习框架：移动设备在本地训练期间收集射频能量以维持持续运行。该研究通过联合优化充电时长、计算分配与本地迭代次数，构建了总效用最大化问题，并采用改进的拉格朗日次梯度法高效求解，结果表明其收敛更快、效用高于基线模型。文献 [13] 提出了一种 UAV 使能的无线供能 MEC 网络，目标是在动态任务到达与信道变化条件下最大化长期计算速率；该问题采用探索增强型 DRL 方法求解，其中上层优化 UAV 轨迹，下层管理 WPT 与卸载。此外，文献 [12] 提出了一种支持无线携能通信（SWIPT）的 UAV 辅助 MEC 系统，通过联合优化 UAV 轨迹、波束成形与功率分割来最大化设备的最小剩余能量。文献 [9] 为无线供能 MEC 网络设计了一种在线部分卸载方案，把 Lyapunov 优化与 DRL 结合，以自适应调整 WPT 时长与卸载比例。特别地，文献 [14] 提出了一种全双工无线供能 MEC 架构以提升频谱效率与能量效率，该问题被分解为使用策略梯度 DRL 的卸载优化与使用凸方法的资源分配两部分，从而以低得多的复杂度取得了接近最优的计算速率。此外，文献 [23] 提出了一种 Lyapunov 引导的、基于卷积神经网络的 DRL 框架，用于提升无线供能 MEC 系统的长期能量效率；该方法把二元卸载决策与连续资源分配分离处理，并保证了队列稳定性。总体而言，这些研究表明：把 WPT 与能量收集机制融入计算与通信框架，能显著提升网络的可持续性。相应地，在 SAGIN 环境中，这类能量感知设计可使系统在空间、空中与地面各层之间更具韧性、更自供电、更具自适应性。这些特性共同为可持续的 6G 通信网络奠定了坚实基础。

随着通信网络的快速演进，为提升网络性能与效率，各种先进技术被相继提出。在这一背景下，把 QC 融入通信网络已成为一种具有变革意义的方法 [17]。例如，文献 [16] 为支持 6G DT 的 UAV 辅助海上网络提出了一种量子近端策略优化（quantum PPO）框架，目标是在能量与计算约束下最小化时延；所提方法相较传统对应方法取得了显著性能增益，在数据受限条件下尤为明显。文献 [24] 采用混合量子双深度 Q 学习（DDQN）模型来优化星地网络中的边缘云选择与带宽分配，该混合方法相比经典 DDQN 收敛更快、回报更高。此外，文献 [18] 采用量子 DRL 方法评估并增强无线网络的能量可持续性，把可持续性优化视为具有短期与长期目标的序贯决策过程，QD-DRL 方法相比传统 DRL 方法收敛更快、能量效率更高。这些研究表明，量子 DRL 框架在无线网络资源分配方面表现更优。文献 [19] 进一步引入了分层（layerwise）QD-DRL 框架用于 UAV 轨迹规划与资源分配的联合优化，该模型采用带局部损失的分层量子嵌入以提升可训练性并避免贫瘠高原（barren plateau），通过联合优化轨迹、用户分组与功率分配来最大化 UAV 能量效率。文献 [15] 将数字孪生辅助的车联网用于 URLLC 约束下的大规模任务卸载优化，其中 QD-DRL 框架与 LSTM 及 DT 集成以增强预测性决策；结果表明，QD-DRL 方法在效率与收敛性方面优于经典 DRL 方法。这些工作共同表明：量子计算增强了 DRL 智能体的决策能力。这种提升源自叠加与纠缠等量子力学原理，它们使智能体能够同时探索多个状态，并通过量子神经网络更高效地学习复杂模式。因此，QD-DRL 在大规模无线网络中能够获得更好的动作与结果，这些性质使其成为未来智能通信系统的一项有前景的进展 [15]、[16]、[17]、[18]、[19]、[21]、[24]。

此外，近期研究表明，量子增强的深度强化学习可以通过量子-经典混合架构，提升复杂优化与控制问题中的学习性能 [17]、[19]。在这些框架中，参数化量子线路常被用作策略或价值学习模块中紧凑的可训练函数逼近器。已有不同研究探索了量子数据编码、变分线路设计与混合 Actor-Critic 结构，以改进状态表示与学习效率。与此同时，这类方法的实际应用仍受当前 NISQ 局限的影响，包括量子比特资源受限、测量含噪、线路深度有限以及变分量子模型训练困难。因此，当前量子 DRL 研究主要聚焦于基于仿真或混合实现的形式，以便在受控设置下考察量子增强模型的学习行为 [25]。

### B. 动机与贡献

尽管 SAGIN 的发展近来取得了进展，现有资源管理框架在高度动态、异构的 6G 环境中运行时仍面临重大挑战。多层的共存、频繁的拓扑变化，以及严格的时延与能量要求，使传统优化方法与 DRL 方法难以有效扩展。为克服这些局限，本文提出一种用于 SAGIN 高效资源优化的 QD-DRL 框架。该框架利用量子叠加与纠缠来加速策略学习，并相较传统 DRL 方法提升决策效率。所提系统集成了具备双电池储能的 UAV、WPT 与 DT 支持，并与纳米卫星星座协作完成分布式计算。问题被建模为一个混合整数非线性规划（MINLP）问题，目标是在 URLLC 与能量因果约束下，通过联合优化 UAV 轨迹、任务卸载、计算功率分配与卫星负载分布来最小化总体系统代价。所构建的 MINLP 问题采用量子增强的 Actor-Critic 算法求解。本文的主要贡献可概括如下：

- 我们构建了一个 SAGIN 系统模型，集成了具备 MEC 能力的 UAV 与由主卫星协调的纳米卫星星座。UAV 从多个地面接入点（AP）收集受 URLLC 约束的数据，一部分在本地处理，并把计算密集型任务卸载到纳米卫星层。配备 WPT 的基站为 UAV 的能量收集提供支持，为任务传输与处理提供功率，以保证可持续运行。
- 我们把轨迹规划、任务调度、计算卸载与卫星负载分布的联合优化问题建模为 MINLP 问题，目标是在 SAGIN 框架内满足时延与能量约束的前提下最小化总体系统代价。
- 我们提出了两种 QD-DRL 算法，即量子驱动的高性价比近端策略优化（QD-CE-PPO）与量子驱动的高性价比优势 Actor-Critic（QD-CE-A2C），以高效地管理离散与连续决策变量，实现跨 SAGIN 各层的联合优化。
- 我们通过大量仿真对所提框架进行了评估。仿真结果表明，QD-DRL 方法优于传统 DRL：它们降低了系统代价、提升了能量效率并降低了时延。这些结果证明了 QD-DRL 在未来 6G SAGIN 网络中实现可扩展、智能化资源管理的潜力。

需要特别指出的是，本研究是在基于仿真的模拟设置下进行评估的，而非直接在物理量子硬件上执行。在实践中，未来的硬件实现将需要具备足够量子比特连通性、更长相干时间以及更低门操作与测量错误率的量子器件，才能支撑可靠的变分线路执行。在混合 DRL 场景中，这些要求尤为重要，因为训练期间策略与价值的更新需要反复执行线路。

### C. 论文结构与符号说明

本文结构安排如下。第 II 节描述所提系统模型与问题建模，包括通信信道建模、UAV 轨迹、数据处理与传输模型，以及基于 MINLP 的优化问题构建。第 III 节介绍为求解所构建 MINLP 优化问题而设计的 QD-DRL 方案。第 IV 节给出仿真结果与详细分析，以评估所提方法的有效性。第 V 节总结主要发现与未来研究方向。

**符号说明**：本文中小写字母表示标量，加粗小写字母表示向量。向量 x 的长度记为 |x|，复数集合记为 C。变量 x_{m,r}(t) 表示时隙 t 处与第 m 个发射端、第 r 个接收端相关联的取值。此外，记号 x ∼ CN(μ, σ²) 表示 x 服从均值为 μ、方差为 σ² 的复高斯分布，而 x ∼ N(μ, σ²) 表示服从均值为 μ、方差为 σ² 的正态分布。符号 (·)ᵀ 与 (·)ᴴ 分别表示转置与复共轭转置。此外，‖·‖ 表示向量的欧几里得范数。量子态之间的张量积用 ⊗ 表示。全文一致采用上述记号，以保证表述清晰。

---

## II. 系统模型

本文考虑一个面向 6G 的、带纳米卫星星座的 SAGIN，如图 1 所示。地面层由 M 个 AP 组成，记为 M = {1, 2, …, M}。每个 AP 从注册用户处收集数据包，并在 URLLC 约束下把它们传输给 UAV，由具备 MEC 能力的 UAV 执行计算。UAV 使用 K 根天线与各 AP 通信，并在地面层上空飞行，利用其主电池储能收集来自各 AP 的任务。然而，由于承担多功能运行，UAV 会产生可观的能量消耗，并在高峰时段面临任务请求溢出的重大挑战。因此，系统中使用一个配备 WPT 的基站（BS）为 UAV 的副电池储能供电。此外，UAV 把超额任务卸载到位于空间层的 LEO 纳米卫星网络。该网络是一个纳米卫星集群，

> **图 1**：面向任务关键型应用的、带纳米卫星星座的 6G 使能空天地一体化网络示意图。

该集群由一颗主纳米卫星（master nano-satellite, MNS）与四颗从纳米卫星（slave nano-satellite, SNS）组成，记为 I = {1, 2, 3, 4}。每颗 SNS 配备单天线与 MEC 能力。MNS 负责协调并管理各 SNS 之间的计算负载分配。当接收到来自 AP 的计算密集型、时延敏感数据后，UAV 把其中一部分数据传输给 MNS，MNS 再以负载均衡的方式把它分发给各 SNS 进行处理。

该网络采用三维（3D）笛卡尔坐标系建模。各 AP 位于地面，位置为 ℓ_m = (x_m, y_m, 0) ∈ R³（m ∈ M）；UAV 在固定高度 H 飞行，位置为 ℓ_u(t) = (x_u(t), y_u(t), H) ∈ R³。在空间层，MNS 的位置为 ℓ_ms(t) = (x_ms(t), y_ms(t), R) ∈ R³，其中 R 表示 LEO 轨道高度。四颗 SNS 排布在以 MNS 为中心的正方形四个角上，彼此等距。此外，配备 WPT 的 BS 位于 ℓ_bs = (x_bs, y_bs, h) ∈ R³，其中 h 为 BS 相对地面的高度。在该系统模型中，我们假设由于 LEO 卫星的密集部署而存在连续的卫星覆盖。此外，我们假设信道建模中的多普勒效应已在接收端得到补偿。

### A. 信道建模

**1）AP 与 UAV 之间的信道**：第 m 个 AP 与 UAV 之间在时刻 t 的信道向量可表示为 [3]

$$ \mathbf{g}_{m,u}(t) = \sqrt{h_{m,u}(t)}\,\tilde{\mathbf{g}}_{m,u}(t) \in \mathbb{C}^{K \times 1}, \tag{1} $$

其中 g̃_{m,u}(t) 为小尺度衰落，服从满足 E[|g̃_{m,u}(t)|²] = 1 的复随机变量。此外，h_{m,u}(t) 是大尺度信道系数，可按 [26] 计算为

$$ h_{m,u}(t) = \left(\frac{4\pi f_c d_{m,u}(t)}{c}\right)^{-\alpha} 10^{-\frac{\kappa^{\mathrm{los}}_{m,u}(t)\eta_{\mathrm{los}} + \kappa^{\mathrm{nlos}}_{m,u}(t)\eta_{\mathrm{nlos}}}{10}}, \tag{2} $$

其中 f_c 为载波频率，c 表示光速。此外，α 为路径损耗指数，d_{m,u}(t) 为第 m 个 AP 与 UAV 在时刻 t 的距离，可按 d_{m,u}(t) = ((x_u(t) − x_m)² + (y_u(t) − y_m)² + H²)^{1/2} 计算。进一步地，κ^los_{m,u}(t) 是 LoS 分量的概率，可计算为

$$ \kappa^{\mathrm{los}}_{m,u}(t) = \left(1 + \mu_1 \exp\left[-\mu_2\left(\arctan\left(\frac{H}{d^{\mathrm{hd}}_{m,u}(t)}\right) - \mu_1\right)\right]\right)^{-1}, \tag{3} $$

其中系数 μ₁ 与 μ₂ 取决于环境。d^hd_{m,u}(t) 是第 m 个 AP 与 UAV 之间的水平距离，可计算为 d^hd_{m,u}(t) = ((x_u(t) − x_m)² + (y_u(t) − y_m)²)^{1/2}。此外，NLoS 分量的概率可计算为 κ^nlos_{m,u}(t) = 1 − κ^los_{m,u}(t)。η_los 与 η_nlos 分别表示 LoS 与 NLoS 分量的额外路径损耗。

**2）UAV 到 MNS 之间的信道**：UAV 与 MNS 之间在时刻 t 的信道向量可表示为 [26]

$$ \mathbf{g}_{u,ms}(t) = \left[g_1(t)\ g_2(t)\ \ldots\ g_K(t)\right] \in \mathbb{C}^{1 \times K}. \tag{4} $$

此处每个分量 g_k(t) 由阴影莱斯（shadowed-Rician）衰落模型刻画，可表示为 g_k(t) = √(h_k(t)) · d_{u,ms}^{−α/2}(t)。其中 d_{u,ms}(t) 是 UAV 与 MNS 之间的距离，可按 d_{u,ms}(t) = ((x_ms(t) − x_u(t))² + (y_ms(t) − y_u(t))² + (R − H)²)^{1/2} 计算。进一步地，h_k(t) 服从阴影莱斯分布，可记为 h_k(t) ∼ SR(Ω_k, Δ_k, ϵ_k)，其中 Ω_k 是直射信号的平均功率，Δ_k 是散射多径分量的半平均功率，ϵ_k 表示 Nakagami-m 衰落参数。

**3）MNS 到 SNS 之间的信道**：MNS 与第 i 颗 SNS 之间在时刻 t 的信道可表示为 [26]

$$ g_{ms,ss(i)}(t) = \sqrt{h_{ms,ss(i)}(t)}\,\tilde{g}_{ms,ss(i)}(t) \in \mathbb{C}^{1 \times 1}, \tag{5} $$

其中 h_{ms,ss(i)}(t) 是大尺度路径损耗，可计算为 h_{ms,ss(i)}(t) = 10^{−κ_{ms,ss(i)}(t)/10}。此处 κ_{ms,ss(i)}(t) 是自由空间路径损耗，可计算为 κ_{ms,ss(i)}(t) = 10α log₁₀(4π f_c d_{ms,ss(i)}(t)/c)，其中 d_{ms,ss(i)}(t) 是 MNS 与第 i 颗 SNS 之间的距离，可计算为 d_{ms,ss(i)}(t) = ((x_ms(t) − x_ss(i)(t))² + (y_ms(t) − y_ss(i)(t))²)^{1/2}。此外，g̃_{ms,ss(i)}(t) 为小尺度衰落，可刻画为 g̃_{ms,ss(i)}(t) ∼ CN(0, 1)。

### B. 通信建模

**1）AP 与 UAV 之间的通信**：来自各 AP 的数据由 UAV 接收。在 URLLC 约束下，第 m 个 AP 与 UAV 之间的数据传输速率可计算为 [16]

$$ R^u_m(t) \approx B\left(\log_2\left(1 + \gamma^u_m(t)\right) - \sqrt{\frac{V^u_m(t)}{N}}\,\frac{Q^{-1}(\epsilon^u_m)}{\ln 2}\right), \tag{6} $$

其中 B 为系统带宽，N 为块长度，参数 ϵ^u_m 对应译码错误概率。函数 Q⁻¹(·) 指函数 Q 的反函数，Q(x) = (1/√(2π)) ∫_x^∞ exp(−t²/2) dt。V^u_m(t) 表示信道色散，由 V^u_m(t) = 1 − [1 + γ^u_m(t)]^{−2} 给出，其中 γ^u_m(t) 是信噪比，可计算为 γ^u_m(t) = p_m ‖g_{m,u}(t)‖² / σ²_u(t)，其中 p_m 表示第 m 个 AP 的发射功率，σ_u(t) 是瞬时噪声功率，由复高斯分布 ∼ CN(0, σ²) 刻画 [3]。

**2）UAV 与 MNS 之间的通信**：UAV 到 MNS 的数据速率可计算为

$$ R^{ms}_u(t) = B \log_2\left(1 + \frac{p_u \|\mathbf{g}_{u,ms}(t)\|^2}{\sigma^2_{ms}(t)}\right), \tag{7} $$

其中 p_u 是 UAV 的发射功率。此外，σ_ms(t) 是瞬时噪声功率，服从复高斯分布 ∼ CN(0, σ²) [10]。

**3）MNS 与 SNS 之间的通信**：MNS 到第 i 颗 SNS 的数据速率可表示为

$$ R^{ss(i)}_{ms}(t) = B \log_2\left(1 + \frac{p_{ms,ss(i)} |g_{ms,ss(i)}(t)|^2}{\sigma^2_{ss(i)}(t)}\right), \tag{8} $$

其中 p_{ms,ss(i)} 是 MNS 到第 i 颗 SNS 的发射功率。σ_ss(i)(t) 是瞬时噪声功率，由复高斯分布 ∼ CN(0, σ²) 刻画 [10]。

### C. UAV 的移动性

为对 UAV 的移动性建模，我们把它总的飞行时间划分为 T 个离散时隙。每个时隙的时长为 τ_t，时隙集合记为 T = {1, 2, …, T}。此外，UAV 在固定高度 H 飞行。在时刻 t，UAV 的运动由沿 x 轴与 y 轴的位移分量 d_x(t) 与 d_y(t) 决定。因此，UAV 从时隙 t 到时隙 t + 1 的位置可计算为 [13]

$$ x_u(t+1) = x_u(t) + d_x(t) + \Delta x_u(t), \tag{9} $$
$$ y_u(t+1) = y_u(t) + d_y(t) + \Delta y_u(t), \tag{10} $$

其中 Δx_u(t) 与 Δy_u(t) 是由风引起的微小随机扰动，建模为 ∼ N(0, σ²)。UAV 在时刻 t 的瞬时速度 v_u(t) 由 v_u(t) = d_xy(t)/τ_t 给出，其中 d_xy(t) = (d_x²(t) + d_y²(t))^{1/2}。相应地，UAV 的功率消耗可表示为

$$ P(v_u(t)) = \chi_1\left(1 + \frac{3v_u(t)^2}{w^2_{\mathrm{tip}}}\right) + \chi_2\left(\sqrt{1 + \frac{v_u(t)^4}{4w_0^4}} - \frac{v_u(t)^2}{2w_0^2}\right)^{1/2} + \frac{1}{2}\tau\rho\Upsilon A v_u^3(t), \tag{11} $$

其中 A 是旋翼盘面积，Υ 表示旋翼实度，ρ 表示空气密度，τ 是机身阻力系数。此外，w₀ 表示悬停时旋翼诱导的平均气流速度，w_tip 表示旋翼桨叶的叶尖速度。系数 χ₁ 与 χ₂ 分别表示桨叶剖面功率与悬停所需的诱导功率。因此，每时隙的总能量消耗可计算为 E^fly_u(t) = P(v_u(t))τ_t [17]。整个飞行时长 T 内所需的飞行能量由 UAV 的主电池储能提供。因此，主电池能量被专门用于覆盖飞行能耗。于是，主电池储能更新可表示为

$$ E^{\mathrm{batt}}_u(t+1) = E^{\mathrm{batt}}_u(t) - E^{\mathrm{fly}}_u(t), \tag{12} $$

其中 E^batt_u(t) 是时刻 t 的电池能量状态，E^fly_u(t) 是 UAV 在时隙 t 内的飞行能耗。

### D. 任务接纳、传输与处理模型

我们考虑每个 AP 在时刻 t 从与该 AP 关联的注册用户处收集数据。收集到的数据被组织成一个数据包（即任务），可表示为 J_m = {D_m, Q_m, T^max_m}，其中 D_m 表示数据量大小，Q_m 表示计算复杂度，T^max_m 表示最大时延要求。为对任务接纳建模，我们定义一个二元变量 π_m(t) ∈ {0, 1}，表示 UAV 是否接纳来自第 m 个 AP 的任务。只有当 AP 处于 UAV 的通信范围内（即 d^hd_{m,u}(t) < d_max）时，任务才会被传输。当多个 AP 同时处于 UAV 覆盖范围内时，UAV 在当前时隙选择其中一个 AP 进行接纳，其余 AP 可在后续时隙被接纳。被接收的任务随后存入 UAV 处的队列 q(t)，并调度到后续时隙处理。相应地，时刻 t 从第 m 个 AP 传输一个任务的能量消耗可表示为 E^trans_{m,u}(t) = π_m(t) p_m D_m / R^u_m(t)。此外，UAV 在本地处理任务的一部分，并把其余部分卸载到 MNS。令 δ_m(t) ∈ [0, 1] 表示决定卸载到 MNS 的比例的连续变量。因此，UAV 处的任务处理能量可计算为

$$ E^{\mathrm{proc}}_{m,u}(t) = \xi_u (1-\delta_m(t)) Q_m \left(f^u_m(t) - \hat{f}^u_m(t)\right)^2, \tag{13} $$

其中 ξ_u 是 UAV 处理器的能量系数，取决于 CMOS 电路。进一步地，我们定义 UAV 处的 DT 服务为 DT^u_m。该服务可表示为 DT^u_m = {f^u_m(t), f̂^u_m(t)}，其中 f^u_m(t) 与 f̂^u_m(t) 分别表示 UAV 在其 DT 中估计的处理速率以及处理速率偏差值。于是，时刻 t 把任务的剩余部分从 UAV 卸载到 MNS 的能量消耗可表示为 E^trans_{u,ms}(t) = p_u δ_m(t) D_m / R^ms_u(t)。与此同时，MNS 使用负载均衡变量 θ_{m,i}(t) ∈ [0, 1] 把接收到的比例分发给各 SNS。因此，时刻 t 把任务的卸载部分从 MNS 传输到第 i 颗 SNS 的能量消耗可计算为

$$ E^{\mathrm{trans}}_{ms,ss(i)}(t) = \theta_{m,i}(t)\, p_{ms,ss(i)}\, \delta_m(t)\, \frac{D_m}{R^{ss(i)}_{ms}(t)}. \tag{14} $$

进一步地，时刻 t 第 i 颗 SNS 处对接收任务部分的处理能耗可表示为 E^proc_{ss(i)}(t) = θ_{m,i}(t) ξ_{ss(i)} Q_m δ_m(t) (f^{ss(i)})²，其中 ξ_{ss(i)} 与 f^{ss(i)} 分别是第 i 颗 SNS 的能量系数与计算频率。因此，时刻 t 空间层的总能耗可计算为 E^total_{ms,ss}(t) = Σ_{i=1}^{4} (E^trans_{ms,ss(i)}(t) + E^proc_{ss(i)}(t))。于是，时刻 t 第 m 个 AP 的任务跨全部 SAGIN 层的总能耗可表示为 E^tot_m(t) = E^trans_{m,u}(t) + E^proc_{m,u}(t) + E^trans_{u,ms}(t) + E^total_{ms,ss}(t)。

### E. 时延模型

当 UAV 接纳来自第 m 个 AP 的任务时，时刻 t 从第 m 个 AP 到 UAV 的传输时延可表示为 T^trans_{m,u}(t) = π_m(t) D_m / R^u_m(t)。进一步地，时刻 t UAV 处理第 m 个 AP 任务的时延可计算为

$$ T^{\mathrm{proc}}_{m,u}(t) = (1-\delta_m(t)) \frac{Q_m}{\left(f^u_m(t) - \hat{f}^u_m(t)\right)}. \tag{15} $$

与此同时，时刻 t 从 UAV 到 MNS 的卸载部分传输时延可表示为 T^trans_{u,ms}(t) = δ_m(t) D_m / R^ms_u(t)。一旦 MNS 接收到来自 UAV 的卸载任务，它就把该任务部分分配给所有 SNS。因此，时刻 t 从 MNS 到第 i 颗 SNS 的传输时延可计算为 T^trans_{ms,ss(i)}(t) = θ_{m,i}(t) δ_m(t) D_m / R^{ss(i)}_{ms}(t)。此外，时刻 t 第 i 颗 SNS 处对应的处理时延可表示为 T^proc_{ss(i)}(t) = θ_{m,i}(t) δ_m(t) Q_m / f^{ss(i)}。相应地，时刻 t 第 m 个 AP 任务的总时延可计算为 T^total_m(t) = T^trans_{m,u}(t) + T^proc_{m,u}(t) + T^trans_{u,ms}(t) + max_{i∈I}(T^trans_{ms,ss(i)}(t) + T^proc_{ss(i)}(t))。

### F. 无线功率传输模型

本文考虑 UAV 通过 WPT 收集 BS 发射的射频能量。所收集的功率用于 UAV 处的任务处理以及向 MNS 的任务传输。为对无线功率传输过程建模，设时刻 t BS 与 UAV 之间的信道增益记为 h_bs,u(t) ∈ C^{1×1}。于是，信道增益可建模为 [22]

$$ h_{bs,u}(t) = G_{bs} G_u h_0 d^{-2}_{bs,u}(t), \tag{16} $$

其中 h₀ 是参考距离处的信道功率增益。d_bs,u(t) 是 BS 与 UAV 之间的距离，可计算为 d_bs,u(t) = ((x_u(t) − x_bs(t))² + (y_u(t) − y_bs(t))² + (H − h)²)^{1/2}。此处，UAV 采用线性能量收集模型，把接收到的射频能量转换为电能以为电池充电。此外，G_bs 与 G_u 分别是 BS 与 UAV 的天线增益。那么，UAV 在时隙 t 收集到的能量可计算为 E^hav_u(t) = η_u P_bs h_bs,u(t) τ_t，其中 η_u ∈ (0, 1] 是 UAV 的能量收集效率，取决于其射频到直流的转换电路。P_bs 是 BS 到 UAV 的恒定发射功率 [22]。这些收集到的能量存储在 UAV 携带的独立副电池储能中。于是，收集能量的储能更新可定义为 E^hav_stored(t+1) = E^hav_stored(t) + E^hav_u(t) − Σ_{m=1}^{M} (E^proc_{m,u}(t) + E^trans_{u,ms}(t))，其中 E^hav_stored(t) 是时刻 t 副电池储能中可用的收集能量，E^hav_u(t) 是 UAV 在时刻 t 收集到的能量。

### G. 问题建模

本文通过联合优化 UAV 轨迹、任务卸载决策、卫星负载分布与 UAV 计算功率分配，在 T 个时隙内最小化 SAGIN 各层的加权系统代价。加权系统代价综合了 UAV 飞行能量、与任务相关的能量消耗以及任务时延。该优化确保来自各 AP 的所有任务都能在 UAV 的轨迹时间 T 内完成。相应地，目标函数可表示为

$$ f(\Omega) = \sum_{t=1}^{T} w_1 E^{\mathrm{fly}}_u(t) + \sum_{t=1}^{T}\sum_{m=1}^{M} \pi_m(t)\left(w_2 E^{\mathrm{total}}_m(t) + w_3 T^{\mathrm{total}}_m(t)\right), \tag{17} $$

其中 Ω ≜ {π, δ, θ, d, f^u}，且 π ≜ {π_m(t)}_{∀m,t}、δ ≜ {δ_m(t)}_{∀m,t}、θ ≜ {θ_{m,i}(t)}_{∀m,i,t}、d ≜ {d_x(t), d_y(t)}_{∀t}、f^u ≜ {f^u_m(t)}_{∀m,t}。w₁、w₂、w₃ 是加权因子。于是，优化问题可建模如下：

$$
\begin{aligned}
(\mathrm{P1}):\quad \min_{\{\pi,\delta,\theta,d,f^u\}}\ & f(\Omega) \tag{18a}\\
\text{s.t.}\quad & 0 \le \delta_m(t) \le 1, & \forall m,t, \tag{18b}\\
& \pi_m(t) \in \{0,1\}, & \forall m,t, \tag{18c}\\
& T^{\mathrm{total}}_m(t) \le T^{\max}_m,\ \text{当}\ \pi_m(t)=1, & \forall m,t, \tag{18d}\\
& 0 \le d_{xy}(t) \le v_{\max}\tau_t, & \forall t, \tag{18e}\\
& 0 \le x_u(t) \le x_{\max},\ 0 \le y_u(t) \le y_{\max}, & \forall t, \tag{18f}\\
& \pi_m(t) d^{\mathrm{hd}}_{m,u}(t) \le d_{\max}, & \forall m,t, \tag{18g}\\
& \sum_{i=1}^{4}\theta_{m,i}(t) = 1, & \forall m,t, \tag{18h}\\
& \sum_{t=1}^{T}\pi_m(t) \le 1, & \forall m, \tag{18i}\\
& 0 \le \theta_{m,i}(t) \le 1,\ i\in I, & \forall m,t, \tag{18j}\\
& R^u_m(t), R^{ms}_u(t), R^{ss(i)}_{ms}(t) \ge R_{\min}, & \forall m,t, \tag{18k}\\
& \sum_{t=1}^{T} E^{\mathrm{fly}}_u(t) \le E^{\mathrm{batt}}_u, & \tag{18l}\\
& f^u_m(t) - \hat{f}^u_m(t) \ge 0, & \forall m,t, \tag{18m}\\
& 0 \le f^u_m(t) \le f^{\max}, & \forall m,t, \tag{18n}\\
& E^{\mathrm{batt}}_u(t+1) \ge 0, & \forall t \in T, \tag{18o}\\
& \sum_{m=1}^{M}\left(E^{\mathrm{proc}}_{m,u}(t) + E^{\mathrm{trans}}_{u,ms}(t)\right) \le E^{\mathrm{hav}}_{\mathrm{stored}}(t) + E^{\mathrm{hav}}_u(t), & \forall t, \tag{18p}\\
& 0 \le E^{\mathrm{hav}}_{\mathrm{stored}}(t+1) \le E^{\mathrm{hav},\max}_{\mathrm{stored}}, & \forall t. \tag{18q}
\end{aligned}
$$

如式 (18) 所规定，(18a) 中的目标函数通过联合优化 UAV 轨迹、任务卸载、卫星负载分布与 UAV 计算功率，同时计入 UAV 飞行能量、任务能量消耗与任务时延，来最小化总体系统代价。相应地，约束 (18b)–(18d) 分别规范任务卸载比例、强制任务接纳的二元决策，并确保时延要求保持在最大可容忍范围内。进一步地，(18e) 依据最大速度限制 UAV 每时隙的位移，而 (18f) 把 UAV 轨迹限制在指定的运行区域内。此外，(18g) 保证只要任务被接纳，AP 到 UAV 就存在连接，而 (18h) 确保卸载到各 SNS 的工作负载被完整地分配到各可用节点上。相应地，约束 (18i) 确保每个 AP 在任务时长内至多被接纳一次。类似地，(18j) 把负载分配变量限制在可行取值范围内。此外，(18k) 强制各通信链路满足最小数据速率要求，从而确保 URLLC 约束下的可靠传输。约束 (18l) 将累计飞行能量限制在 UAV 电池容量之内，从而维持长期能量可行性。进一步地，(18m) 保证有效处理速率保持非负，而 (18n) 确保分配给 UAV 的 CPU 频率不超过其最大计算能力。约束 (18o) 保证 UAV 主电池状态在每个时隙的非负性。此外，(18p) 确保 UAV 为服务所有 AP 而用于任务处理与向 MNS 传输的总能耗，在每个时隙都不超过可用的收集能量。约束 (18q) 保证收集能量的储存在每个时隙都保持非负且不超过最大容量。

---

## III. 所提解决方案

式 (18a) 中的优化问题是一个同时包含连续与二元决策变量的 MINLP 问题。这些变量包括二元变量 π_m(t)、连续卸载比例 δ_m(t)、卫星负载分布变量 θ_{m,i}(t)、UAV 轨迹分量 d_x(t) 与 d_y(t)，以及 UAV 计算频率分配 f^u_m(t)。离散与连续决策变量的共存，加上能量、时延与移动性约束之间的非线性耦合，使所建模的问题极具挑战性。尽管可以采用传统优化方法求解该问题，但由于搜索空间维度高、变量之间强相互依赖，它们在动态 SAGIN 环境中的实现变得困难。经典深度强化学习为这类序贯决策问题提供了切实可行的替代方案。然而，在如此高度耦合的状态-动作空间中，经典 DRL 在探索效率、收敛速度与学习质量方面仍可能面临挑战。

为应对这些挑战，我们提出了 QD-DRL 框架，利用量子特征编码来高效地表示复杂的状态-动作空间。具体而言，我们采用 QD-CE-A2C 与 QD-CE-PPO 在耦合的非线性约束下联合优化离散与连续决策变量。通过把量子变分线路与经典 DRL 主干网络相结合，所提框架改善了组合决策空间中的探索，并加速了向近优解的收敛。其目标是在动态环境中最小化加权系统代价，即 UAV 飞行能量、与任务相关的能量消耗以及端到端任务时延的组合，从而在面向 6G 的 SAGIN 中实现可扩展且高效的资源分配。

此外，在所提框架中，QD-CE-PPO 更适合那些稳定学习性能与可靠收敛较为重要的场景。它的主要优势在于裁剪（clip）更新机制，该机制有助于避免训练期间策略发生剧烈变化。这提升了鲁棒性，并使学习过程更稳定；在状态转移高度动态的复杂环境中尤为有用。然而，这种稳定性的提升是以更高的更新复杂度为代价的。相比之下，QD-CE-A2C 更适合那些偏好更简单学习结构与更低计算复杂度的场景。其主要优势是相对简单的 Actor-Critic 架构，可实现更快、更直接的更新。另一方面，与 PPO 相比，A2C 通常对回报波动更敏感，训练期间的收敛稳定性可能更低。因此，当收敛稳定性是主要关切时，QD-CE-PPO 更受青睐；而当算法简洁性与更低的学习复杂度更重要时，QD-CE-A2C 更具吸引力。

### A. 向量子驱动深度强化学习框架的转化

本节提出 QD-DRL 算法以求解所构建的 MINLP 问题。首先，我们把 MINLP 问题转化为一个受约束的马尔可夫决策过程（MDP），以便在不确定性下进行序贯决策。该 MDP 建模刻画了 UAV 轨迹、能量收集、任务卸载、UAV 计算资源分配与卫星负载分布的动态特性。QD-DRL 框架中的这一 MDP 建模可由三个组成部分完整描述，即动作空间、观测空间与奖励函数。

**1）动作空间**：动作空间 A 表示智能体在每个时隙 t 可用的全部可行控制决策，它定义了智能体影响环境并驱动未来回报的选项集合。相应地，每个时隙 t 的动作空间可定义为 a(t) = {π(t), δ(t), θ(t), d_x(t), d_y(t), f^u(t)}，其中 π(t) 选择在时隙 t 接纳并传输到 UAV 的 AP 索引。被接纳的任务存入队列内存，并在后续时隙使用其关联的控制变量进行处理。此外，δ(t) 是被选中 AP 的任务卸载到 MNS 的比例。进一步地，θ_i(t) 结合 (18h) 把任务分配到所有 SNS 上。d_x(t)、d_y(t) 分别是 UAV 在 x 与 y 方向上的位移变量。此外，f^u(t) 是 UAV 处的计算频率。

**2）观测空间**：观测空间 S 定义了智能体在每个时隙 t 可获得的信息。这起着关键作用，因为它决定了智能体如何感知环境、以及可以使用哪些知识来指导其动作。因此，时刻 t 的观测空间 s(t) 可定义为 s(t) = {{x_u(t), y_u(t)}, E^remain_u(t), AP_served(t), {Δx_near(t), Δy_near(t)}, E^tot_m(t), T^tot_m(t), q(t)/M}，其中 {x_u(t), y_u(t)} 是 UAV 在时刻 t 的坐标，E^remain_u(t) 是时刻 t 归一化后的 UAV 主电池剩余能量。此外，AP_served(t) = Σ_{m=1}^{M} π_m(t)/M 是时刻 t 已服务 AP 的百分比。{Δx_near(t), Δy_near(t)} 是相对最近未服务 AP 的相对坐标。E^tot_m 是第 m 个 AP 的总能量消耗，T^tot_m 表示第 m 个 AP 任务的端到端总时延。此外，q(t)/M 表示时刻 t 归一化后的队列长度。为把这一经典观测空间融入量子学习框架，必须将其映射到量子希尔伯特空间。

**3）奖励函数**：奖励函数 R 在引导 QD-DRL 智能体的学习过程中起关键作用，它提供评估智能体动作质量的反馈。奖励函数被设计为使强化学习最大化长期回报的目标与系统最小化加权代价的目标保持一致。因此，时刻 t 的奖励函数 r(t) 可表示为：

$$ r(t) = \frac{1}{W_S}\left(-w_1 E^{\mathrm{fly}}_u(t) - \sum_{m=1}^{M}\left(w_2 E^{\mathrm{total}}_m(t) + w_3 T^{\mathrm{total}}_m(t)\right)\right) + B(t) - P(t). \tag{19} $$

其中 W_S 是奖励函数的缩放因子。在该式中，目标函数以负值形式表达，从而使最小化加权系统代价等价于最大化累积回报。此外，B(t) 表示奖励函数中的奖励项（bonus），可定义为 B(t) = B(served) + B(all served)。这里 B(served) 是时刻 t 服务了一个 AP 时的奖励，B(all served) 是所有

AP 都被服务时的奖励；否则 B(t) = 0。该奖励项鼓励智能体优先保证所有 AP 被成功服务与系统覆盖。此外，P(t) 是为强制满足约束 (18b)–(18q) 而设计的惩罚函数，每当预定义的系统约束未被满足时即施加该惩罚。相应地，奖励函数在系统代价最小化与任务完成度、约束满足之间取得平衡。这确保智能体学习到的策略既能量高效、时延高效，又能维持整体系统可靠性。

### B. 量子驱动深度强化学习框架

在所提 QD-DRL 框架中，量子组件被用作 Actor-Critic 结构内紧凑的可训练模块。首先，经典环境状态在进入量子线路之前被投影为较低维的特征表示。随后，这一降维后的特征向量通过带可学习参数的角度编码被编码到量子比特系统中。采用 RY、RZ 旋转门把经典特征映射为可训练的量子态表示。在编码阶段之后，变分量子线路施加可训练的旋转层与纠缠操作，以把编码后的状态变换为更丰富的隐表示。纠缠门有助于刻画不同状态变量之间的耦合，这在本研究所考虑的 SAGIN 环境中非常重要。最后，测量 Pauli-Z 期望值并映射回经典域，供 Actor 与 Critic 模块用于策略与价值估计。相应地，量子线路充当一个紧凑的混合表示模块，在本文所考虑的设置下支撑有效的策略学习。

### C. 量子驱动的高性价比近端策略优化算法

**1）数据编码与 QNN**：本节介绍 QD-CE-PPO 框架，它是一种无模型 DRL 策略，融合了量子-经典混合学习结构。其工作流程首先把经典观测空间数据编码为希尔伯特空间中的量子态表示。作为数据编码方案，我们采用一种受「带可学习旋转的角度编码」（angle encoding with learnable rotations, AELP）[27] 启发的、可学习的角度编码策略。之所以采用这种编码，是因为它提供了一种可训练且紧凑的方式把经典特征映射为量子态，同时允许编码层自身在学习过程中自适应调整。在每个时间步 t，经典观测向量 s(t) ∈ R^{d_s} 经过一个可训练线性层处理，生成与量子线路输入维度兼容的特征向量。该变换表示为 y(t) = W′s(t) + b，y(t) ∈ R^{2n}，其中 W′ 与 b 是可训练参数。所得向量经归一化以保证单位长度与数值稳定性，ỹ(t) = y(t)/‖y(t)‖₂。然后，归一化向量 ỹ(t) 驱动一个初始处于基态 |0⟩^{⊗n} 的 n 比特量子线路。编码层施加依赖输入的量子旋转，其中每个量子比特经过 RY 与 RZ 门。选择这些门是因为它们提供了一种简单而有效的方式，把经典特征映射为可训练的量子旋转，同时保持线路结构高效。旋转角由归一化特征与可训练编码参数共同决定。相应地，编码操作可表示为

$$ U_{\mathrm{enc}}(\tilde{y}(t)) = \prod_{r=1}^{n}\left[\left(\bigotimes_{j=0}^{n-1} R_Y\left(\theta^y_{r,j}(t)\right) R_Z\left(\theta^z_{r,j}(t)\right)\right)\left(\prod_{j=0}^{n-2} \mathrm{CNOT}_{j,j+1}\right)\right]. \tag{20} $$

其中 θ^y_{r,j}(t) 与 θ^z_{r,j}(t) 表示在第 r 个编码块中施加于量子比特 j 的、依赖输入的旋转角。这些角度由归一化特征向量 ỹ(t) 与可训练编码参数共同决定。该编码由多个编码块组成，每个块通过依赖输入的旋转把归一化特征向量的一个子集注入量子线路。因此，得到的编码量子态可记为 |s_q(t)⟩ = U_enc(ỹ(t))|0⟩^{⊗n}。相应地，该量子态充当 QD-CE-PPO 框架内 QNN 的输入。

在编码阶段之后，编码量子态 |s_q(t)⟩ 由两个 QNN 处理。这些线路构成量子 Actor-Critic 框架的学习组件。第一个 QNN 充当 Actor 网络，用于生成策略；第二个 QNN 充当 Critic 网络，用于评估每个状态的价值以指导学习。两条线路采用相同的结构设计，均由 L 个变分层组成，但它们的参数集合彼此独立。选择该设计是为策略学习与价值学习同时提供紧凑而富有表现力的可训练表示。每个 QNN 执行一个酉变换，可表示为

$$ U_{\mathrm{ppo}}(\Phi, \Theta) = \prod_{l=1}^{L}\left[\left(\bigotimes_{i=0}^{n-1} R_Y(\Phi^{(l)}_i)\, R_Z(\Theta^{(l)}_i)\right)\left(\prod_{i=0}^{n-2} \mathrm{CZ}_{i,i+1}\right)\right]. \tag{21} $$

此处，R_Y(Φ^{(l)}_i) 与 R_Z(Θ^{(l)}_i) 是作用在第 l 层第 i 个量子比特上的单比特旋转门，而 CZ_{i,i+1} 在相邻量子比特之间引入纠缠。采用局部旋转与最近邻纠缠的组合，使线路既能刻画状态变量之间的耦合关系，又能保持拟设（ansatz）对硬件友好且相对较浅。为定义 Actor QNN，我们定义由 Φ_a = {Φ^{(l)}_{a,i}} 与 Θ_a = {Θ^{(l)}_{a,i}} 参数化的酉算子 U^actor_ppo(Φ_a, Θ_a)，这些参数通过更新来优化决定智能体动作的策略。相比之下，Critic QNN 由 U^critic_ppo(Φ_c, Θ_c) 定义，其参数为 Φ_c = {Φ^{(l)}_{c,i}} 与 Θ_c = {Θ^{(l)}_{c,i}}。该线路专注于评估每个状态的期望回报以指导策略更新。当 Actor QNN U^actor_ppo(Φ_a, Θ_a) 处理编码态 |s_q(t)⟩ 时，在 Pauli-Z 基下执行测量以获得期望值 ⟨π_{Φa,Θa}⟩ = {⟨Z_i⟩}^{n−1}_{i=0}。随后，测量结果经过一个经典

解码函数 F^actor_decode，生成策略分布。解码函数把量子期望值映射为正态分布 N(μ, σ) 的均值 μ。一个线性层后接 tanh 激活函数产生 μ，而标准差定义为 σ = exp(σ_log)。类似地，Critic QNN U^critic_ppo(Φ_c, Θ_c) 作用于同一个量子态 |s_q(t)⟩，其 Pauli-Z 测量结果由线性解码器处理，用于估计标量价值函数 V(s(t))。随后得到期望值 ⟨v_{Φc,Θc}⟩ = {⟨Z_i⟩}^{n−1}_{i=0}。

**2）QD-CE-PPO 算法**：在策略采样之后，智能体在 SAGIN 环境中执行所选动作 a(t)，获得即时奖励 r(t)，并转移到下一状态 s(t+1)。基于这种交互，时序差分（TD）目标 y(t) 可计算为

$$ y(t) = r(t) + \gamma V_{\Phi_c,\Theta_c}[s(t+1)], \tag{22} $$

其中 V_{Φc,Θc}[s(t+1)] 表示由参数为 Φ_c、Θ_c 的 Critic QNN 估计的状态价值，γ 表示折扣因子。此外，优势函数衡量在当前状态下执行动作 a(t) 的相对收益，可表示为

$$ A(t) = r(t) + \gamma V_{\Phi_c,\Theta_c}[s(t+1)] - V_{\Phi_c,\Theta_c}[s(t)], \tag{23} $$

为优化策略，Actor 损失 L_actor 可计算为

$$ L_{\mathrm{actor}} = -\mathbb{E}\left[\min\left(\frac{\pi_{\Phi_a,\Theta_a}(a(t)|s(t))}{\pi_{\Phi^{old}_a,\Theta^{old}_a}(a(t)|s(t))}A(t),\ \mathrm{clip}\left(\frac{\pi_{\Phi_a,\Theta_a}(a(t)|s(t))}{\pi_{\Phi^{old}_a,\Theta^{old}_a}(a(t)|s(t))}, 1-\epsilon, 1+\epsilon\right)A(t)\right)\right]. \tag{24} $$

其中 ϵ 是裁剪参数，用于限制过大的策略更新以保证训练稳定性。随后迭代计算 Actor 梯度 ∇_{Φa,Θa} L_actor 以更新 Actor QNN 的参数。此外，Critic 损失 L_critic 最小化预测值与目标值之间的均方误差，可表示为

$$ L_{\mathrm{critic}} = \mathbb{E}\left[\left(V_{\Phi_c,\Theta_c}(s(t)) - y(t)\right)^2\right], \tag{25} $$

相应地，对应梯度可计算为

$$ \nabla_{\Phi_c,\Theta_c} L_{\mathrm{critic}} = \mathbb{E}\left[2\left(V_{\Phi_c,\Theta_c}(s(t)) - y(t)\right)\cdot \nabla_{\Phi_c,\Theta_c} V_{\Phi_c,\Theta_c}(s(t))\right]. \tag{26} $$

类似地，Critic 网络的参数也相应更新。详细算法见算法 1。

> **算法 1：用于求解 (18) 的所提 QD-CE-PPO 算法**
> 1: 初始化：
> 2:   按指定参数初始化 SAGIN 环境。
> 3:   用参数 Φ_a, Θ_a 初始化 U^actor_ppo(Φ_a, Θ_a)。
> 4:   用参数 Φ_c, Θ_c 初始化 U^critic_ppo(Φ_c, Θ_c)。
> 5:   设置 Adam 优化器与超参数。
> 6: **for** episode = 1 **to** E **do**
> 7:    &nbsp;&nbsp;&nbsp;&nbsp;重置环境；获得初始状态 s₀。
> 8:    &nbsp;&nbsp;&nbsp;&nbsp;初始化经验回放缓冲区 Rollout = ∅。
> 9:    &nbsp;&nbsp;&nbsp;&nbsp;设置时间步 t = 0。
> 10:   &nbsp;&nbsp;&nbsp;&nbsp;**while** rollout 未完成 **and** t < T **do**
> 11:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把 s(t) 编码为量子态 |s_q(t)⟩。
> 12:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过测量与解码，应用 Actor QNN 得到策略 π_{Φa,Θa}(a(t)|s(t))。
> 13:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;采样动作 a(t) ∼ π_{Φa,Θa}(·|s(t))。
> 14:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 SAGIN 中执行 a(t)；获得奖励 r(t)、下一状态 s(t+1)、完成标志 d(t)。
> 15:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把转移 {s(t), a(t), r(t), s(t+1), d(t)} 存入 Rollout。
> 16:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 s_t ← s(t+1)；递增 t ← t+1。
> 17:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** d(t) = 1 **then**
> 18:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break
> 19:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
> 20:   &nbsp;&nbsp;&nbsp;&nbsp;**end while**
> 21:   &nbsp;&nbsp;&nbsp;&nbsp;用 (22) 与 (23) 为 Rollout 中的转移计算 TD 目标与优势。
> 22:   &nbsp;&nbsp;&nbsp;&nbsp;**for** update iteration = 1 **to** K **do**
> 23:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从 Rollout 中采样小批量（minibatch）。
> 24:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算 Actor 损失梯度 ∇_{Φa,Θa} L_actor。
> 25:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;计算 Critic 损失梯度 ∇_{Φc,Θc} L_critic。
> 26:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 Actor 参数：Φ_a, Θ_a ← Φ_a, Θ_a + α_a ∇_{Φa,Θa} L_actor。
> 27:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 Critic 参数：Φ_c, Θ_c ← Φ_c, Θ_c + α_c ∇_{Φc,Θc} L_critic。
> 28:   &nbsp;&nbsp;&nbsp;&nbsp;**end for**
> 29: **end for**

### D. 量子驱动的高性价比优势 Actor-Critic 算法

**数据编码与 QNN**：我们进一步提出 QD-CE-A2C 算法，这是一种量子增强的、无模型 Actor-Critic DRL 方法。与 QD-CE-PPO 类似，我们使用 AELP 进行量子数据编码。在通过编码阶段获得量子态 |s_q(t)⟩ 之后，它被送入两个不同的 QNN。Actor QNN 从当前状态决定动作选择策略，而 Critic QNN 评估期望回报以指导学习。尽管两个网络采用相同的 L 层变分线路结构，但它们保持彼此独立的可训练参数，从而使策略函数与价值函数能够独立优化。每个 QNN 内的量子线路可表示为

$$ U_{\mathrm{A2C}}(\Phi,\Theta) = \prod_{l=1}^{L}\left[\left(\bigotimes_{i=0}^{n-1} R_Y(\Phi^{(l)}_i)\, R_Z(\Theta^{(l)}_i)\right)\left(\prod_{i=0}^{n-2} \mathrm{CZ}_{i,i+1}\right)\right], \tag{27} $$

其中 R_Y(Φ^{(l)}_i) 与 R_Z(Θ^{(l)}_i) 是作用在第 l 层第 i 个量子比特上的单比特旋转门，CZ_{i,i+1} 在相邻量子比特之间引入纠缠。此处，Actor QNN 由 U^actor_A2C(Φ_a, Θ_a) 参数化，其中 Φ_a = {Φ^{(l)}_{a,i}} 与 Θ_a = {Θ^{(l)}_{a,i}} 表示可训练参数。Critic QNN 采用相同架构，可表示为 U^critic_A2C(Φ_c, Θ_c)，其参数为 Φ_c = {Φ^{(l)}_{c,i}} 与 Θ_c = {Θ^{(l)}_{c,i}}。

当 Actor QNN U^actor_A2C(Φ_a, Θ_a) 处理输入状态 |s_q(t)⟩ 时，在 Pauli-Z 基下执行投影测量以获得期望值 ⟨π_{Φa,Θa}⟩ = {⟨Z_i⟩}^{n−1}_{i=0}。这些值在 N_shot 次测量采样上取平均以缓解采样噪声，平均结果由经典函数 F^actor_decode 解码为：

$$ \pi_{\Phi_a,\Theta_a}(a(t)|s(t)) \xleftarrow{F^{\mathrm{decode}}_{\mathrm{actor}}} \frac{1}{N_{\mathrm{shot}}}\sum_{k=1}^{N_{\mathrm{shot}}} \mathcal{M}\left(\langle \pi_{\Phi_a,\Theta_a}\rangle\right). \tag{28} $$

解码函数通过一个线性层与 tanh 激活把测量得到的期望值映射为正态分布 N(μ, σ²) 的均值 μ，其中标准差定义为 σ = exp(σ_log)。类似地，Critic QNN U^critic_A2C(Φ_c, Θ_c) 作用于同一编码态 |s_q(t)⟩，Pauli-Z 测量产生期望向量 ⟨v_{Φc,Θc}⟩ = {⟨Z_i⟩}^{n−1}_{i=0}。这些值通过线性层 F^critic_decode 解码，以估计标量价值函数 V_{Φc,Θc}(s(t))。

**QD-CE-A2C 算法**：在从 Actor 策略 π_{Φa,Θa}(a(t)|s(t)) 中采样动作 a(t) 并在 SAGIN 环境中执行之后，智能体获得奖励 r(t) 并转移到下一状态 s(t+1)。于是 TD 目标可计算为

$$ y(t) = r(t) + \gamma V_{\Phi_c,\Theta_c}(s(t+1)), \tag{29} $$

其中 γ 是折扣因子。优势函数可定义为

$$ A(t) = r(t) + \gamma V_{\Phi_c,\Theta_c}(s(t+1)) - V_{\Phi_c,\Theta_c}(s(t)). \tag{30} $$

于是，用于策略优化的 Actor 损失可表示为

$$ L_{\mathrm{actor}} = \mathbb{E}\left[-\log \pi_{\Phi_a,\Theta_a}(a(t)|s(t)) A(t) - C_e H\left(\pi_{\Phi_a,\Theta_a}\right)\right], \tag{31} $$

其中 H(π_{Φa,Θa}) 表示策略的熵，C_e 是熵系数。此外，Critic 损失可表示为

$$ L_{\mathrm{critic}} = \mathbb{E}\left[C_v\left(V_{\Phi_c,\Theta_c}(s(t)) - y(t)\right)^2\right], \tag{32} $$

其中 C_v 表示价值损失系数。两个 QNN 的梯度均被迭代计算以更新全部可训练参数。所提 QD-CE-A2C 框架的详细流程见算法 2。

**1）复杂度分析**：所提 QD-CE-PPO 算法的计算复杂度主要来自量子比特测量以及 Actor 与价值 QNN 的梯度评估。每一步包括复杂度为 O(Bn²) 的 AELP 编码，随后是经 L 层演化的 QNN，其复杂度为 O(BL(3n−1))，以及对 n 个量子比特进行 N_shot 次测量，复杂度为 O(BnN_shot)。经典解码过程对动作空间施加线性映射 O(n(d_c + d_d))、对价值估计施加 O(n)，合计 O(Bn(d_c + d_d + 1))。使用参数平移（parameter-shift）规则计算梯度还需额外付出 O(BL(3n−1)) 与解码部分 O(Bn(d_c + d_d + 1)) 的代价。因此，单步计算复杂度可表示为 O(Bn(L(3n−1) + N_shot + d_c + d_d + 1))。考虑 T 个时间步、E 个回合以及带有经验回放的 N_e 轮次（epoch），总复杂度为 O(E N_e T B n (L(3n−1) + N_shot + d_c + d_d + 1))。

> **算法 2：用于求解 (18) 的所提 QD-CE-A2C 算法**
> 1: 初始化：
> 2:   按指定参数初始化 SAGIN 环境。
> 3:   用参数 Φ_a, Θ_a 初始化 U^actor(Φ_a, Θ_a)。
> 4:   用参数 Φ_c, Θ_c 初始化 U^critic(Φ_c, Θ_c)。
> 5:   设置 Adam 优化器及其他超参数。
> 6: **for** episode = 1 **to** E **do**
> 7:    &nbsp;&nbsp;&nbsp;&nbsp;重置环境以获得初始状态 s(t)。
> 8:    &nbsp;&nbsp;&nbsp;&nbsp;初始化时间步 t = 0。
> 9:    &nbsp;&nbsp;&nbsp;&nbsp;**while** t < T **and** 未完成 **do**
> 10:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把状态 s(t) 编码为 |s_q(t)⟩。
> 11:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;应用 Actor QNN：U^actor(Φ_a, Θ_a)|s_q(t)⟩，测量并解码，得到策略 π_{Φa,Θa}(a(t)|s(t))。
> 12:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;从策略 π_{Φa,Θa}(a(t)|s(t)) 中采样动作 a(t)。
> 13:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 SAGIN 中执行 a(t)；接收 {r(t), s(t+1), d(t)}。
> 14:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对 s(t) 应用 Critic QNN：U^critic(Φ_c, Θ_c)|s_q(t)⟩，测量并解码，得到价值估计 V_{Φc,Θc}(s(t))。
> 15:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;把状态 s(t+1) 编码为 |s_q(t+1)⟩。
> 16:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;对 s(t+1) 应用 Critic QNN：U^critic(Φ_c, Θ_c)|s_q(t+1)⟩，测量并解码，得到价值估计 V_{Φc,Θc}(s(t+1))。
> 17:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用 (29) 计算 TD 目标。
> 18:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;使用 (30) 计算优势。
> 19:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 Actor 参数：Φ_a, Θ_a ← Φ_a, Θ_a + α_a ∇_{Φa,Θa} L_actor。
> 20:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新 Critic 参数：Φ_c, Θ_c ← Φ_c, Θ_c + α_c ∇_{Φc,Θc} L_critic。
> 21:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;更新状态 s(t) ← s(t+1)；递增 t ← t+1。
> 22:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** d(t) = 1 **then**
> 23:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break
> 24:   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end if**
> 25:   &nbsp;&nbsp;&nbsp;&nbsp;**end while**
> 26: **end for**

此外，所提 QD-CE-A2C 的计算复杂度来自量子比特测量过程以及 Actor 与价值 QNN 的梯度计算。每一步包括复杂度为 O(Tn²) 的带可学习参数角度编码（AELP），经 L 层演化的 QNN 复杂度为 O(TL(3n−1))，以及对 n 个量子比特进行 N_shot 次测量，复杂度为 O(TnN_shot)。经典解码阶段对动作空间施加线性映射 O(n(d_c + d_d))、对价值估计施加 O(n)，合计 O(Tn(d_c + d_d + 1))。使用参数平移规则计算梯度还需额外 O(TL(3n−1)) 与解码的 O(Tn(d_c + d_d + 1)) 代价。因此，单步计算复杂度可表示为 O(Tn(L(3n−1) + N_shot + d_c + d_d + 1))。考虑 T 个时间步与 E 个回合，总复杂度为 O(E T n (L(3n−1) + N_shot + d_c + d_d + 1))。

---

## IV. 数值结果与讨论

### A. 仿真设置

仿真使用 TorchQuantum 库完成，该库支持在经典硬件上对含噪 NISQ 设备进行模拟 [28]。所有仿真均在一台

支持 CUDA 的经典计算平台上完成，使用 NVIDIA 4 GB GPU、16 GB 内存与 Intel Core i5 处理器。所选用的线路设置特意与当前小规模量子硬件的实际限制保持一致，同时仍能在模拟条件下对所提量子-经典混合学习框架进行有意义的评估。重要的是，为保证公平比较，所提 QD-CE-PPO 与 QD-CE-A2C 方法，连同经典 A2C 与 PPO 基线，均在相同的 SAGIN 环境、状态与动作定义、奖励结构与训练时长下评估。基线超参数的选取旨在保证在所考虑设置下具有稳定的学习行为。尽管并未对所有基线方法开展大规模的超参数调优研究，但所有被比较算法都保持了相同的实验条件，以提供均衡的评估。对于 QD-CE-PPO，Actor 与 Critic QNN 均采用 4 量子比特、线路深度 L = 4，测量采样数 1024。Actor 学习率设为 1 × 10⁻³，Critic 学习率设为 1 × 10⁻²。折扣因子 γ = 0.99，广义优势估计系数 λ = 0.99，PPO 裁剪参数 ϵ = 0.2，梯度裁剪范数为 0.5，更新轮次 N_e = 10，批量大小 B = 64。对于 QD-CE-A2C，Actor 与 Critic QNN 同样采用 4 量子比特、线路深度 L = 4，测量采样数 1024。Actor 学习率为 1 × 10⁻³，Critic 学习率为 1 × 10⁻²。折扣因子 γ = 0.99，熵系数为 0.1，价值损失系数 C_v = 1.0。两种算法的训练均进行 E = 1000 个回合。每个回合的最大步数为 200 步；但如果所有 AP 都已被服务，或 UAV 主电池耗尽，回合可能提前终止。

此外，所采用的 DT 误差因子为 4%，用于刻画 DT 估计值与 UAV 实际计算 CPU 周期之间的偏差。该偏差影响 UAV 的有效处理速率，并直接影响本地计算时延与 UAV 侧能耗。具体而言，较大的 DT 误差会降低决策精度，可能导致次优的资源分配、低效的卸载以及整体加权系统代价的上升。反之，较低的 DT 误差会改善虚拟实体与物理实体之间的一致性，从而实现更准确的控制决策与更好的系统性能。因此，4% 的设置提供了一个实用的折中点，用于在贴近实际的 DT 条件下评估所提 SAGIN 框架的鲁棒性。此外，SAGIN 环境参数汇总于表 I。

> **表 I：SAGIN 环境参数 [3], [8], [22]**
>
> | 参数 | 取值 |
> |---|---|
> | AP 数量（M） | 10 |
> | UAV 天线数（K） | 8 |
> | 区域（x^max, y^max） | (1000, 1000) m |
> | 高度（H, R） | (100 m, 300 km) |
> | 载波与带宽（f_c, B） | (28 GHz, 20 MHz) |
> | 路径损耗指数（α） | 2.0 |
> | 发射功率 [p_m, p_u, p_ms,ss] | [1, 5, 5] W |
> | 计算功率 [f^u_max, f^ss] | [1, 2] GHz |
> | 处理能量系数（ξ_u, ξ_ss） | 1 × 10⁻²⁷ |
> | 任务大小（D_m） | [1×10⁴, 2×10⁴] bits |
> | DT 误差 | 4% |
> | 任务复杂度（Q_m） | [1×10⁸, 2×10⁸] cycles |
> | 最大 UAV 覆盖与速度（d^max, v^max） | (200 m, 40 m/s) |
> | WPT 设置（η_u, P_bs） | (0.99, 10 W) |
> | 均衡因子（w₁, w₂, w₃） | (5×10⁻⁴, 0.5, 0.5) |

### B. 数值结果

本节对所述 QD-DRL 算法与系统性能进行全面评估。

**1）收敛性能**：我们评估所提 QD-CE-A2C 与 QD-CE-PPO 算法相较传统 A2C 与 PPO 框架的收敛行为，如图 2 所示。随着训练推进，所有算法的累积回报都呈上升趋势，说明它们具备随时间学习到更优策略的能力。在基线方法中，A2C 表现出缓慢但更迟滞的改善；相比之下，PPO 在训练早期阶段之后收敛更快，并达到更高的回报水平。所提 QD-CE-A2C 在训练过程的大部分时间内表现出优于经典 A2C 的收敛行为：它从更高的回报水平起步，改善更稳定，并且以更快的整体学习速度达到有竞争力的最终回报。这表明，所提量子-经典混合框架能够在所考虑的 SAGIN 环境中以更少的 QNN 可训练参数有效支撑策略学习。类似地，所提 QD-CE-PPO 也表现强劲，并在训练过程的大部分时间内保持优于经典 PPO 的回报性能，这在训练早中期更为明显：它更早达到较高回报水平，显示出更快的收敛行为。在训练后期，两种基于 PPO 的方法达到相近的最终回报值，说明经典 PPO 在该设置下同样表现良好；但所提 QD-CE-PPO 仍在训练时长的大部分区段内表现出更快的回报提升。这表明所提量子-经典混合结构能够以更少的 QNN 可训练参数，为所考虑的环境提供有效的策略学习框架。相应地，图 2 的结果表明：所提 QD-DRL 方法相比其经典对应方法取得了更优的收敛行为。特别地，QD-CE-A2C 相较 A2C 有明显改善，而 QD-CE-PPO 相比 PPO 表现出更快的回报提升并取得强劲的最终性能。图中阴影区域表示训练期间的回报波动。

**2）训练回合数与已服务 AP 数**：如图 3 所示，所有算法的已服务 AP 数量随训练回合数增加而上升，体现出持续学习与策略改进。所提 QD-CE-A2C 相比经典 A2C 取得了更快的初期改善，说明在早期阶段就具有更高的服务水平。类似地，所提 QD-CE-PPO 表现出快速学习能力，并在收敛后保持稳定行为。两种量子驱动框架都呈现出更平滑的过渡与更少的波动，反映出稳定且一致的策略自适应。相比之下，经典 A2C 需要更多回合才能达到可比的服务水平。最终，所有算法都收敛到服务全部可用 AP，说明任务成功完成、学习收敛。总体而言，所提基于 QD-DRL 的方法相比传统对应方法实现了更快、更稳定的收敛。

**3）AP 数量变化下的加权平均系统代价**：我们评估 AP 数量对加权平均系统代价的影响，如图 4 所示。不同 AP 数量下的加权平均系统代价取最后 100 个训练回合的平均值，以反映各方法最终稳定的学习性能。可以观察到，随着 AP 数量增加，所有算法的系统代价都呈上升趋势——这是可预期的，因为更密集的 AP 部署会带来更高的协同复杂度并增加网络整体资源需求。随着 AP 数量增加，系统必须处理更多任务到达、更多通信链路以及更重的计算负担。在被比较的方法中，传统 A2C 与 PPO 框架随着 AP 密度增长出现明显的代价上升；所提 QD-CE-A2C 与 QD-CE-PPO 方法也遵循相同的总体趋势，但在所评估的各种设置下都保持了有竞争力且总体更低的系统代价。这表明所提 QD-DRL 框架在更密集的 AP 条件下，能更有效地适应通信、计算与资源分配决策之间更强的耦合。同时也可以观察到，即便系统变得更密集、资源交互变得更复杂，所提方法仍能保持其学习有效性。这一点很重要，因为在密集 SAGIN 场景中，算法不仅要应对更高的负载水平，还要应对任务接纳、卸载、负载均衡与能量相关决策之间更强的相互依赖。在这种情况下，更有效的策略学习结构能够带来更好的决策。总体而言，图 4 的结果表明：所有方法都受 AP 密度增加的影响，但所提基于 QD-CE 的算法在不同网络条件下都保持了很强的代价性能，证明了它们在密集 SAGIN 环境中进行资源管理的有效性。

**4）任务复杂度变化下的加权平均系统代价**：此处我们评估任务复杂度对平均加权系统代价的影响，如图 5 所示。不同任务复杂度设置下的加权平均系统代价同样取最后 100 个训练回合的平均值。可以观察到，随着任务复杂度提高，所有方法的系统代价都呈上升趋势。这种行为是可以预期的，因为更复杂的任务需要更高的计算资源、更长的处理时间，以及跨 SAGIN 系统更谨慎的卸载与调度决策。随着任务复杂度增长，UAV-MEC 与卫星计算层的负担也随之增加，这直接影响本地处理、任务卸载与资源分配决策。与此同时，更高的任务复杂度会增加时延与能耗，从而导致整体加权系统代价上升。因此，所有方法的上升趋势都与所考虑优化问题难度的增加相一致。在被比较的方法中，传统 A2C 与 PPO 框架随着任务复杂度提高出现明显的代价上升；所提 QD-CE-A2C 与 QD-CE-PPO 方法也遵循同样的总体规律，但在所评估设置下保持了有竞争力且总体更低的系统代价。这表明所提 QD-DRL 框架在更高任务复杂度下，能更有效地适应计算需求、通信负担与任务调度决策之间更强的耦合。同时也可以观察到，即便任务变得更具挑战性，所提方法仍保持了很强的代价性能。这一点很重要，因为在高任务复杂度下，学习算法必须在任务应当本地处理、卸载还是跨可用资源进行均衡等决策上做出更高效的选择。在这种情况下，更好的策略学习结构有助于减少不必要的代价增长，并提升系统整体效率。

**5）存在 DT 误差时、任务复杂度变化下的加权平均系统代价**：如图 6 所示，平均加权系统代价在不同 DT 误差水平下随任务复杂度变化。结果同样取最后 100 个训练回合的平均值，以反映各方法最终稳定的学习性能。所有配置都显示：随着任务复杂度上升，系统代价明显增加，这反映了更复杂任务所带来的更高计算需求、通信负担与资源分配难度。当 DT 误差水平较低时，两种所提算法都保持相对更低的系统代价与更稳定的性能——在这种情况下，数字孪生能够更准确地表示环境，从而支撑更好的任务卸载、轨迹控制与资源分配决策。然而，当 DT 误差变高时，两种方法的系统代价都会上升，这是因为系统表示的精度下降影响了所学决策的质量，使高效资源管理更加困难。在两种方法之间，QD-CE-PPO 在所有任务复杂度范围内都表现出比 QD-CE-A2C 更好的代价效率与更平滑的代价变化。这表明当系统同时面临更高任务需求与更强 DT 不确定性时，QD-CE-PPO 能够更有效地适应。尽管 QD-CE-A2C 对任务复杂度与 DT 精度的变化也做出一致响应，但在 DT 误差更高的设置下其代价受到的影响更大。因此，两种所提方法对 DT 精度与任务复杂度的变化都做出一致响应，但 QD-CE-PPO 对 DT 引入的不确定性表现出更强的鲁棒性，并取得更好的整体代价性能。

**6）QD-DRL 算法下的 UAV 轨迹**：如图 7 所示，该图给出所提 QD-CE-A2C 与 QD-CE-PPO 算法得到的最优轨迹。两条轨迹都从指定起点出发，穿越服务区域以覆盖所有 AP 位置。UAV 遵循自适应路径，确保完整的服务覆盖，这验证了量子驱动框架所学决策策略的有效性。QD-CE-A2C 表现出更广泛的探索模式，而 QD-CE-PPO 则沿着更直接、更稳定的路线依次抵达各 AP。两种算法最终都完成了对所有 AP 的服务，展示了高效的轨迹优化与可靠的路径规划性能。

根据仿真结果，所提量子驱动算法的性能提升主要来自量子特征编码与变分量子线路所带来的表示能力增强。由于所考虑的 SAGIN 优化问题涉及 UAV 轨迹控制、任务接纳、任务卸载与卫星侧负载分布之间的强非线性耦合，高效的状态表示对于准确的策略学习非常重要。在本工作中，量子增强的映射帮助智能体比标准经典映射更有效地捕捉这些复杂关系。因此，所学策略包含更有用的信息，从而改善了探索、支撑更快的

收敛，并带来更高的累积回报。这也有助于 UAV 在任务时长内更有效地服务所有 AP，同时满足能量与时延要求。因此，量子计算在所提框架中的作用是：通过更好的特征表示与决策过程来强化 Actor-Critic 的学习。

---

## V. 结论与未来工作

本文研究了一个能量感知的、面向 6G 的 SAGIN，其中包含具备能量收集能力的 MEC 辅助 UAV 以及一个纳米卫星星座。所构建的 MINLP 问题通过联合优化 UAV 轨迹、任务卸载、计算资源分配与卫星负载均衡，在满足能量与时延约束的同时最小化总体系统代价。所提 MINLP 问题采用 QD-CE-A2C 与 QD-CE-PPO 算法求解，这两种算法借助量子特征编码与变分学习，有效处理了高维混合整数优化问题。仿真结果证实，与经典对应方法相比，两种量子驱动框架都实现了更快的收敛、更高的累积回报与更低的总体系统代价。特别地，QD-CE-PPO 在变化条件下展现出更优的稳定性与适应性，凸显了量子增强策略优化在动态 6G 环境中的优势。相应地，未来研究可以在此基础上扩展：引入多 UAV 与多卫星协作；探索更先进的量子线路结构以增强策略表达能力；以及集成分布式或多智能体量子学习，以实现 SAGIN 各层之间可扩展的协同。

---

## 参考文献（保留原文著录格式）

[1] Y. Lin et al., "Satellite-MEC integration for 6G Internet of Things: Minimal structures, advances, and prospects," IEEE Open J. Commun. Soc., vol. 5, pp. 3886–3900, 2024.

[2] X. Yang, Z. Zho, and B. Huang, "URLLC key technologies and standardization for 6G power Internet of Things," IEEE Commun. Standards Mag., vol. 5, no. 2, pp. 52–59, Jun. 2021.

[3] D. V. Huynh et al., "Joint sensing, communications, and computing design for 6G URLLC service-oriented MEC networks," IEEE Internet Things J., vol. 11, no. 20, pp. 32429–32439, Oct. 2024.

[4] J. Yang, J. Shi, Y. Sun, and A. Men, "Task prediction-based edge computing offloading of satellite-HAP-terrestrial integrated network," IEEE Netw. Lett., vol. 7, no. 3, pp. 185–189, Sep. 2025.

[5] Y.-H. Hsu and T. T. T. Phan, "A DRL-based energy-efficient service caching and task offloading scheme for 6G MEC SAGINs," IEEE Trans. Commun., vol. 73, no. 12, pp. 13967–13982, Dec. 2025.

[6] F. Tang, C. Wen, L. Luo, M. Zhao, and N. Kato, "Blockchain-based trusted traffic offloading in space-air-ground integrated networks (SAGIN): A federated reinforcement learning approach," IEEE J. Sel. Areas Commun., vol. 40, no. 12, pp. 3501–3516, Dec. 2022.

[7] W. Fan, Q. Meng, G. Wang, H. Bian, Y. Liu, and Y. Liu, "Satellite edge intelligence: DRL-based resource management for task inference in LEO-based satellite-ground collaborative networks," IEEE Trans. Mobile Comput., vol. 24, no. 10, pp. 10710–10728, Oct. 2025.

[8] Z. Shao, H. Yang, and Z. Xiong, "Intelligent latency-oriented optimization for multi-UAV-assisted mobile edge computing in space-air-ground integrated networks," IEEE Trans. Commun., vol. 73, no. 12, pp. 13384–13398, Dec. 2025.

[9] L. Sun, R. Liang, L. Wan, K. Liu, Z. Ning, and J. Wang, "Online partial computation offloading optimization in wireless powered mobile edge computing network," IEEE Trans. Cognit. Commun. Netw., vol. 12, pp. 1481–1495, 2026.

[10] Y. Jiang, X. Tang, B. Li, R. Zhang, J. Liu, and N. Liu, "Energy-efficient UAV edge computing for space-air-ground integrated networks," in Proc. IEEE Wireless Commun. Netw. Conf. (WCNC), Milan, Italy, Mar. 2025, pp. 1–6.

[11] G. S. Kim, Y. Cho, S. Park, S. Jung, and J. Kim, "Quantum multiagent reinforcement learning for joint cube satellites and high-altitude long-endurance aerial vehicles in SAGIN," IEEE Trans. Aerosp. Electron. Syst., vol. 61, no. 4, pp. 9490–9510, Aug. 2025, doi: 10.1109/TAES.2025.3556050.

[12] X. Hu, P. Wen, H. Xiao, W. Wang, and K.-K. Wong, "Maximizing energy charging for UAV-assisted MEC systems with SWIPT," IEEE Trans. Veh. Technol., vol. 74, no. 5, pp. 8442–8447, May 2025.

[13] S. Zhu, B. Zhu, K. Chi, K. Yu, and S. Mumtaz, "Long-term computation rate maximization in UAV-enabled wirelessly powered MEC," IEEE Trans. Commun., vol. 73, no. 11, pp. 12545–12560, Nov. 2025.

[14] S. Bao, S. Zhang, K. Chi, K. Yu, and S. Mumtaz, "A DRL-framework for full-duplex WPCN enabled mobile edge computing," IEEE Trans. Veh. Technol., vol. 74, no. 10, pp. 16121–16136, Oct. 2025.

[15] A. Paul, K. Singh, C.-P. Li, O. A. Dobre, and T. Q. Duong, "Digital twin-aided vehicular edge network: A large-scale model optimization by quantum-DRL," IEEE Trans. Veh. Technol., vol. 74, no. 2, pp. 2156–2173, Feb. 2025.

[16] S. C. Prabhashana, D. Van Huynh, and T. Q. Duong, "Quantum DRL for UAV-RIS-aided maritime communications with 6G digital twin applications," in Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops), Jun. 2025, pp. 396–401.

[17] Silvirianti, B. Narottama, and S. Y. Shin, "UAV coverage path planning with quantum-based recurrent deep deterministic policy gradient," IEEE Trans. Veh. Technol., vol. 73, no. 5, pp. 7424–7429, May 2024.

[18] B. Narottama, S. Aïssa, and Z. Mohamed, "AI-enabled framework for energy sustainability evaluation and enhancement of wireless networks," IEEE Trans. Green Commun. Netw., vol. 10, pp. 978–993, 2026.

[19] Silvirianti, B. Narottama, and S. Y. Shin, "Layerwise quantum deep reinforcement learning for joint optimization of UAV trajectory and resource allocation," IEEE Internet Things J., vol. 11, no. 1, pp. 430–443, Jan. 2024.

[20] B. Narottama and S. Y. Shin, "Quantum neural networks for resource allocation in wireless communications," IEEE Trans. Wireless Commun., vol. 21, no. 2, pp. 1103–1115, Feb. 2022.

[21] S. C. Prabhashana, D. V. Huynh, H. Jung, B. Canberk, S. L. Cotton, and T. Q. Duong, "Quantum deep reinforcement learning for URLLC satellite-air-ground integrated networks with digital twin applications," IEEE Internet Things J., vol. 13, no. 3, pp. 4230–4246, Feb. 2026.

[22] H. Zhou, J. Wang, L. Zhao, D. Meng, G. Feng, and R. Li, "Joint optimization of charging time and resource allocation in wireless power transfer aided federated learning," IEEE Internet Things J., vol. 12, no. 17, pp. 35065–35077, Sep. 2025.

[23] B. Zhu, L. Huang, K. Chi, A. Alharbi, K. Yu, and M. Guizani, "Enhancing energy efficiency in wireless-powered MEC systems through Lyapunov-guided deep reinforcement learning," IEEE Trans. Wireless Commun., vol. 24, no. 9, pp. 7563–7580, Sep. 2025.

[24] S. Huang, L. Wang, X. Wang, B. Tan, W. Ni, and K.-K. Wong, "Edge intelligence in satellite-terrestrial networks with hybrid quantum computing," IEEE Wireless Commun. Lett., vol. 14, no. 5, pp. 1341–1345, May 2025.

[25] T. Q. Duong, L. D. Nguyen, B. Narottama, J. A. Ansere, D. V. Huynh, and H. Shin, "Quantum-inspired real-time optimization for 6G networks: Opportunities, challenges, and the road ahead," IEEE Open J. Commun. Soc., vol. 3, pp. 1347–1359, 2022.

[26] M.-H.-T. Nguyen et al., "Real-time optimized clustering and caching for 6G satellite-UAV-terrestrial networks," IEEE Trans. Intell. Transp. Syst., vol. 25, no. 3, pp. 3009–3019, Mar. 2024.

[27] E. Ovalle-Magallanes, D. E. Alvarado-Carrillo, J. G. Avina-Cervantes, I. Cruz-Aceves, and J. Ruiz-Pinales, "Quantum angle encoding with learnable rotation applied to quantum–classical convolutional neural networks," Appl. Soft Comput., vol. 141, Jul. 2023, Art. no. 110307.

[28] H. Wang et al., "QuantumNAS: Noise-adaptive search for robust quantum circuits," in Proc. IEEE Int. Symp. High-Perform. Comput. Archit. (HPCA), Apr. 2022, pp. 692–708.

---

## 图目录

- **图 1**：面向任务关键型应用的、带纳米卫星星座的 6G 使能空天地一体化网络示意图。
- **图 2**：QD-DRL 算法与其经典对应方法的收敛性能（横轴：训练回合 0–1000；纵轴：累积回报约 −1.0 ~ 0.6；四条曲线：A2C、PPO、所提 QD-CE-A2C、所提 QD-CE-PPO）。
- **图 3**：已服务 AP 数量随训练回合的变化。
- **图 4**：加权平均系统代价随 AP 数量的变化。
- **图 5**：加权平均系统代价随任务复杂度的变化。
- **图 6**：存在 DT 不确定性时、不同任务复杂度下的加权平均系统代价。
- **图 7**：QD-DRL 算法下的 UAV 轨迹。

## 表目录

- **表 I**：SAGIN 环境参数（见上文第 IV.A 节）。

> **译者说明**：本文档是 IEEE JSAC 2026 论文《Quantum Machine Learning for Wireless-Powered UAV Positioning in 6G Digital Twin SAGIN With Cooperative Nano-Satellite Constellations》的全文中文翻译，按原文结构逐节对应，公式编号与原文一致（如 (13)、(15)、(18m)），参考文献保留英文著录格式。所有技术判断与数据均以原文为准，若需引用请核对英文原文。
