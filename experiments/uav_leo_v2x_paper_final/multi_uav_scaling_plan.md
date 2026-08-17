# 多 UAV + 多用户 + 随机任务场景扩展方案（Multi-UAV Scaling Plan）

> 目标：回应"单 UAV 轨迹太少、12/16 用户太少、1 km 区域太小"的审稿风险，
> 把仿真规模提升到中档期刊主流水平（2–4 架 UAV、50–80 用户、1.5–3 km），
> 同时保持并强化现有贡献（PMEO 决策顺序），把"用户-UAV 关联"也做成新的可写点。
> 日期：2026-08-12。关联文档：`route_a_method.md`、`scale_sweep_summary.md`、`paper_results.md`。

---

## 0. 结论先行（TL;DR）

- 现有代码**单 UAV 是写死的**（`env.py` 标量位置/队列/电池、单热点），
  扩多 UAV 不是改参数就能解决，需要一次中等规模的接口重构（Phase 1，约 2–3 天）。
- 文献对标：2024 年多 UAV MEC/VEC 主流规模是 **2–4 架 UAV、10–80 用户、1–3 km 区域**，
  且"用户-UAV 关联 + 轨迹 + 卸载"联合优化是标准问题设定。我们目前 1 UAV / 12–16 用户明显偏小。
- 建议**主场景升级为中规模 S2：3 UAV + 50 用户 + 2 km × 2 km + 3 个移动热点**，
  再配小/大规模做敏感性（S1: 2 UAV/20 用户/1.5 km；S3: 4 UAV/80 用户/3 km）。
- 算法侧：PMEO 自然扩展为 **PMEO-M（逐 UAV 专家轨迹 + post-move 关联/卸载）**，
  复杂度 O(U×K×15)/时隙，50 用户 × 3 UAV 完全可跑（比现有单 UAV 12 用户只多 ~12 倍仿真量）。
- 新增贡献点：**"关联也必须移动后求解"（post-move association）**——多 UAV 时
  用户-UAV 关联与卸载都依赖移动后的实际位置，这是单 UAV 论文写不了的、区别于
  Cai 等 2025 GDRL 文献的新机制。

---

## 1. 现状：单 UAV 模型的局限（代码层面）

### 1.1 写死点（文件:行号）

| 位置 | 现状 | 问题 |
|---|---|---|
| `uav_leo_experiment/config.py:8-10` | `UavLeoConfig` 只有 `users/leos`，**没有 `uavs` 字段** | 无法声明 UAV 数量 |
| `uav_leo_experiment/env.py:20-26` | `uav_pos`（单组坐标）、`uav_backlog`（标量）、`uav_battery`（标量） | 全部是单 UAV 状态 |
| `uav_leo_experiment/env.py:30-33, 66-75` | `hotspot_pos/vel/road/along` 都是**单个热点** | 无法表达多热点同时出现 |
| `uav_leo_experiment/env.py:62` | `uav_pos` 初始化为区域中心一点 | 多 UAV 无法分布到不同区域 |
| `uav_leo_experiment/env.py:77-97` | `_sample_tasks()` 用**单个** `hotspot_pos` 做高斯加权泊松 | 多热点时概率场只有一个峰 |
| `uav_leo_experiment/env.py:120-132` | `_move_hotspot()` 只移动一个热点 | 单热点沿道路往返 |
| `uav_leo_experiment/env.py:202-203` | `rate_user_uav_vec(c, user_pos, uav_pos)` 返回 (U,) 向量 | 多 UAV 需要 (U, K) 矩阵 |
| `uav_leo_experiment/physics.py:57-63, 208-237` | `rate_user_uav` / `rate_uav_leo` 单 UAV 标量/一维接口 | 全部链路函数要支持广播到 K 架 UAV |
| `uav_leo_experiment/algorithms.py:58, 81` | `evaluate_actions_vec` 动作是 `(P,2)` 单 UAV 移动 | PMEO/MPC 的轨迹搜索都是单 UAV |
| `uav_leo_experiment/scenario_spec.py` | `v2x_hotspot_hard`=12 用户、`v2x_hotspot_stress`=16 用户，区域 1000 m | 规模天花板过低 |

### 1.2 为什么必须改

- 单 UAV 时"用户-UAV 关联"自由度不存在，论文无法覆盖多 UAV 场景下
  **关联(association) → 轨迹 → 卸载**三层耦合，审稿人容易认为问题过于简化。
- 1 km × 1 km、12–16 用户对应的是"单热点小规模"，与车联网/城市尺度（几 km、
  数十用户）差距大；`scale_sweep_summary.md` 已证明面积放大到 2 km 后决策顺序
  收益反而增大（+8.6 vs +2.2），说明**规模放大对现有贡献有利**，值得做。

---

## 2. 文献参考：别人怎么设的（参数对标表）

> 引用均为此处标注的文献（写作时按期刊要求复核卷/期/页码）。

| 文献（年份/出处） | UAV 数 | 用户/设备数 | 区域 | 任务生成方式 | 移动模型 | 优化对象 |
|---|---|---|---|---|---|---|
| Ju et al., *JNCA* 2024, "A multi-UAV assisted task offloading and path optimization via MADRL", doi:10.1016/j.jnca.2024.103919 | 3 | 10 UEs | 1000 m × 1000 m | 随机任务到达 | 用户/无人机移动 | 任务卸载 + 路径优化（MADRL） |
| Liu et al., arXiv:2409.14782 (2024), "Energy-Efficient Multi-UAV-Enabled MEC over SAGIN" | 多 UAV | 多个 MU | 典型 1–2 km 地面区域 | 计算密集任务随机生成 | 无人机轨迹 + 用户分布 | MU-UAV 关联 + 轨迹 + 卸载 + 算力 + 功率（交替优化） |
| Wang et al., arXiv:2410.04151 (2024), 多 UAV ISCC（通信+感知融合） | 多 UAV | 多用户 | km 级区域 | 感知/通信任务 | **Gaussian-Markov 用户移动模型**，MAPPO | 联合通信感知 + 轨迹 + 卸载 |
| TMC 2024, "Multi-Objective Optimization for Multi-UAV-Assisted MEC" (TMC.2024.3350078) | 多 UAV | 数十用户 | km 级 | 多目标优化 | 用户移动 | 能耗/时延多目标 |
| MDPI *Drones* 2023, "Joint UAV Deployment and Task Offloading for Multi-UAV Edge Computing" | 多 UAV | 多用户 | km 级 | 时延敏感任务 | 静态/移动混合 | 部署 + 卸载 |
| 电子科技大学学报 2024, "多无人机辅助 MEC 任务卸载及路径优化"（MADRL） | 3 | 10 UEs | 1000 m | 随机任务 | 移动 | 卸载 + 路径 |
| 本文现有 | **1** | 12 / 16 | **1 km** | 单热点泊松 | 用户沿道路 + 单热点移动 | 轨迹 + 卸载（PMEO） |

**对标结论**：
- 3 UAV / 50 用户 / 2 km 处在文献主流区间内，不夸张、可信度高；
- 文献中"多热点随机出现 + 随机任务 + 用户移动"是标准做法，我们的 `_sample_tasks()`
  泊松采样已具备雏形，扩展成本低；
- 没有发现文献强调"卸载/关联必须在移动后位置求解"这一决策顺序问题 → 这是我们的差异化叙事。

---

## 3. 建议场景设计（三档）

原则：**物理量保持一致**（UAV 25 m/s、热点 18–20 m/s、用户 16–18 m/s、
链路/算力参数沿用 `v2x_hotspot_hard`），只增加 UAV 数、用户数、热点数、区域，
否则审稿人会说"速度不物理"（`scale_sweep_summary.md` 第 5 节已有教训）。

| 档位 | UAV 数 K | 用户数 U | 热点数 M | 区域 | 每 UAV 服务用户 | 用途 |
|---|---|---|---|---|---|---|
| S1 小规模 | 2 | 20 | 2 | 1.5 km | ~10 | 与现有 1 UAV 场景做对照桥接 |
| **S2 中规模（主场景）** | **3** | **50** | **3** | **2 km** | ~17 | 论文主结果（对标 JNCA 3UAV 风格） |
| S3 大规模 | 4 | 80 | 4 | 3 km | ~20 | 上限鲁棒性展示 |

初始化策略：
- UAV 初始位置沿道路/热点分布（如 3 架分别靠近 3 个初始热点），避免全部从中心出发；
- 热点在道路网络上随机选点初始化，沿路以 `hotspot_speed` 移动（复用 `_move_hotspot` 逻辑，
  改成多热点数组）；
- 用户仍按"沿道路随机 + 热点附近聚集"采样（复用 `_sample_tasks` 的分布逻辑，热点改数组）。

---

## 4. 任务随机生成方案（多热点 + 突发）

现有 `_sample_tasks()`（`env.py:77-97`）已经是"空间泊松 + 热点高斯权重"，
扩展为多热点后保留该机制，另加两档随机性：

1. **多热点随机出现/消失（temporal-spatial burst）**：
   - 每 `T_burst` 时隙以概率 `p_spawn` 在随机道路位置生成临时热点（持续 `L_burst` 时隙后消失）；
   - 模拟演唱会/大型活动/事故拥堵等"随机区域任务爆发"；
   - 参数建议：`T_burst=10` 时隙、`p_spawn=0.3`、`L_burst=5` 时隙、额外热点半径 80 m。
2. **稀疏随机任务（远离热点的低概率任务）**：
   - 保留 `base_arrival_prob=0.05` 的全局泊松底噪，确保"追热点"不是唯一策略，
     无人机需要在覆盖与集中之间权衡（这正是多 UAV 关联的意义）。
3. **任务类型随机化**：沿用现有 `task_bits`（0.8–3 Mb）与 `cycles_per_bit`（900–1700）
   的均匀随机，小任务本地可算、大任务必须卸载（保证卸载决策仍是核心）。

> 可选加分项：用户移动升级为 **Gaussian-Markov 模型**（arXiv:2410.04151 的做法），
> 与现有"沿道路移动"二选一作为消融，证明结果对移动模型鲁棒。建议先不加，保持改动面可控。

---

## 5. 算法适配方案（PMEO-M 与基线）

### 5.1 PMEO-M（多 UAV 版，主方法）

每个时隙三步：

1. **关联（association）**：每个用户选"链路速率最高且非满队列"的 UAV
   （速率 = `rate_user_uav_vec` 的 (U,K) 矩阵逐行 argmax），或者按负载均衡
   （每个 UAV 服务子集人数 ≤ ceil(U/K)）；LEO 作为全局兜底。
2. **专家轨迹（per-UAV demand-predictive move）**：每架 UAV 朝"其服务子集的任务加权
   质心 + 最近热点一个时隙后的位置"的混合目标移动（速度上限内）——与单 UAV 版一致，
   只是质心按关联子集计算。
3. **post-move 精确卸载（PMEO 核心）**：以移动后的实际位置，对每个用户枚举
   {关联 UAV, LEO, local} × 比例网格求最小 cost。**关联本身也在移动后重算一次**（见 5.3）。

复杂度：O(U × K × 15) 次链路/计算仿真/时隙（50×3×15=2250，比现有 12×15 大 12.5 倍，
CPU 单集 < 1 s 量级，无训练）。

### 5.2 基线矩阵（多 UAV 版本）

| 基线 | 多 UAV 适配方式 |
|---|---|
| Current-pos exact | 移动前位置做关联+卸载（决策顺序消融，对应现有 `current_exact`） |
| Predict-TEA / Follow-TEA | 关联子集质心追踪 / 直接追最近热点（轨迹消融） |
| MPC-H3（多 UAV） | 每 UAV 独立 H=3 步前瞻采样轨迹（相互解耦近似） |
| MAPPO / MADRL（新增学习基线） | 每 UAV 一个 actor + 集中 critic，作为"要学习才做得好"的对照 |
| Random / All-LEO / All-local | 沿用现有实现，逐 UAV 广播 |

> 建议主表 = PMEO-M vs Current-pos vs Follow-TEA vs MPC-H3 vs MAPPO，与单 UAV 主表
> 结构一致，审稿人一眼能对上。

### 5.3 新增贡献点：post-move association（关联顺序）

单 UAV 论文只能写"卸载必须在移动后求解"。多 UAV 后出现**第二个顺序问题**：
用户-UAV 关联如果在移动前按旧位置计算，会把用户分给"这一时隙实际到不了"的 UAV，
系统性高估可达速率。因此可以提出：

> **命题 3（关联-卸载联合顺序）**：多 UAV 场景下，先在移动后位置重算关联、再解
> 卸载，其总代价 ≤ 移动前关联 + 移动后卸载 的代价（同理可证），且差距随
> UAV 移动量/用户密度增大而增大。

这就是一个与 Cai 等 2025 GDRL 论文**不同的新机制**：他们做的是图强化学习做
关联+卸载+资源分配，我们做的是"决策顺序"——并能在多 UAV 场景下给出更强的
实验证据（决策顺序收益随 UAV 数/规模单调上升）。

---

## 6. 可信度提升设计

1. **规模敏感性矩阵**（主实验）：UAV 数 × 用户数 × 区域 × 热点数 = S1/S2/S3 三档，
   每档 10 seeds × 15–40 集（沿用 `scale_sweep` 的 seed 集：73,1,7,42,2024,12345,999,314,555,888）。
2. **统计方法**：5-seed 池化配对 t 检验 + 跨 seed 均值±std（与 `route_a_method.md` 第 4 节一致），
   报效应量而非只报均值。
3. **与文献对标表**：把第 2 节参数表放进论文 Related Work/Experiments，明确"我们的
   3 UAV / 50 用户 / 2 km 处于主流区间，且任务生成多热点化"。
4. **消融**：多热点开关、突发任务开关、关联负载均衡 vs 速率贪心、Gaussian-Markov 移动（可选）。
5. **可复现**：`run_multi_uav.bat` 一键跑完 S1/S2/S3 + 画图，CSV 落在
   `experiments/uav_leo_v2x_paper_final/multi_uav/`。

---

## 7. 实施计划（分阶段）

| 阶段 | 内容 | 验证标准 | 预计耗时 |
|---|---|---|---|
| P0 | 方案评审（本文档） | 用户确认 S2 为主场景、3 UAV/50 用户/2 km | 1 天 |
| P1 | 代码重构：`config.py` 加 `uavs/hotspots`；`env.py` 状态改 (K,)/(K,2)/(M,2)；`physics.py` 链路函数广播到 (U,K)/(K,L) | **单 UAV 配置下结果与现有完全一致（回归）**；新场景 smoke test 不报错 | 2–3 天 |
| P2 | `algorithms.py` 实现 PMEO-M / Current-pos / Follow-TEA / MPC-M；关联模块独立成函数 | 行为合理性（UAV 不重叠、用户都有归属、队列不溢出） | 1–2 天 |
| P3 | 新场景 `v2x_multi_hard`（S2 主场景）+ S1/S3 派生；`run_sweep` 支持多 UAV | 三档都能跑 15 集 × 10 seeds | 1 天 |
| P4 | 全量实验：S1/S2/S3 × 10 seeds × 方法矩阵 → CSV | 与单 UAV 结论一致性检查（PMEO-M 仍 ≥ 基线） | 1–2 天（CPU） |
| P5 | 画图 + 统计 + 写作：规模敏感性图、决策顺序收益随规模曲线、与文献参数对照表 | 图件进入 `figures/`，风格与现有论文图一致 | 1–2 天 |

**里程碑检查点**：
- P1 结束必须做"单 UAV 回归"——若单 UAV 结果漂移 >0.5，说明重构引入了行为差异，需回退修复；
- P4 结束检查 PMEO-M vs MPC 在 2 km 是否仍打平/占优（若多 UAV 下 MPC 反超，需要按
  `route_a_method.md` 第 3 节思路处理：轨迹层可选 MPC，卸载层仍用 post-move exact）。

---

## 8. 风险与对策

| 风险 | 可能性 | 对策 |
|---|---|---|
| 多 UAV 下 PMEO-M 与 MPC 差距翻转 | 中 | 论文表述改为"PMEO 负责卸载层，轨迹层可选 MPC"（已有先例）；或把主规模调回决策顺序收益最大的 2 km + 3 UAV 组合 |
| 重构引入单 UAV 回归 | 低 | P1 强制回归测试；`env.py` 保持 `uavs=1` 时走原代码路径 |
| 80 用户 × 4 UAV 仿真过慢 | 低 | O(U×K×15) 向量化已够；S3 可降为 10 集 |
| 关联贪心导致负载不均衡 | 中 | 加负载均衡规则（每 UAV 用户数 ≤ ceil(U/K)），并作为消融报告两种关联策略 |
| 审稿人质疑"为什么不用 MADRL" | 中 | 加 MAPPO 基线（P2），并用 C2 叙事："学习在多 UAV 关联下仍无额外收益，决策顺序才是关键"——这反而是更强的主张 |

---

## 9. 需要用户确认的点

1. **主场景**是否定为 S2：3 UAV + 50 用户 + 3 热点 + 2 km？（推荐，与文献对标最好）
2. 是否接受**每 UAV 服务用户 ~17**的负载（50/3），还是希望 50 用户配 4 UAV（每架 ~12，
   与现有 hard 场景的 12 用户/UAV 更接近）？
3. 是否需要把 **MAPPO/MADRL 学习基线**纳入 P2（+1 天工作量，但审稿防御更强）？
4. 该方案是否同步进飞书 wiki（授权仍在有效期内）？

---

## 附录 A：参考文件位置

- 权威场景定义：`GDRL/uav_leo_experiment/scenario_spec.py`
- 单 UAV 状态：`GDRL/uav_leo_experiment/env.py:20-33, 62-97`
- 链路函数：`GDRL/uav_leo_experiment/physics.py:57-71, 208-237`
- PMEO 实现与复杂度：`GDRL/uav_leo_experiment/algorithms.py:58-90`
- 方法叙事：`GDRL/experiments/uav_leo_v2x_paper_final/route_a_method.md`
- 规模扫描先例：`GDRL/experiments/uav_leo_v2x_paper_final/scale_sweep_summary.md`

## 附录 B：建议引用文献（写作时按期刊格式整理）

1. T. Ju, L. Li, S. Liu, Y. Zhang, "A multi-UAV assisted task offloading and path
   optimization for mobile edge computing via multi-agent deep reinforcement learning,"
   *J. Netw. Comput. Appl.*, vol. 229, art. 103919, 2024. doi:10.1016/j.jnca.2024.103919.
2. W. Liu et al., "Energy-Efficient Multi-UAV-Enabled MEC Systems over Space-Air-Ground
   Integrated Networks," arXiv:2409.14782, 2024.
3. （arXiv:2410.04151）多 UAV ISCC 联合通信感知，Gaussian-Markov 用户移动 + MAPPO，2024。
4. "Multi-Objective Optimization for Multi-UAV-Assisted MEC," *IEEE TMC*, 2024
   (doi:10.1109/TMC.2024.3350078)。
5. "Joint UAV Deployment and Task Offloading Scheme for Multi-UAV-Assisted Edge
   Computing," *Drones* 2023（MDPI）。