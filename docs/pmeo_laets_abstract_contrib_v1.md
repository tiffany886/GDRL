# PMEO+LAETS v1 — 新摘要、三条贡献与红队预案（2026-09-09）

> 定位：TVT / IoT-J / WCL。**不冲** JSAC/Nature；VA-DAG-HO（B 线）永不并入本稿。
> 依据：`docs/plan_pmeo_laets_rebuttal_upgrade.md`；证据见
> `docs/pmeo_laets_phase1_formalization.md`。

## 1. 一句话定位（写作锚点）

> 在离散时隙空天地 MEC 中，轨迹更新与卸载求值之间的**状态相位错配**是可量化的
> 次优性来源：本文刻画其消失/放大条件，提出免训练的**结构化短视界校正**（PMEO /
> PMEO-M-Eco），并在**有限同步带宽**下用负载漂移驱动的事件触发刷新保护该校正
> 的收益不被陈旧状态侵蚀。

## 2. 新摘要（英文原稿级草稿）

> In slot-based air–ground–space MEC, offloading decisions that rely on the
> UAV position of the previous decision instant are evaluated against the
> channel realized *after* the UAV moves. We formalize this state-phase
> misalignment: for a fixed trajectory the post-move exact solve is never
> worse, its gain vanishes with the per-slot displacement and reappears when
> the rate map has steep spatial gradients or tasks are deadline-urgent. On a
> 2 km, three-UAV vehicular testbed the proposed training-free structured
> myopic policy (load-balanced association, candidate move set, H-step
> flight-energy lookahead, exact post-move offloading) beats a receding-horizon
> controller on reward/latency/success at equal or better energy, over a
> 10-seed × 10-episode protocol, and dominates decentralized DRL baselines
> trained for 30k steps. Under limited synchronization bandwidth we show that
> load-drift-triggered state refresh preserves most of the correction gain
> while cutting refresh cost versus fixed-period refresh during bursts.

## 3. 三条贡献（v1 表述）

1. **错误源刻画（理论 + 数值机制）**：给出状态相位错配的形式化（命题 1–3），
   并数值验证增益随位移消失、随信道梯度/热点强度/deadline 紧急度放大的条件
   曲线 —— 同时澄清“用户位置观测噪声”与“UAV 决策相位偏差”是两类不同误差。
2. **结构化短视界校正（方法 + 实验）**：PMEO/PMEO-M-Eco-H5 —— 免训练、确定性、
   单时隙 ms 级复杂度；与 MPC 的对照以**性能–复杂度 Pareto** 呈现
   （`figures/fig_pareto_complexity_v3.png`），不宣称“碾压 DRL”。
3. **不完美信息下的事件触发刷新（系统）**：把 LAETS 重定位为 AoI/事件触发问题：
   同步成本–决策质量前沿上的**负载漂移触发刷新**，量化突发场景下相对固定周期
   的前沿改善（沿用 `digital_twin_uav_leo` RQ1–RQ4 证据，措辞改 AoI 口径）。

## 4. 措辞与 Related Work 锚点

- 禁用：数字孪生“新一代”标题、先飞后卸作为范式发现、以“击败 DRL 三倍”开场。
- 允许：state-phase misalignment；structured myopic planning = MPC with
  H=1 + domain-structured candidate actions；AoI / event-triggered control /
  NCS with limited communication（新坐标）。
- DRL 主文措辞：“PMEO is a deployment-oriented, training-free alternative;
  DRL serves as a reference baseline”（不是“证伪对象”）。附录给状态维度对齐清单。
- MPC 措辞：“equivalent to H=1 exact receding-horizon with a structured,
  load-aware candidate move set”（诚实等同关系）。

## 5. 红队预案（对照批评三条英文攻击）

1. **“Gain is an artifact of discrete slots.”** → 命题 2 + α/`v_max` 曲线：
   增益在 `α→2`、`deadline→1.0`、`Δp→0` 时消失（已实测），存在条件是工程
   `Δt` 不可任意小 + 梯度陡 + 紧急业务，不是仿真 bug。
2. **“You just did MPC with H=1.”** → 承认 MPC-H1+结构候选 = 我们方法；卖点改为
   复杂度（0.86–24 ms/slot 区间可选）与结构化移动集的**可解释增量**，Pareto 图
   显示无训练下在多个复杂度档位支配 MPC-H3。
3. **“DT is a relabeled periodic sync.”** → 术语改 AoI/事件触发；因果链改：
   相位校正依赖状态新鲜度 → 负载陈旧伤害更大 → 事件触发以负载漂移为特征 →
   突发下刷新成本–决策质量前沿改善（RQ4 曲线）。若审稿人仍咬，Limitation 主动
   承认“同步延迟/丢包鲁棒为后续工作”。

## 6. 与导师口径（一段话）

> 顶刊批评让我们把贡献从“离散实现细节抬成范式”收回：定位 TVT/IoT-J；主问题写成
> “相位错配的可分析误差源 + 不完美信息下的事件触发状态刷新”；对照改为与 MPC 的
> 性能–复杂度 Pareto；LAETS 挂 AoI/事件触发坐标；灾后 DAG 线已物理分离另文。

## 7. 状态检查（相对 plan Phase 0–4）

- Phase 0（决议）：✅ 本文档 + plan §5 冻结。
- Phase 1：✅ 形式化 + 命题（见 phase1 doc）；机制扫参 + 两种误差图已出；
  ⬜ 摘要/Intro 英文全文改写待并入论文主文件。
- Phase 2：✅ Pareto 主图（k3，dl=0.65）+ per-slot 开销表；⬜ DRL 加长训练/
  optimization-embedding 公平版（可选项）；⬜ CSI 陈旧小实验（可选项）。
- Phase 3：措辞替换表见 `docs/pmeo_laets_aoi_wording_map.md`；正文替换待做。
- Phase 4：红队预案即本文 §5。

## 8. 2026-09-10 第三轮硬化（顶刊级评审回应）

> 本节覆盖 §7 的旧状态。逐条映射见 `docs/pmeo_laets_topjournal_revision_memo.md`。

- **定理 1（取代原命题 1）**：两阶段解耦误差界 / 近似比，证明残余误差只来自轨迹项且比值不随视界退化。
- **推论 1**：相位校正增益 ≡ 陈旧决策的后悔；判据为决策单元是否改变，幅度由边界代价落差决定。
- **命题 2 重写**：非切换区局部 Lipschitz + 切换边界阶跃界 `λ·N_cross` + 无全局 Lipschitz 常数 + Sigmoid 软化。
- **§3.1**：一阶漂移 vs 二阶色散 Taylor 形式化；`½Δ_p c = 6.23e-3 m⁻²`（R²=0.9991），解析 `σ* ≈ 19.9 m` 与实测一致。
- **可观测性**：新增 (O1)(O2)；`load` 模式降级为 oracle 上界；本地入流量代理 LAETS-loc 零信令且支配前沿
  （+223/+340/+161）；纯本地变点 LAETS-cp 低于前沿（−319/−405/−272，诚实负结果）。
- **开销**：物理模型 4.1e-4 J/次（空口 4.1 ms）；主张限定在硬带宽预算；软代价最优区间 ω∈[24.6,153]，
  物理代价 ≈4e-7 低于下界——边界已如实写入。
- **Pareto**：三目标非支配前沿上 MPC 在三档 w_e 全被支配；Eco-H5 能耗仅在 w_e=1e-3 单点 +2.68%。
- **DRL**：新增公平性定界段（动作空间结构 / 30k 步≈冷启动 / 结论落点为免除超参与非凸收敛风险）。
- 状态：Phase 1/3 正文并入 ✅；Phase 2 主件 ✅（CSI 陈旧、加长 DRL 训练仍为可选）。
