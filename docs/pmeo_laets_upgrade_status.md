# PMEO+LAETS 改进包 — 执行状态总览（2026-09-09）

Goal：按 `docs/plan_pmeo_laets_rebuttal_upgrade.md` 四刀改进执行并完善文档
（Phase 0–4 分期；定位 TVT/IoT-J；VA-DAG-HO 物理分离）。

## 1. 进度表

| 阶段 | 状态 | 已完成 | 证据 |
|---|---|---|---|
| Phase 0 决议 | ✅ | 冻结定位/禁语；B 线剥离；本文档 + abstract_contrib §6 | `pmeo_laets_abstract_contrib_v1.md` |
| Phase 1 理论+机制 | ✅（数据与图；证明/正文待并入论文） | 命题 1–3 形式化；v_max/α/热点强度/deadline 扫参；两种误差图 | `pmeo_laets_phase1_formalization.md`；`phase1_misalignment/` |
| Phase 2 基线防守 | 🔶 主件完成 | MPC 性能–复杂度 Pareto + per-slot 开销表；DRL 措辞降级稿 | `fig_pareto_complexity_v3.png`；`phase2_overhead.json`；abstract_contrib §4 |
| Phase 3 LAETS→AoI | ✅（主件） | 措辞替换映射表；正文第 4 章标题/动机/RQ/架构/小结改为 AoI/事件触发；§1.2.4 Related Work 按 NCS/AoI 重写 | `pmeo_laets_aoi_wording_map.md`；`thesis_draft_pmeo_laets.md` ch4（引文编号待补 2–3 条） |
| Phase 4 红队 | ✅ | 三条英文攻击预案 | abstract_contrib §5 |

## 2. 本轮新产物索引（repo 相对路径）

实验（`experiments/uav_leo_v2x_paper_final/`）：
- 脚本：`_sweep_phase_misalignment.py`、`_plot_phase1_misalignment.py`、
  `_plot_pareto_v3.py`
- 数据：`phase1_misalignment/{sweep_summary,user_noise_sensitivity,urgency_sweep}.csv`
  + `run_log.json`；`phase2_overhead.json`
- 图：`figures/fig_phase1_misalignment_mechanisms.png`、
  `figures/fig_phase1_two_errors.png`、`figures/fig_pareto_complexity_v3.png`

文档（`docs/`）：
- `pmeo_laets_phase1_formalization.md`（形式化+命题+数值）
- `pmeo_laets_abstract_contrib_v1.md`（新摘要+三条贡献+红队）
- `pmeo_laets_aoi_wording_map.md`（Phase 3 措辞映射）
- 本文件；`plan_pmeo_laets_rebuttal_upgrade.md` 已追加执行状态段

## 3. 关键数字备忘（修复版 env）

- 单 UAV 机制：相位增益 α=2.0→0.00、α=3.0→+2.46、α=3.5→+8.90；热点强度
  0.4→+0.94、1.0→+2.46；deadline 1.0→+0.06、0.5→+4.95；v_max 12→+4.54（非单调）。
- 观测噪声代价（post-move，σ m）：5→-0.19、10→-0.73、20→-2.53、30→-5.59；
  相位偏差（v=25, α=3）=+2.46 —— 两误差同量级点在 σ≈20 m≈20% 热点半径。
- k3 per-slot 开销：Eco-H5 24.44 ms / MPC-H3 16.32 ms / Eco-H3 10.48 ms /
  DRL 推理 ≈6.7 ms（另需训练）/ PMEO-M 0.86 ms。
- k3 dl=0.65 主表（10 seeds×10 ep）：Eco-H5 -3594.8 最优；Eco-H3 -3644.0；
  MPC-H3 -3687.1；Current-pos -4193.4；M-DQN -10581 / M-TD3 -12519 / M-SAC -13084。

## 4. 注意与下一步

- 所有 `experiments/**` 产物在 git 中被 `.gitignore` 忽略；若要版本化请
  `git add -f`（尤其 `phase1_misalignment/`、两个 figure、新脚本）。
- 下一步（按优先级）：把命题并入论文主文件第 3 章与英文写作；Phase 3 术语替换
  落到 `thesis_draft_pmeo_laets.md`；可选补 λ_drop 与覆盖边缘网格、CSI 陈旧实验。

## 5. 续跑补充（同日第二轮）
- Phase 1 正文整合：形式化/命题 3 与机制表、两种误差解释已并入 `docs/thesis_draft_pmeo_laets.md`
  §3.1/§3.4；标题/摘要/创新点/关键词/§3.6.2/§3.8/§5.1 同步改为新口径；§3.7 表 3-1 替换为
  2026-09-09 修复版 10-seed 数字（来源 `multi_uav_k3_table_v3.csv`）。
- Phase 3 正文：thesis 第 4 章（标题/动机/RQ/架构/小结）改为 AoI/事件触发口径；§1.2.4 Related
  Work 段已按 NCS/AoI 坐标重写（引文编号待补 2–3 条）。
- Phase 2 补充：单 UAV 性能–复杂度 Pareto（`fig_pareto_complexity_single_v3.png` +
  `phase2_overhead_single.json`；MPC-H3 以约 3× 开销换 0.3 reward，PMEO 仍在前沿）。
- 历史 mentor/walkthrough/outline 文档已加“口径更新”头注（旧表述不并入新正文）。
- plan §5 检查表已按实际完成度打勾（Phase0/1/4 ✅，Phase2/3 🔶 主件+文档层完成，
  剩余均为可选项）。

## 6. Go/No-Go 对照（plan §6 判定，2026-09-09）

| Go 判据 | 状态 | 证据 |
|---|---|---|
| 贡献不再依赖“离散伪问题当范式”；有错配定义 + 增益随 Δp 消失证据 | ✅ | `phase1_formalization` §2（命题 2）；sweep α=2.0→0.00、deadline 1.0→+0.06 |
| 主对照是 MPC Pareto；DRL 不夸张 | ✅ | k3 与单 UAV Pareto 图 + thesis 表 3-1/§3.6.2 口径 |
| LAETS 挂事件触发/AoI 坐标，突发前沿仍在 | ✅ | thesis 第 4 章 + `aoi_wording_map` + digital_twin RQ4 |
| Limitation 主动写清物理简化 | ✅ | thesis §5.2；`aoi_wording_map` §4（零延迟/无丢包假设为边界） |
| 与 B 线（VA-DAG-HO）完全分离 | ✅ | plan §0/§4.2 与 `abstract_contrib` §1/§6 |

结论：按 plan §6 判据 **Go（TVT/IoT-J 防守加强版）**；JSAC/Nature 明确 No-Go。
剩余工作均为正文/引文/可选实验层面（见 §5 与 plan §5 未勾选项），不阻塞 Go。

## 6. 续跑补充（2026-09-10 第三轮：理论硬化 + 可观测性 + 开销）

针对「顶刊级」评审意见（命题 1 同义反复、命题 2 Lipschitz 危机、触发依赖全局真值、
能耗劣势、DRL 公平性），本轮改动如下。

### 6.1 理论（`thesis_draft_pmeo_laets.md` §3.1 / §3.4）
- **命题 1 替换为定理 1（两阶段解耦误差界 / 近似比）**：
  `C_PMEO − C* ≤ T(L+2L_g)Δ`，比值 `≤ 1 + (L+2L_g)Δ/c_min`；证明已写入。
  要点：卸载阶段精确求解 ⇒ 残余误差只来自轨迹项，且比值**不随视界 T 退化**。
- **新增推论 1（增益 = 陈旧决策的后悔）**：把「取 min 不会更差」改写为可检验刻画——
  增益 >0 当且仅当决策单元发生改变；幅度由单元边界代价落差决定。
- **命题 2 重写**：区分(a) 非切换区域局部 Lipschitz、(b) 切换边界阶跃界
  `|ΔJ| ≤ L_R Δp + λ·N_cross`、(c) 不存在全局 Lipschitz 常数、(d) Sigmoid 软化。
- 证据（新）：`two_error_expansion.json`、`fig_pareto_objectives_v3.png` 等。
- **§3.1 新增 Taylor 形式化**：零均值噪声为二阶 `½Tr(∇²c Σ)`（拟合 `½Δ_p c = 6.23e-3 m⁻²`，
  R²=0.9991），相位错配为一阶 `∇c·Δp`；解析交叉点 `σ* ≈ 19.9 m`，与实测相等点（σ=20 m）一致。
- **修正一处旧结论**：v_max 非单调**不是**「实际移动量」驱动——实测位移单调上升（5.97→24.91 m），
  真正原因是**单元改变频率升（0.52%→1.69%）而单次改变代价降（20.7→5.2）**，总增益在中等机动处最大。

### 6.2 触发机制的可观测性（§4.2 / §4.4 / §4.5 RQ5）
- 新增**可观测性假设 (O1)(O2)**，明确不假设读取全局 `Load_phys`；原 `load` 模式降级为 oracle 上界。
- 新增**本地入流量代理 LAETS-loc** 与**纯本地变点 LAETS-cp**，代码改动：
  `digital_twin_uav_leo/adaptive_sync.py`（`local_ingress_error` / `change_point_error`）、
  `digital_twin_uav_leo/runner.py`（`task_mode ∈ {local_arr, local_cp}` + trace + probe 计数）。
- 结果（k3 burst，5 seeds × 10 ep）：LAETS-loc 三个档位**全部支配固定周期前沿**（+223/+340/+161），
  **零额外信令**；LAETS-cp 三档**全部低于前沿**（−319/−405/−272）——孪生预测确有必要（诚实负结果）。

### 6.3 探测开销模型与适用边界（§4.5 RQ6）
- 物理模型：512 B/次全量刷新、68 B/次探测、1 Mbps、20 dBm ⇒ `E_sync = 4.1e-4 J`（空口 4.1 ms）。
- **主张的成立区间**改为**硬带宽预算**（能量从来不是瓶颈：100 次刷新≈0.04 J vs 整回合 35 kJ，1e-6）。
- 软代价下的最优区间：`ω ∈ [24.6, 153]`；物理代价 `≈4e-7` 远低于下界 ⇒
  **若刷新免费且带宽无限，逐时隙刷新才最优**——边界已如实写入 §4.5 RQ6 与 §5.2 第 4 条。

### 6.4 非支配性与 DRL 公平性（§3.7.1 / §3.7.2 / §5.2 第 7 条）
- 三目标 (drop, 时延, 能耗) 非支配前沿（表 3-2 + `fig_pareto_objectives_v3.png`）：
  **MPC-M-H3 在三档 w_e 上全部被 Eco 变体支配**；Eco-H5 能耗仅在 `w_e=1e-3` 单点高 2.68%，
  而该点 MPC 本身不在前沿上；`w_e=1e-4/1e-2` 时 Eco-H5 能耗分别低 4.07% / 2.50%。
- 新增 DRL 公平性定界段（动作空间结构、30k 步≈冷启动、结论落点为「免除超参与非凸收敛风险」）。

### 6.5 本轮新增产物
- 脚本：`experiments/uav_leo_v2x_paper_final/_run_local_proxy.py`、`_analyze_sync_overhead.py`、
  `_analyze_two_errors.py`、`_plot_hardening_v3.py`
- 数据：`local_proxy/local_proxy_raw.csv`、`local_proxy/local_proxy_traces.json`、
  `sync_overhead_analysis.json`、`two_error_expansion.json`
- 图：`figures/fig_pareto_objectives_v3.png`、`figures/fig_sync_overhead_sensitivity.png`
- 仍未做（可选）：§1.2.4 的 NCS/AoI 引文编号；CSI 陈旧/丢包鲁棒性实验；英文全文合并润色。

## 7. 续跑补充（2026-09-11：P0 引文 + P1 可计算 Δ）

### 7.1 P0：文献与编号（拆掉「DT 包装」攻击面）
- §1.2.1 增补李雅普诺夫在线调度引文；§1.2.2 增补部分卸载与 MPC 稳定性/最优性理论引文；
  §1.2.4 增补**事件触发/自触发控制 3 条 + AoI 2 条 + 数字孪生 1 条**。
- 参考文献由 9 条扩至 **18 条**，并**按正文首次出现顺序重新编号**（验证：first-appearance = 1..18 严格递增）。
- 一致性校验：无未引用条目、无悬空引用。

### 7.2 P1：定理 1 的「可计算 Δ」
- 新增脚本 `_measure_delta.py`；产物 `delta_tracking.json`、`delta_tracking_slots.csv`、
  `figures/fig_delta_tracking.png`。
- 协议：k3 非突发，5 seeds × 10 ep，共 5000 时隙；以 MPC-M-H3 轨迹作为最优轨迹的可测代理。
- 结果：Δ 均值 62.1 m / 中位 35.3 m / p90 149.3 m / p99 391.1 m / 最大 540.0 m。
- 单时隙指派代价差均值 **−0.454**（PMEO 更低）；一阶线性化高估 **6.3×**；
  定理 1 型界 Σ L_tΔ_t = **10591.4** vs 实测 442.2 → **界成立、松紧 24×**。
- 解读：界的价值是「可验证包络」而非紧估计；**代价景观沿轨迹偏差方向近乎平坦**（平均相差 62 m 却几乎不付代价），
  与推论 1「增益由决策单元是否改变决定」一致，说明两阶段解耦稳健。
- 边界声明已写入正文与 §5.2 第 1 条：参考轨迹非联合最优，验证的是**尺度关系与界的非空性**，不构成最优性证明。

### 7.3 代码修正（证据一致性）
- `runner.py` 原先把本地触发也记为 100 次「探测」，会与正文「零额外信令」自相矛盾；
  已拆为 `probe_rounds`（真实空口探测）与 `eval_rounds`（本地计算）。
- 证据 CSV 已重跑：oracle 负载触发 = 50 次探测；**LAETS-loc / LAETS-cp = 0 次探测**，数值不变。
- 回归验证：`fixed` / `oracle-load` / `per_user` 三条旧路径正常，无回归。

### 7.4 P1-5：刷新通道不完美时的鲁棒性（§4.5 RQ7）

**问题**：RQ1–RQ6 全部建立在「全量刷新零时延、零丢包」上；若优势只是该理想假设的副产品，结论不可部署。

**设置**：两种最小损伤——刷新延迟 $\Delta_{\mathrm{sync}}=2$ 时隙、刷新丢包 $p_{\mathrm{loss}}=0.1$；
**在每种损伤内部重新扫描固定周期前沿**（$\tau\in\{1,2,3,5,10\}$）与本地触发器，使比较始终针对同条件前沿。
协议 k3 突发，**10 seeds × 10 episodes**；领先量按**种子配对**（同一交通实现内插值前沿）并给 t 检验。
通道模型如实声明：在途不重发（固定周期有效周期 $\to\tau+2$）；丢失于下一时隙重传（实际刷新数≈名义 90%）；初始快照立即可用。

**结果（LAETS-loc $\delta=0.2$ 相对同条件前沿的领先）**

| 刷新通道 | 领先量 | $t$（n=10） | 成功率 | 刷新次数 |
|---|---|---|---|---|
| 理想 | **+226 ± 20** | 11.1（显著） | 0.8874 | 43.6 |
| 延迟 2 时隙 | **+140 ± 13** | 10.9（显著） | 0.7869 | 82.7 |
| 丢包 10% | **+33 ± 24** | 1.4（**不显著**） | 0.8862 | 41.4 |

- **方向稳健**：三种条件下 LAETS-loc 与 oracle 上界都仍位于同条件前沿之上 → 「只在理想通道下成立」的质疑排除。
- **时延比丢包更伤**：延迟下触发器自动提高刷新频率 1.9×（负反馈正确），但成功率 −10.1 个百分点、领先收窄 38%。
- **丢包的绝对代价小、相对优势塌**：成功率几乎不变，但丢包同样扰动固定周期节奏，$\tau=3$ 的 reward 反而
  系统性改善 +383（10 种子方向一致），使同预算前沿整体抬升 → 领先落入噪声。
- **诚实边界**：插值前沿本身对刷新节奏敏感，该边界需更大样本或解析模型复检；已写入 §5.2 第 5 条。

**同时修正的两处证据一致性问题**

1. 旧稿 RQ5/RQ6 把 oracle 触发标为「每 2 时隙探测」，但可复现的 oracle 数值（49.4 / −8162.9）实际来自
   `probe_m=1`（**逐时隙探测，100 次/回合**）。已改标为逐时隙，并说明这是给 oracle 的**最充分信息 + 更高开销**，
   不影响 LAETS-loc 的零信令结论。
2. ω 区间表把 oracle 触发误标为「本地负载」——该触发需要全局真值，已统一改为「**oracle 全局负载**」。

**新增产物**：`experiments/uav_leo_v2x_paper_final/_run_channel_robustness.py`（支持 `--seeds/--out`）、
`_analyze_channel_robustness.py`（种子配对领先量 + t 检验）、`_plot_channel_robustness.py`、
`_emit_rq7_markdown.py`（正文段落由数据直接生成，防止数字漂移）；
数据 `channel_robustness/{channel_robustness_raw.csv,_extra.csv}`、`channel_robustness_summary.json`；
图 `figures/fig_channel_robustness.png`。

**回归校验**：本节 clean 条件在共享的 5 个种子上与 §4.5 已发布数字**逐项一致**
（LAETS-loc δ=0.2：43.5 / −8733.3；δ=0.3：38.0 / −9065.9；oracle：49.4 / −8162.9），
说明不是环境漂移或代码回归，而是上一节的 probe 节奏标注错误。
