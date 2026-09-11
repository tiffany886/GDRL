# 方案 B 执行计划：机制为主 + 轻量学习模块

> 交给 DeepSeek / 其他助手直接执行。目标：在**不推翻 PMEO 决策顺序主线**的前提下，增加一个「可写进论文的轻量学习模块」，应付「要有模型」的审查口味。
>
> 仓库根：`/code/docs/GDRL`  
> 主实验环境：2026-08-25 修复版 env  
> 主叙事仍以 `experiments/uav_leo_v2x_paper_final/` 与 `docs/thesis_draft_pmeo_laets.md` 为准

---

## 0. 一句话目标

**保留：** post-move exact offloading（决策顺序）+ PMEO / PMEO-M-Eco + LAETS 因果链。  
**新增：** 一个参数量小、可训练、可消融的学习模块（建议优先做「轨迹混合权重学习」），论文表述为：

> 本文以决策顺序敏感的免训练卸载为骨架；进一步提出轻量学习模块自适应轨迹混合权重（或候选打分），相对端到端 DRL 更稳、样本更省。

**禁止：** 把端到端 PPO/GAT/Transformer/GDRL 重新写成「本文提出方法」。

---

## 1. 推荐做什么模块（三选一，默认做 M1）

### M1（推荐，优先实现）：LearnMix — 学习轨迹混合权重

**现状：** 专家轨迹固定  
`target = 0.5 * centroid + 0.5 * hotspot_pred`，再按 `vmax` clip。

**改成：**  
`target = α * centroid + (1-α) * hotspot_pred`，其中 `α ∈ (0,1)` 由小网络根据状态输出（或直接学标量 α，更简单）。

| 项 | 建议 |
|---|---|
| 输入 | 低维手工特征：到质心距离、到热点距离、热点速度模、活跃任务数、队列长度等（≤16 维） |
| 网络 | 2 层 MLP，隐层 32–64，输出 sigmoid → α |
| 训练 | 可用 REINFORCE / PPO 只训 α 头；或网格搜 α 作上界对照 |
| 卸载 | **仍用 post-move exact 枚举**，不学卸载 |
| 卖点 | 「有模型」但极轻；主贡献仍是顺序 |

### M2（备选）：LearnScore — Eco 候选轨迹打分网络

多机 PMEO-M-Eco 的候选集（原地/专家/质心/负载均衡/LEO 附近）用小网络打分代替规则打分。  
实现比 M1 重，和多机代码耦合深。**仅当 M1 一周内做不完再换。**

### M3（备选）：LearnDelta — 学 LAETS 阈值 δ

用小网络或带参策略调 δ。适合强化创新点 2，但「模型感」弱于 M1。可作后续。

**默认决策：先完整落地 M1（单机 + 可选多机复用 α）。**

---

## 2. 论文创新点怎么改写（给写作用）

### 创新点 1（主）

决策顺序敏感的免训练卸载 PMEO / PMEO-M-Eco（post-move exact）。

### 创新点 1 的「模型插件」表述（新增段落，不要升格成单独抢戏的创新点，除非导师强要两个点都偏算法）

在 PMEO 骨架上提出 **LearnMix**：用轻量策略网络自适应轨迹混合权重 α，卸载层保持精确枚举。

### 创新点 2（保持）

LAETS：负载感知事件触发同步；孪生新鲜度是顺序收益前提。

### 与端到端 DRL 的差异句

> 端到端 DRL 同时学飞与卸，样本贵且在本场景增益有限；LearnMix 仅学一维/低维轨迹偏好，决策顺序与卸载最优性由机制保证。

---

## 3. 实现任务清单（按顺序做）

### Phase 0：对齐代码入口（0.5 天）

1. 定位单机专家轨迹实现位置（约在 `uav_leo_experiment/baselines.py` / `mpc_traj.py` / `hado.py` 等，搜 `centroid`、`expert`、`0.5`）。
2. 定位 PMEO 评估入口（`run_experiment.py`、paper_final 下 eval 脚本）。
3. 确认奖励与 `energy_weight`、`drop_penalty` 与主表一致。
4. 写一页 `docs/learnmix_design.md`：输入特征定义、网络结构、训练超参、文件路径。

**验收：** 能指出「改哪几个函数」；跑通现有 PMEO hard 一次。

### Phase 1：实现 LearnMix 推理（1 天）

1. 新增模块，例如：  
   `uav_leo_experiment/learn_mix.py`  
   - `class LearnMixPolicy`  
   - `build_features(env_state) -> np.ndarray`  
   - `forward -> alpha`  
   - `expert_target(centroid, hotspot, alpha)`
2. 在 PMEO 轨迹步替换固定 0.5/0.5。
3. CLI：`--traj_mode fixed|learnmix`，`--learnmix_ckpt path`。
4. 未加载权重时：α=0.5，行为与原 PMEO **比特级可复现**（同一 seed）。

**验收：** `traj_mode=fixed` 与旧 PMEO hard 40ep reward 差 < 0.1。

### Phase 2：训练脚本（1–2 天）

1. 新增 `uav_leo_experiment/train_learnmix.py`（或并入现有 `train_drl.py` 的小模式）。
2. 建议设定（可改，但先固定一套）：  
   - 场景：`v2x_hotspot_hard`（先单机）  
   - 步数：5k–20k（远小于端到端 40k 全动作）  
   - 算法：PPO 或 REINFORCE + baseline；**只更新 LearnMix 参数**  
   - 卸载：每步仍调用 post-move exact  
   - seed：至少 3 个（如 73, 74, 75）
3. 保存：`experiments/uav_leo_v2x_paper_final/learnmix/hard_seed{}.pt` + `train_curve.csv`。
4. 日志：episode reward、α 均值/方差、成功率、能耗。

**验收：** 训练曲线能出图；推理可加载 ckpt。

### Phase 3：对比实验（1–2 天）

在 **同一 seed 池、同一 env** 下跑：

| 方法 | 说明 |
|---|---|
| PMEO (α=0.5 fixed) | 原主方法 |
| LearnMix (trained) | 新模块 |
| α grid best | 对 α∈{0.1,0.3,0.5,0.7,0.9} 扫最优（无学习上界） |
| Current-pos exact | 顺序消融（证明主贡献仍在顺序） |
| PPO / 现有最强 DRL 之一 | 端到端对照 |
| MPC-H3 | 免训练强基线 |

场景：至少 **hard**；有余力加 **stress**。  
集数：与主表一致优先 40 ep 配对；报告 mean 与 vs PMEO 的 Δ。

**验收表（写入 `experiments/uav_leo_v2x_paper_final/learnmix_results.md`）：**

- LearnMix vs PMEO：Δ reward、是否显著  
- LearnMix vs 端到端 DRL：仍应明显更好或相当但训练成本更低  
- Current-pos 仍显著差于 post-move（顺序主线不能坏）  
- 训练步数 / 墙钟时间 vs 端到端 DRL  

**结果解读规则（必须遵守）：**

- 若 LearnMix ≈ PMEO（Δ<0.5%）：论文写「轨迹权重可学但增益有限，进一步说明顺序与精确卸载是主因；LearnMix 提供自适应接口」。仍算完成 B（有模型+诚实）。  
- 若 LearnMix 明显更好：写成「轻量自适应轨迹 + 顺序正确卸载」。  
- 若 LearnMix 更差：查 bug；不要强行写进主方法。

### Phase 4：多机（可选，+1 天）

若单机完成：把同一 α 网络接到 PMEO-M 的「专家移动」候选上（每机一个 α 或共享）。  
与 PMEO-M-Eco 比：不必强求超过 Eco；超过更好，不超过就写「可与 Eco 候选集组合」。

### Phase 5：写作与图（1 天）

1. 更新 `docs/thesis_draft_pmeo_laets.md`：  
   - 3.2 后增加「3.2.x LearnMix 轻量轨迹自适应」  
   - 结构图：在 PMEO 流程「②启发式」改为「② LearnMix 输出 α + 混合」  
2. 画/改图：PMEO+LearnMix 框图；训练曲线；α 分布直方图。  
3. 更新飞书论文页与汇报通读稿中「方案 B」说明。  
4. Related Work 加 2–3 句：残差策略 / 参数化启发式 / learning-to-adapt 与端到端区别。

---

## 4. 文件改动预期（供 PR/提交）

```
uav_leo_experiment/learn_mix.py          # 新
uav_leo_experiment/train_learnmix.py     # 新
uav_leo_experiment/baselines.py 或轨迹相关文件  # 接 α
docs/learnmix_design.md                  # 新
experiments/uav_leo_v2x_paper_final/learnmix/  # ckpt + curves
experiments/uav_leo_v2x_paper_final/learnmix_results.md
docs/thesis_draft_pmeo_laets.md          # 增节
docs/GDRL/docs/make_thesis_figures_cn.py # 可选：加 LearnMix 框图
```

---

## 5. 明确不要做的事

1. 不要把 GDRL/GAT-PPO 改称「本文方法」。  
2. 不要为了有模型取消 post-move exact。  
3. 不要在摘要里只吹 LearnMix、不提决策顺序。  
4. 不要宣称 LearnMix「全面 SOTA」；单机仍可能略逊 MPC。  
5. 不要一上来就上 Transformer；MLP 足够。

---

## 6. 给 DeepSeek 的系统提示（可复制）

```
你在 /code/docs/GDRL 实现方案 B：机制为主 + 轻量学习模块 LearnMix。

硬约束：
1) 主贡献仍是 post-move exact offloading（决策顺序）与 PMEO；LearnMix 只学轨迹混合权重 α。
2) 卸载继续枚举；禁止用端到端网络替代卸载。
3) traj_mode=fixed 且 α=0.5 时必须与原 PMEO 数值对齐。
4) 先做单机 v2x_hotspot_hard；输出 learnmix_results.md。
5) 若 LearnMix 相对 PMEO 增益极小，诚实写入结果，不要编造。

请按 docs 中「方案 B 执行计划」Phase 0→5 执行；每阶段结束给出改了哪些文件、如何复现命令、关键数字。
```

---

## 7. 时间盒建议

| 阶段 | 时间 |
|---|---|
| Phase 0–1 | 1.5 天 |
| Phase 2 | 1–2 天 |
| Phase 3 | 1–2 天 |
| Phase 4 | 可选 1 天 |
| Phase 5 | 1 天 |
| **合计** | **约 5–7 天（不含可选多机）** |

---

## 8. 完成定义（Definition of Done）

- [ ] LearnMix 代码可训练、可推理、可关断（回退 PMEO）  
- [ ] hard 上有对照表：PMEO / LearnMix / Current-pos / 至少 1 个 DRL / MPC-H3  
- [ ] 顺序消融结论未被破坏  
- [ ] 论文草稿有独立小节 + 1 张结构图 + 训练曲线  
- [ ] 摘要/创新点口径：机制主、LearnMix 辅  

---

## 9. 风险与预案

| 风险 | 预案 |
|---|---|
| 学完 α 仍≈0.5 | 保留模块作「自适应接口」；主文强调顺序；可加 α grid 显示场景依赖 |
| 训练不稳 | 减输入维、学标量 α、加 BC 到 0.5、减学习率 |
| 导师仍嫌模型太小 | 再加 M2 候选打分，仍不要端到端替代卸载 |
| 时间不够 | 只交 M1 单机 hard + 写作；stress/多机标 future work |
