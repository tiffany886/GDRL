# 选题决定备忘（向汇报对象提交）— 2026-09-09

**一句话结论**：主推 `experiments/uav_leo_v2x_paper_final` 的 **PMEO-M-Eco（Route A）**
成文投稿；`disaster_va_dag_ho`（VA-DAG-HO）的 P0 闸门已触发红线，降为条件地图研究或
Route A 的附录素材，不再单独作为"方法优越性"论文。

## 1. 为什么主推 Route A（PMEO-M-Eco）
- 现有内部证据（`multi_uav_experiment_summary.md`、`reviewer_critique_v2.md`、
  `overhead_analysis.md`）显示该线已有一轮内部红队评审，且存在**可辩护的核心声明**：
  PMEO-M-Eco 在 9 个规模点上对 MPC-M-H3 reward 显著更高（paired p<0.05）且每时隙能耗更低
  （k3 主结果 -2864.4 vs -2955.9，+91.5，p=0.034；能耗 322.7 vs 329.2 J）；
  零训练成本、单时隙 11.1 ms（vs M-TD3/M-DQN 效果差一个量级且需训练）。
- 与导师建议方向（TVT 2025 能效联合优化卸载）一致，素材已覆盖 deadline/drop 压力、
  规模、能耗、复杂度。

## 2. VA-DAG-HO P0/P1 闸门结果（今天跑完，数据真实）
- 知道热点位置的静止基线 B7 在 aligned 布局上成功率/时延不输巡逻、能耗显著更低
  （能耗 +3.2 kJ，p=0.006）→ **红线 A**。
- 热点移到小区内部后，巡逻相对悬停的成功率优势消失（+0.015 / −0.007，n.s.）→ **红线 B**。
- 唯一保得住的最小声明：**无热点先验时**，巡逻显著优于均匀固定站位（aligned +0.095
  p=0.001；corners +0.064 p=0.002），且与"知道热点位置"的 B7 成功率无显著差异。
- 结论：VA-DAG-HO 不写成"我们比 hover/RL 好"，最多写成"零信息轨迹 vs 知情站位的
  条件边界"；需要的时间（重构叙事+补充实验）明显高于其剩余价值。

## 3. 我的处理决定（请审阅）
1. 本周剩余算力（≤24 h）全部投入 Route A 的收口实验与论文重写（差距清单见
   `experiments/uav_leo_v2x_paper_final/GAP_ANALYSIS_2026-09-09.md`）。
2. VA-DAG-HO：停止以"巡逻 vs hover"为主线的任何新实验；保留闸门数据用于汇报与
  附录；是否单独成文由本篇 Route A 进度决定。
3. 所有数据保持可复现（seed/配置/CSV 入库，`run_log.csv` 记账），不做任何"事后选格"。

## 4. 需要汇报对象拍板的两个问题
1. Route A 的头条定位：**"能耗+零训练+低复杂度" vs "奖励在压力条件下显著更优"**——
   我推荐前者（主指标差异更大、更不容易被 marginal-improvement 攻击）。
2. VA-DAG-HO 是否必须在你这里单独成文？如果不是，我按"附录级"处理。
