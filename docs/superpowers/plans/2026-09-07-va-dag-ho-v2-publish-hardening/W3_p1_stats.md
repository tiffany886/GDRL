# W3（=P1-lite ②③ 统计显著性 + B4 加预算）

## 任务
- [x] `scripts/p1_stats.py` 配对单边 Wilcoxon：Ours−hover / Ours−B4 / B6−Ours 时延（详见 `results/v2_p1_stats_report.txt`）。
  - Ours−hover、Ours−B4（同 scene seed 配对）成功率差：配对单边 Wilcoxon；
  - B6−Ours 时延差（中心格，scoped）：配对单边 Wilcoxon；
  - seeds 0–9 优先；若 0–4 已显著可先报并注明扩样计划。
- [x] 中心格 10 seeds 已跑：`v2_p1_sens_c10_{raw,summary}.csv`（Ours 0.736±0.059 / hover 0.641 / chase 0.729 / B4 0.606）。
- [x] B4 加预算重训（`center-b4-500`）：250ep 0.598 → 500ep 0.597（配对差 −0.001，p=0.81）→「加预算无质变」，flat 已平台。
- [x] 加预算后 B4 未追上 embed（中心格 Ours−B4 +0.131，p=0.001），无 No-Go 触发。
- [ ] B4 收敛曲线图：per-episode CSV 未落盘（run_sens 训练不存曲线），需在论文阶段用 `run_curves.py` 按冻结 config 重训 1 seed 出图；「250→500ep 无质变」已用配对数据支撑，不作为本 goal 阻断。
- [x] 时延口径：p1_stats 报告同时给 succ / cond_delay / eff_lat（公式已实现，deadline_s=12），主表敏感性一致。

## 验收
- [x] 中心格 n=10：Ours−hover +0.0949 p=0.0010；Ours−B4 +0.1306 p=0.0010（Wilcoxon W=55）。
- [x] B4 加预算结论落盘（PUBLISH_P1.md 统计与 B4 预算节：「加预算无质变」）。
- [ ] 显著性差图已用文字表替代（p1_stats_report）；收敛曲线图见上，论文阶段补。
