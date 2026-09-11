# W4（=P1-lite ④⑤⑥⑦ 辩护：B6/DAG、有效时延、R 话术、chase 叙事）

## 任务
- [x] w/o DAG-order「假成功」量化：scheduler `dep_violations` 已接 env 指标并重跑
  `ablation-v2` 5 seeds（合法行=0；w/o DAG-order 276.2±10.0/集 ≈ 3.3 违例/完成 app，
  见 `PUBLISH_P1.md`「W4」节与 `results/ablation_v2_summary.csv`）。
- [x] B6 辩护：中心格 n=10 B6−Ours 时延 +0.172 s p=0.0137（成功率差 ns +0.0013）；口径
  Ours 决策期拦截 745 vs B6 放行后 doomed 264 → 正文用「相近成功率下时延显著更低」。
- [x] 时延选择偏差口径已文档化（PUBLISH_P1.md + p1_stats_report 全表给 eff_lat）；
  `REPRODUCE.md` §7 补 PUBLISH_P1 引用见本轮更新。
- [x] R≈800 物理话术已写入 PUBLISH_P1.md（UAV-BS 圆盘覆盖文献 + R∈[600,900] 成立区间）；
  SPEC §1.2 注释待论文阶段随引文核对。
- [x] chase 叙事决策 = 方案 (a)，已写入 PUBLISH_P1.md：chase 0.729≈Ours 0.736（p=1.0）但能耗
  更高（+2130 J，p=0.0098）；chase-mover 下 embed vs flat +0.174（p=0.0312）与 patrol 一致。

## 验收
- [x] dep_violations 落盘：`ablation_v2_summary.csv`（w/o DAG-order 276.2±10.0/集 ≈ 3.3 违例/完成 app；其余行 0）。
- [x] 有效时延口径进 p1_stats 主表/敏感性报告。
- [x] R 话术 + chase 决策均已写入 PUBLISH_P1.md。
