# W5：收尾审计 + 可选 W-C（= SPEC §4 W-C / §5 don'ts / §0 表述）

## W5.1 防飘审计（SPEC §5 + 收尾红线）

- [x] 真实出现 `assignment=-1`：检查主表 frac_unassociated 数据（≥ 0.15 已由 W3 保证；Ours 0.2463±0.0417，逐 seed 0.192–0.295）；
- [x] 主表没有 MAPPO 轨迹：`run_table` v2 分支里 UAV vel 一律来自轨迹模块（B4/B5 速度头惰性，mover=patrol）；
- [x] 硬覆盖没有为了成功率被静默关掉：v2 主表 config 恒定 `uav_cover_radius_m` 非空（main_v2.yaml = 800.0，关掉仅存于 ablation-no-hard-cover 行）；
- [x] B6 口径无文档改动：`vw_prune=False` 语义与 v1 一致（去掉决策期剪枝、执行按跨窗即败，已写入 REPRODUCE §7.4 口径注释）。

## W5.2 表述核对（论文贡献段，SPEC §0/§6）

- [x] 主贡献措辞 = 硬覆盖 + 可见窗下的嵌入调度（非学轨迹；SPEC §0 新主贡献 1–3）；
- [x] 明确删除：MAPPO 学轨迹为主贡献、「联合轨迹与卸载的学习」（SPEC §0 删除/降级表）；
- [x] 与 PMEO「先飞后卸」/ LAETS 同步表述仍错开（SPEC §0 一句话故事末句）。

## W5.3 可选 W-C（仅当 W3 标定发现“太易”才做）

- [ ] 附录：学飞 vs T-patrol 一行对比（非正文必须）；
- [ ] UT–LEO 直连（差信道+短窗）——本轮默认不做。

## W5.4 收尾

- [x] `python -m pytest disaster_va_dag_ho/tests -q` 全绿（2026-09-07 复核：45 passed）；
- [x] 本文档 W1–W5 与 README 状态总览全部打勾；
- [x] 三份 spec 拷贝一致（diff 为空）；REPRODUCE.md v1+v2 章节数字一致可复现（§7 = results/main_table_v2_summary.csv / ablation_v2_summary.csv）。

## W5 验收

- [x] 审计六问全过 → 本 goal 可标 complete 并把 token/耗时上报用户。
