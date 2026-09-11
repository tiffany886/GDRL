# W4：结果落盘 + REPRODUCE.md v2（= SPEC §4 W-B「更新 REPRODUCE.md」）

## W4.1 数据落盘

- [x] v2 主表 CSV（B1–B7/B6 rows ×5 seeds，均值±std）与消融 CSV 落到
  `disaster_va_dag_ho/results/`（新文件名如 `main_table_v2_summary.csv`、`ablation_v2_summary.csv`，不覆盖 v1）；
- [x] 记录 wall-clock 与学习法 episodes（B4/B5 训练曲线留档可选）；
- [x] 记录 `frac_unassociated`、`fail_uncovered` 主表值（正文/附录素材）。

## W4.2 REPRODUCE.md 更新

- [x] `disaster_va_dag_ho/REPRODUCE.md` 增加 v2 章节：v2 配置、命令、
  预期主表数字、B4/B5/B6 口径注释、R 调参历史；
- [x] 保留 v1 章节（归档可复现），顶部注明 v1/v2 差异。

## W4.3 勾选 SPEC 清单

- [x] `disaster_va_dag_ho/SPEC_V2_DS.md` §4 W-A 全部打勾；
- [x] §4 W-B 全部打勾（W-C 保持未勾，除非做了附录）；
- [x] 同步勾 `docs/GDRL/docs/spec_va_dag_ho_v2_hard_no_traj.md` 与
  `docs/GDRL/docs/superpowers/specs/2026-09-07-va-dag-ho-v2-hard-no-traj.md`（三处一致）；
- [x] 勾完本文件夹 W1–W4 全部 checkbox，README 状态总览同步。

## W4 验收

- [x] 按 REPRODUCE v2 章节能从零复现主表至少一行；spec 三份拷贝 diff 为空。
