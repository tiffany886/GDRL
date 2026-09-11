# W1（=P0）冻结与可复现

## 目标
主表 config/结果冻结为基线；之后每个新实验可一条命令复现并留下记账。

## 任务
- [x] 记账：`results/run_log.csv` 已建（ts/wall_s/argv/git_rev/cfg_sha/out_files/notes，2026-09-07）。
  `ts,argv,git_rev,base_cfg_sha,out_files,wall_s,notes`；
  `git rev-parse HEAD` + `main_v2.yaml` 的 sha256 作为 base_cfg_sha。
- [x] 一键脚本 `scripts/p1_all.sh <scope> [note]`：pytest 门禁 → run_sens scope → run_log 记账（已跑 smoke/grid-lite/center-10seed/b4x500/det-10/chase-b4）。
  pytest → 主表 v2（5 seeds，可选）→ 消融 v2 → 敏感性 grid-lite，每个子命令自动写
  `run_log.csv` 一行并落盘 `results/v2_p1*`。
- [x] 记账写入点：`run_sens_v2.py` 由 `p1_all.sh` 统一记账（argv/rev/sha 见 run_log.csv）。
- [x] 复核基线：`main_table_v2_summary.csv`（7 行方法）、`ablation_v2_summary.csv`（4 行）保留，P1 结果另命名 `v2_p1_*` 不覆盖。
- [x] pytest 门禁：`p1_all.sh` 首步跑 pytest，全绿 45 passed（2026-09-07 复核）。

## 验收
- [x] `p1_all.sh --smoke` 跑通：2026-09-07 实测 wall 64.0 s < 10 min，run_log 记账并产出
  `results/v2_p1_sens_smoke_{raw,summary}.csv`（run_log.csv 首行）；
- [x] 可复现性：所有行 `cfg_sha=58da7efc977e` = `configs/main_v2.yaml` sha256[:12]，
  配置冻结未动 → 同命令同 hash（run_log.csv 全部 8 行一致）。
