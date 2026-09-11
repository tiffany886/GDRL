# W2（=P1-lite ①敏感性 + chase）6 格网格

## 网格（其余锁定 main_v2.yaml，seeds 0–4）
- R ∈ {600, 800, 900}，λ_high ∈ {0.1, 0.2}（λ_low 固定 0.03），6 格。
- 每格方法：Ours(T-patrol)、B3-hover（确定性）；B4-flat（每 train seed 250ep，
  train_seed=100+s，评估到同 scene seed s，保证配对）。
- 中心格（R=800, λ_high=0.2）额外：B2-chase（10 seeds 时进 W3）。

## 任务
- [x] `run_sens_v2.py` 支持 scopes：center-smoke / grid-lite / center-10seed / center-b4-500 / det-10 / chase-b4。
  每行 meta 带 `cell,R,lambda_high,method,scene_seed`（确定性行）或
  `cell,R,lambda_high,method,train_seed,eval_seed`（RL 行），写
  `results/v2_p1_sens_{raw,summary}.csv`。
- [x] center-smoke 与冻结主表逐位一致：Ours 0.710526… / hover 0.568807…（PUBLISH_P1.md P0 节）。
- [x] grid-lite 6 格×5 seeds 已落盘：`results/v2_p1_sens_grid_{raw,summary}.csv`。
- [x] 成立区间分析已文档化（`v2_p1_stats_report.txt` + `PUBLISH_P1.md`）：frac∈[0.15,0.45]≈R∈[600,800]
  成立；R=900 覆盖太弱退化向 v1；R=800 是区间内工作点非挑值。热力图可论文阶段补。
- [x] chase 行（c10 + det-10 + chase-b4 mover 对照）已跑；chase-mover 决策见 W4 与 PUBLISH_P1.md chase 叙事节。

## 验收
- [x] 6 格数据齐全，成立区间含 R=600/λ=0.1 解释（硬覆盖弱/强 λ 处 hover 追平，见 PUBLISH_P1）。
- [x] `configs/main_v2.yaml` 未改动（文件时间戳 2026-09-07 02:34 冻结值；`disaster_va_dag_ho/` 为 git 未跟踪目录，以时间戳+diff 复核）。
