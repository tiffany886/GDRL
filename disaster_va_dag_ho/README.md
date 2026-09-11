# disaster_va_dag_ho

灾后 UAV–LEO 可见窗感知 DAG 分层卸载（VA-DAG-HO）实现目录。

- 规格（锁定 2026-09-04）：`docs/superpowers/specs/2026-09-04-va-dag-ho-disaster-design.md`
- 任务清单（W1–W5）：`docs/superpowers/plans/2026-09-04-va-dag-ho-disaster.md`
- 复现命令与主表记录：`disaster_va_dag_ho/REPRODUCE.md`
- Python：`/root/miniconda3/envs/asr_env/bin/python`；测试：`python -m pytest disaster_va_dag_ho/tests -q`

结构：

```
env/         场景、终端/UAV、简化星历、信道、SoC、因果 step
scheduler/   ScheduleDAG + 可见窗剪枝
agents/      MAPPO 轨迹策略 + 规则关联
baselines/   B1 Random/Sweep、B2 Chase-Backlog、B3 Hover、B4/B5、B6
configs/     *.yaml（主表/消融/扫描）
scripts/     train / run_table / run_curves / sweep / figures
tests/       单测与集成冒烟
results/     CSV + 图（复现见 REPRODUCE.md）
```
