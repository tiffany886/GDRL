# W2：主方法接入与基线改造（= SPEC §4 W-B 前半）

> SPEC 映射：`SPEC_V2_DS.md` §2（v2 主方法定义）+ §3.1（B1–B6 口径）+ §4 W-B 前两条。
> 代码：`disaster_va_dag_ho/{env,agents,baselines,scripts}`。
> 目标：v2 环境下「移动 = 轨迹规则模块」，RL 只可能出现在 B4/B5 的**卸载决策头**。
> 状态：**已完成（2026-09-07）** — Mappo `flat_no_vel`、SacFlat `offload_only=True + mover`、trainer `mover`、run_table scopes `v2/ablation-v2/smoke-v2`（基底 main_v2.yaml）；`--scope smoke-v2` 出表通过。

## W2.1 轨迹模块接入 env/训练回路

- [x] `env/env.py::step` 保持现有签名 `step(uav_actions, offload_decisions)`；
  主方法/基线在 runner 层把 `uav_actions` 换成 `trajectories.make_trajectory(cfg.trajectory)` 输出。
- [x] `scripts/run_table.py` / `scripts/train.py`：v2 scope 用轨迹模块提供 vel，
  不再调用 MAPPO 的 `act()["vel"]`（新增 v2 runner 分支或 `--trajectory` 参数）。

## W2.2 B4/B5 改「固定 T-patrol + 只学卸载」

- [x] `agents/mappo.py`：为 flat 模式提供 `mode="offload_only"`（丢弃/禁用 vel 头；
  或调用侧忽略其 vel 并喂入 T-patrol 动作）。
- [x] `baselines/sac_flat.py`（及 td3_flat 参考实现）同样支持卸载-only：act 返回
  `{"vel": 由外部轨迹给出, "dec": 学到的卸载}`；env 移动不消耗它们的 vel。
- [x] B1–B3/B6 轨迹固定：B1 random、B2 chase、B3 hover、**B6 = T-patrol + vw_prune=False**。
- [x] 口径注释写入 config/代码 docstring：B4/B5 卸载动作空间 = {0 UAV, 1 LEO, 2 wait}
  （沿用现有 `offload_decisions` 槽位），速度与 Ours 相同（T-patrol）。

## W2.3 run_table v2 scope 与消融骨架

- [x] `run_table.py` 支持 v2 主表 rows：B1/B2/B3/Ours(T-patrol)/B4/B5/B6，5 seeds。
- [x] 消融 scope：w/o VW-prune(=B6)、w/o DAG-order、w/o hard-cover（软连接对照）、
  w/o embed(=B4)；输出 CSV 字段含 `success / delay / energy / invalid / frac_unassociated / fail_uncovered`。
- [x] `scripts/eval_baselines.py`/`_stats` 补充 v2 指标统计（含 hover 成功率口径 = apps_done/apps_arrived）。

## W2 验收

- [x] `run_table --scope smoke`（v2 配置，2 seeds、少 episodes）能出表无异常；
- [x] 主方法 episode 内 UAV 位置变化确实来自 T-patrol（可用小测试断言）。
