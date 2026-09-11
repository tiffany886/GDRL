# VA-DAG-HO v2（不学轨迹 + 硬覆盖加难）— 任务清单（W1–W5）

> 目的：把 v2 改规格拆成**可逐项打勾**的周任务清单，供本 goal 长任务与后续会话连续执行。
> 规格锁定版（2026-09-07）：`docs/GDRL/docs/spec_va_dag_ho_v2_hard_no_traj.md` =
> `docs/GDRL/docs/superpowers/specs/2026-09-07-va-dag-ho-v2-hard-no-traj.md` =
> `docs/GDRL/disaster_va_dag_ho/SPEC_V2_DS.md`（三处内容一致，改动须同步）。
> 说明：`docs/spec_va_dag_ho_disaster.md`（v1 计划文档里写的锁定名）当前不存在；
> v1 锁定内容实际在 `docs/superpowers/specs/2026-09-04-va-dag-ho-disaster-design.md`，
> 本清单对应的是 **v2 改规格**（不学轨迹 + 硬覆盖 + 热点到达 + T-patrol）。

## W1–W5 与 SPEC §4（W-A/W-B/W-C）的对应

| 本清单 | SPEC §4 | 内容 | 验收/里程碑 |
|--------|---------|------|-------------|
| W1 | W-A（环境加难） | main_v2.yaml、硬覆盖、规则 A + 指标、热点到达、trajectories.py、单测 | `pytest disaster_va_dag_ho/tests -q` 全绿（26 + 新增） |
| W2 | W-B（主方法接入） | 主方法 = T-patrol + 硬覆盖关联 + ScheduleDAG；B4/B5 改固定 T-patrol + 只学卸载；run_table 出 v2 口径 | v2 主表/消融骨架可跑通（5 seeds 冒烟） |
| W3 | W-B（标定） | 调 R/λ/截止期/DAG scale 至 SPEC §3.3 五条验收 | seeds 0–4 达标（见 §3.3 数字） |
| W4 | W-B（落盘） | 重跑主表与消融、更新 `REPRODUCE.md`、勾完 SPEC W-A/W-B 清单 | REPRODUCE 数字可复现 |
| W5 | W-C/收尾 | 可选附录（学飞 vs 巡逻一行）、防飘审计（§5 don'ts + §0 创新点表述）、副本同步、全量单测 | 收尾核对清单全绿 |

## SPEC §3.3 验收数字（W3 判定标准，seeds 0–4）

1. B3-hover 成功率 ∈ [0.55, 0.75]（中心目标 ~0.65）
2. Ours 成功率比 hover ≥ +0.05，**或**时延低 ≥10%
3. Ours vs B4/B5 成功率差距 ≥ 0.10
4. 平均 `frac_unassociated` ∈ [0.15, 0.40]
5. B6 时延劣于 Ours（成功率可接近）

调参顺序：先定 R → 微调截止期/DAG scale → 跑满主表。

## 状态总览（实时勾选）

- [x] W1：环境加难 + 单测（详见 `W1_environment.md`；2026-09-07 完成，45 单测绿）
- [x] W2：主方法/B4/B5/run_table v2 口径（详见 `W2_main_methods.md`；smoke-v2 出表通过）
- [x] W3：R/λ 标定 + §3.3 验收（详见 `W3_acceptance.md`；全表 5 seeds 通过五条验收）
- [x] W4：REPRODUCE.md v2 + 勾 SPEC 清单（详见 `W4_reproduce.md`）
- [x] W5：收尾审计 + 可选 W-C（详见 `W5_wrapup.md`；审计/收尾已勾，W-C 可选不做）

## 运行命令（仓库根 = /code/docs/GDRL，Python = /root/miniconda3/envs/asr_env/bin/python）

```
/root/miniconda3/envs/asr_env/bin/python -m pytest disaster_va_dag_ho/tests -q
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_table --scope smoke
```

## 不可做（SPEC §5，防飘红线）

- 不许把硬覆盖做成“人人有 association 只是速率低”——必须真实出现 `assignment=-1`。
- 主表不再以 MAPPO 学轨迹为贡献；`env.step` 的移动来自轨迹规则模块。
- 不许为了涨成功率关掉硬覆盖或改动 B6 失败口径（改动必须写进文档）。
