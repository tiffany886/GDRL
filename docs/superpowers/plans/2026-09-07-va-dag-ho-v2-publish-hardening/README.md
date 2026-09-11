# VA-DAG-HO v2 投稿加固 — 任务清单（P0–P4，goal = P0 → P1-lite）

> 目的：把「调参凑出来的验收」升级为「可辩护、可复现、统计显著的投稿证据」。
> 依据 2026-09-07 评审定稿：**P1 必做（精简网格）、P2 机制图精简版必做、MILP 与完整 P3 降为可选闸门、P4 紧随**。
> 定位目标：IoT-J / 中等会议（Go 低配）；TVT/TWC 需再开 P3（可选闸门）。
> 主表 config 冻结：`disaster_va_dag_ho/configs/main_v2.yaml`（= 2026-09-07 锁值）。敏感性只做**分析**，不改锁值。

## W1–W5 ↔ P0–P4 对应

| 本清单 | 阶段 | 内容 | 验收 |
|--------|------|------|------|
| W1 | P0 冻结/可复现 | run 记账（config hash + git rev + 墙钟 + 结果）、一键跑全套脚本 | 任意新实验一条命令复现，`results/run_log.csv` 有记录 |
| W2 | P1-lite 敏感性 | 6 格网格 R∈{600,800,900}×λ_high∈{0.1,0.2}，Ours/hover/B4（中心格 +chase）；「结论成立区间」 | 6 格×5 seeds 数据落盘，成立区间图/表，主表未动 |
| W3 | P1-lite 统计 | 配对 Wilcoxon（Ours−hover、Ours−B4、B6−Ours 时延）、B4 加预算、收敛曲线、10 seeds | 中心格 Ours−hover / Ours−B4 p<0.05；B4 加预算不追上 embed |
| W4 | P1-lite 辩护 | B6/DAG-order「假成功」量化、时延选择偏差（有效时延口径）、R≈800 物理话术、chase 叙事 | 每个 red-team 点有一句量化证据 + 文档 |
| W5 | Go/No-Go + P4 预装 | 判定标准、P2/P3 可选闸门、P4 红队清单（含 chase≥Ours）、文档同步 | Go/No-Go 结论 + 红队 80% 有证据 |

## Go / No-Go（2026-09-07 评审改严版）

- **No-Go**：敏感性中心格 Ours−hover 或 Ours−B4 **不显著**；或 B4 加预算后追上 embed（差 <0.10 或不再显著）。
- **Go · IoT-J/中等会议**：W1–W4 全绿 + W5 红队 80% 有量化证据；MILP 不做也可。
- **Go · TVT/TWC**：Go(IoT-J) 之上 + P3 至少一维规模（N∈{30,40,60}）或换热点；chase 显著或诚实讨论；MILP 可选。

## 红线（全程）

- 不改 `main_v2.yaml` 锁值凑指标；敏感性网格是分析不是刷榜。
- 不关硬覆盖、不改 B6 失败口径；改动必入档。
- 每个实验跑完立即落盘 `results/`（命名 `v2_p1*`），并写 `run_log.csv`。
- 时延口径禁止只报成功样本均值 → 用「成功率+条件时延」联合或「有效时延」（失败按截止期计罚）。

## 运行命令（仓库根 = /code/docs/GDRL，Python = /root/miniconda3/envs/asr_env/bin/python）

```
/root/miniconda3/envs/asr_env/bin/python -m pytest disaster_va_dag_ho/tests -q
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_sens_v2 --scope center-smoke
/root/miniconda3/envs/asr_env/bin/python -m disaster_va_dag_ho.scripts.run_sens_v2 --scope grid-lite --seeds 0,1,2,3,4
```

## 状态总览（实时勾选）

- [x] W1：P0 冻结/记账/一键脚本（2026-09-07 完成；`run_log.csv` + `p1_all.sh`）
- [x] W2：P1-lite 敏感性 6 格 + chase（grid/c10/det10/chaseb4/b4x500 全落盘）
- [x] W3：P1-lite 统计显著性 + B4 加预算（中心格 n=10 全部显著；加预算无质变）
- [x] W4：P1-lite 辩护（B6 显著、DAG 假成功 276.2/集、eff_lat 口径、R 话术、chase 决策 a）
- [x] W5：Go/No-Go 判定 = **Go(IoT-J/中等会议)**（判定+红队归档见 `W5_go_nogo.md`）
- [ ] W5 备注（论文阶段，非阻断）：B4 收敛曲线图（重训 1 seed 按冻结 config）、P2 机制图、
  P2-MILP / P3 规模扫为可选闸门，Go(TVT/TWC) 时再开
