# W3：R/λ 标定 + §3.3 验收（= SPEC §4 W-B 后半）

> SPEC 映射：`SPEC_V2_DS.md` §3.3（验收数字）+ §4 W-B「调 R 直到验收通过」。
> 方法：先定 `uav_cover_radius_m` 让 hover 成功率与 frac_unassociated 落窗，
> 再微调 deadline/DAG scale / burst_multiplier，最后跑满主表 5 seeds 复核。
> 探针记录（2026-09-07，seeds 0–2 均值，v2 默认 λ/截止期）：
> R=450 hover 0.425/0.642 · R=500 0.482/0.592 · R=550 0.573/0.533 ·
> R=600 0.645/0.458 · R=650 0.667/0.400 · R=700 0.690/0.367（hover 成功率/frac_unassociated）。
> band-patrol（现 TPatrol）：R=650 0.709/0.400；R=700 0.715/0.353。
> 标定落点（2026-09-07，seeds 0–4，app_deadline_s=12，hotspot_centers=[500,150,1500,1850]）：
> R=800 → hover 0.669±0.063 / frac 0.245；Ours(patrol) 0.745±0.063（+0.076 ✓）；
> B6(无剪枝) 时延 1.684 > Ours 1.613 ✓（成功率接近，允许）。
> 已写入 configs/main_v2.yaml（uav_cover_radius_m=800、hotspot_centers 固定）。
> 完成（2026-09-07 全表复跑）：B4 0.5705±0.0437、B5 0.5967±0.0241 → Ours−B4 +0.161、
> Ours−B5 +0.134（均 ≥0.10 ✓）；§3.3 五条全部通过，见 REPRODUCE.md §7.2。

## W3.1 快速标定（hover-only 探针）

- [x] hover 成功率探针：`R ∈ {400, 450, 500, 550, 600}` × seeds 0–4，记录
  `apps_done/apps_arrived` 与 `frac_unassociated`；
- [x] 目标 1+4：hover ∈ [0.55, 0.75]（~0.65）且 frac_unassociated ∈ [0.15, 0.40]；
- [x] 若 hover > 0.80 → R 调小；若 < 0.45 → R 调大（SPEC §1.2 指引）；
- [x] 把选定的 R 写回 `configs/main_v2.yaml` 与 REPRODUCE 口径注释。

## W3.2 主指标复核（seeds 0–4）

- [x] 跑 Ours（T-patrol）与 hover：Ours 成功率 ≥ hover + 0.05，或时延 ≤ hover×0.9；
- [x] Ours vs B4（flat RL offload-only）、B5（SAC-flat）：成功率差 ≥ 0.10；
- [x] B6（无决策期窗剪枝）时延劣于 Ours（成功率可接近）；
- [x] 任一主指标不满足 → 回来微调 deadline_margin / app_deadline / dag_cycles_scale /
  burst_multiplier（每次调参记录到 REPRODUCE 或本文件历史区）。

## W3.3 消融复核

- [x] w/o VW-prune(=B6)、w/o DAG-order、w/o hard-cover、w/o embed(=B4) 四消融 ×5 seeds；
- [x] 消融结果方向与正文主张一致（硬覆盖改变问题、嵌入优势稳健）。

## W3 验收

- [x] §3.3 五条在 seeds 0–4 全部满足，数据落在 `results/`（v2 命名避免覆盖 v1）。
