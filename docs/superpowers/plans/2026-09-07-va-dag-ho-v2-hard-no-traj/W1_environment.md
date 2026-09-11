# W1：环境加难 + 单测（= SPEC §4 W-A）

> SPEC 映射：`docs/GDRL/disaster_va_dag_ho/SPEC_V2_DS.md` §1.1–1.5 + §4 W-A。
> 代码：`docs/GDRL/disaster_va_dag_ho/{env,configs,baselines,tests}`。
> 原则：v1 行为保持可归档（旧 `main.yaml` 不动）；新开关默认不改变 v1 语义。
> 状态：**已完成（2026-09-07）** — 单测 45 项全绿（26 旧 + 14 env_v2 + 5 run_v2）。
> 关键实现：R<=0 = 软连接（消融 w/o hard-cover 用 `ablation_no_hard_cover.yaml`）；`associate_terminals` 跳过 0 速率对，真实产生 -1。

## W1.1 配置层（config）

- [x] `env/config.py::VAConfig` 新增字段：
  - `uav_cover_radius_m: Optional[float] = None`（None = 软连接/v1，硬覆盖启用时给 400–600）
  - `hotspot_arrivals: bool = False`、`num_hotspots: int = 2`、`hotspot_radius_m: float`
  - `hotspot_centers: Optional[List[float]] = None`（None → 确定性默认中心）
  - `burst_multiplier: float`（突发 [15,35] 全局乘子）
  - `trajectory: str = "patrol"`（主表默认固定轨迹名；`baselines/trajectories.py` 的 key）
- [x] `configs/main_v2.yaml`：N=40、K=3、L=2、T=50，`uav_cover_radius_m: 500`（起调点），
  热点参数 + 注释调参顺序；`configs/main.yaml` 保持 v1 归档不动。
- [x] `env/config.py` 的 `_coerce`/`load_config` 对新增类型（None/float/list/bool）无回归；
  旧 yaml 都能照常加载（回归用 `pytest tests/`）。

## W1.2 硬覆盖（核心，SPEC §1.2）

- [x] `env/channel.py` 增加硬覆盖掩码 helper：水平距离 `||x_n - p_k|| <= R` 才可服务；
  R 为 None 时返回原矩阵（软连接消融/w-o-hard-cover 对照）。
- [x] `env/env.py::_associate`：先用信道矩阵 → 应用硬覆盖掩码（覆盖外 r=0/不进候选）
  → 再走 `associate_terminals`；`_rates_nk` 存掩码后矩阵（调度器用同源）。
- [x] `agents/association.py`：允许 `assignment[n]=-1` 正常返回（已有 fallback，补注释/单测锁定）。
- [x] 每时隙记录 `unassociated_terminal_slots`（Σ 无关联终端数），reset 清零。

## W1.3 连不上规则 A + 指标（SPEC §1.3）

- [x] `AppRequest` 增加 `suffered_uncovered: bool = False`；pending 且 `assignment[terminal]<0`
  的时隙置位。
- [x] 注册逻辑保持：`assignment < 0` 的终端应用不注册/不放置新节点（规则 A 上半）。
- [x] 已 pin 应用在终端脱离覆盖时：掩码后 r=0 → 上行停滞（调度器自然拒放/慢下来）；
  截止期到仍未完成且 `suffered_uncovered` → 失败并按 `fail_uncovered` 计（规则 A 下半）。
- [x] `reset()` 指标新增 `fail_uncovered`（累计）、`unassociated_terminal_slots`（累计）；
  env 暴露 `frac_unassociated` = 累计无关联 /（steps × N）。

## W1.4 热点非齐次到达（SPEC §1.4）

- [x] `env/tasks.py::ArrivalProcess` 支持空间非齐次：每终端 λ = 热点内 `lambda_high_per_s` /
  热点外 `lambda_low_per_s`（仅 `hotspot_arrivals=True` 生效，False 时保持 v1 全局突发语义）。
- [x] 突发窗 [15,35] 可作全局乘子 `burst_multiplier`（默认先 1.0，标定阶段再调）。
- [x] 热点中心确定性：默认 2 个（区域 ~1/3 与 ~2/3 处），可用 `hotspot_centers` 覆盖；
  传入当前 `terminal_pos` 判定在圈内。

## W1.5 轨迹规则模块（SPEC §1.5）

- [x] `baselines/trajectories.py`：统一 `act(env) -> (K,2)` m/s 接口：
  - `TPatrol`（T-patrol，主表默认）：每 UAV 一条走廊条带、条带内来回巡逻、越界翻转；
  - 复用 hover / sweep / chase（不新造轮子，可直接 import `baselines/no_rl.py` 同名类）。
  - `make_trajectory(name)` 调度器，`cfg.trajectory` 默认取 `patrol`。
- [x] 注释明确：主方法 = T-patrol + 硬覆盖关联 + ScheduleDAG（VW-prune + DAG-order）；
  MAPPO 轨迹代码保留但主表不跑。

## W1.6 单测（tests/test_env_v2.py 新增，覆盖 SPEC 要求）

- [x] 配置加载：main_v2.yaml N=40、R=500、热点参数可解析。
- [x] 覆盖外不可关联：远处终端 `assignment=-1`；覆盖内正常关联。
- [x] 覆盖外 pending 应用不能注册/放置（scheduler 无该 app / 无 placement）。
- [x] 走出覆盖导致失败：先 pin 后脱覆盖，到截止期 → `fail_uncovered` +1。
- [x] 热点内外到达率差异：`rate_per_terminal` 圈内 = high、圈外 = low。
- [x] trajectories：TPatrol 输出有界、位置不越界、shape (K,2)。

## W1 验收

- [x] `python -m pytest disaster_va_dag_ho/tests -q` → 26 + 新增全绿；
- [x] v1 旧配置 `main.yaml` 行为回归不受影响（覆盖/热点开关默认关）。
