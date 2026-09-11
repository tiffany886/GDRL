# 数字孪生驱动的 UAV-LEO 边缘计算 —— 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。
> 注：按仓库约定不自动 git commit；如需要提交请在任务完成后请用户确认。
> 执行状态：2026-08-28 全部任务完成（9 项测试通过，RQ1-4 + k3 数据齐，图/汇总/README 已产出）。RQ3 用 RQ1 数据提取（methods×τ 覆盖）。

**目标：** 在现有 UAV-LEO 仿真器上加数字孪生（DT）层，一周内产出同步周期-性能权衡（RQ1）、预测模型/误差鲁棒性（RQ2）、决策器鲁棒性（RQ3）、误差触发自适应同步（RQ4）四组指标结果。

**架构：** 物理世界 = 现有 `env.py`；数字孪生 = `TwinnedWorld`（每 τ 时隙同步一次，间隙内用 predictor 外推 user/hotspot 位置与任务）；决策器复用现有策略（PMEO / PMEO-M-Eco / MPC-H3），只改输入（喂孪生 obs，UAV 位置用物理真值）。

**技术栈：** Python 3.10 / numpy / 现有 `uav_leo_experiment` 包 / pytest（`/root/miniconda3/envs/asr_env/bin/python`）。

**规格：** `docs/superpowers/specs/2026-08-28-digital-twin-uav-leo-design.md`

---

## 文件结构

```
digital_twin_uav_leo/            # 新建包（位于 /code/docs/GDRL/ 下）
  __init__.py                    # 空
  dt_env.py                      # TwinnedWorld：sync / step_forward / get_obs / probe_error
  predictors.py                  # freeze / linear / linear_noise
  adaptive_sync.py               # 误差估计 + 阈值触发逻辑（可与 dt_env 合并，见 Task 3）
  runner.py                      # run_dt_episode：孪生驱动的单 episode 评估，输出指标 dict
  run_experiments.py             # RQ1-4 CLI：加载场景、批量跑、写 CSV
  make_figures.py                # 权衡曲线 + 自适应对比图
  results/                       # CSV + summary.md（gitignore 之外，需建目录）
tests/
  test_dt_env.py                 # Task 1/2/3 单测
  test_dt_runner.py              # Task 4 集成冒烟
README.md 修改                   # 顶部加一段 DT 方向入口（Task 8 合并进 digital_twin_uav_leo/README.md）
```

关键接口（已核实，勿凭记忆）：
- 场景配置：`from .config import make_config`；`cfg = make_config("v2x_hotspot_hard", ablation="none", seed=..., episodes=..., uavs=1)`；k3 多 UAV 参考 `uav_leo_experiment/run_multi_uav.py:43-60` 的 `build_config`（preset `k3` = 3 UAV / 51 用户 / 3 热点 / 2 km）。
- 环境：`env = Env(cfg)`；`obs = env.observation()`；`obs, reward, done, info = env.step(action)`；`env.reset(seed)`。
- 决策器：`action = policy.act(obs, rng, config)`，action 为 `{"move": (K,2) 或 (2,), "targets": (U,) int, "ratios": (U,) float}`。
- 单 UAV 策略：`baselines.PostmoveExactPolicy()`、`baselines.CurrentExactPolicy()`（若名不同用 `multi_uav_policies` / `run_multi_uav.py` 里注册名核对）；MPC：`mpc_traj.MPCTrajPolicy(horizon=3, offload="exact")`；多 UAV：`multi_uav.PmeoMEcoPolicy()`。
- 热点反弹预测：`mpc_traj.predict_hotspot(obs, config, k)`（k 为未来时隙数）。
- obs 字段：`t, user_pos, user_vel, uav_pos, leo_pos, task_bits, cycles_per_bit, uav_backlog, leo_backlog, uav_battery, hotspot_pos, hotspot_vel`（`env.observation()`，见 `env.py:160-186`）。注意 `hotspot_pos` 在 M=1 时是 1D，M>1 时是 (M,2)。

---

## Task 1：包脚手架 + TwinnedWorld 核心（同步/外推/观测视图）

**文件：**
- 创建：`digital_twin_uav_leo/__init__.py`、`digital_twin_uav_leo/dt_env.py`
- 测试：`tests/test_dt_env.py`

- [x] **步骤 1：写失败的测试**

```python
# tests/test_dt_env.py
import numpy as np
from uav_leo_experiment.config import make_config
from uav_leo_experiment.env import Env
from digital_twin_uav_leo.dt_env import TwinnedWorld

def _cfg(seed=1):
    return make_config("v2x_hotspot_hard", ablation="none", seed=seed,
                       episodes=2, uavs=1)

def test_sync_mirrors_physical_obs():
    cfg = _cfg()
    env = Env(cfg); env.reset(seed=1)
    tw = TwinnedWorld(env, tau=2, predictor="linear", eps=0.0, rng=np.random.default_rng(0))
    obs = env.observation()
    tw.sync(obs)
    tw_obs = tw.get_obs()
    # 关键：UAV 位置必须用物理真值
    assert np.allclose(tw_obs["uav_pos"], obs["uav_pos"])
    assert np.allclose(tw_obs["user_pos"], obs["user_pos"])
    assert np.allclose(tw_obs["task_bits"], obs["task_bits"])

def test_no_sync_between_sync_points():
    cfg = _cfg()
    env = Env(cfg); env.reset(seed=1)
    tw = TwinnedWorld(env, tau=2, predictor="linear", eps=0.0, rng=np.random.default_rng(0))
    tw.sync(env.observation())
    p0 = tw.get_obs()["user_pos"].copy()
    tw.step_forward(dt=1.0)          # 间隙内推进一次
    p1 = tw.get_obs()["user_pos"]
    assert p1.shape == p0.shape       # 外推不改变形状
    assert tw.since_sync == 1
```

- [x] **步骤 2：运行测试确认失败**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：FAIL，`ModuleNotFoundError: digital_twin_uav_leo`

- [x] **步骤 3：实现 `dt_env.py`**

要点：
- `TwinnedWorld.__init__(self, env, tau, predictor="linear", eps=0.0, rng=None, sync_mode="fixed", probe_m=2, probe_n=3, delta=10.0)`。
- 内部保存 `self._pred`（dict 镜像 obs 中需要预测的字段）与 `self._obs_physical`（物理 obs 引用，供 UAV 位置真值 + 误差探测）。
- `sync(obs)`：拷贝 `user_pos/user_vel/hotspot_pos/hotspot_vel/task_bits/cycles_per_bit/leo_pos/leo_backlog/uav_backlog/t`，置 `since_sync=0`，`sync_count+=1`。
- `step_forward(dt)`：调用 `predictors.predict(self._pred, dt, cfg, rng, eps)` 更新预测字段；`since_sync += 1`。
- `get_obs()`：返回 `self._obs_physical` 的副本，但 `user_pos/task_bits/...` 替换为 `self._pred` 值；`uav_pos/uav_battery` 保留物理真值。
- 需要 `env.config` 访问场景参数；`env` 持有物理状态引用（供 `step_forward` 后 probe 用）。

- [x] **步骤 4：运行测试确认通过**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：PASS（2 passed）

- [x] **步骤 5：核对 API 与设计文档一致**（τ、predictor、ε、sync_mode 字段与规格 §2.2 一致）

---

## Task 2：predictors.py（freeze / linear / linear_noise）

**文件：**
- 创建：`digital_twin_uav_leo/predictors.py`
- 测试：追加 `tests/test_dt_env.py`

- [x] **步骤 1：写失败的测试**

```python
def test_linear_user_extrapolation():
    cfg = _cfg()
    env = Env(cfg); env.reset(seed=1)
    pred = {"user_pos": np.array([[100., 100.]]), "user_vel": np.array([[2., 0.]]),
            "hotspot_pos": None, "hotspot_vel": None}
    from digital_twin_uav_leo.predictors import predict
    out = predict("linear", pred, dt=1.0, cfg=cfg, rng=None, eps=0.0)
    assert np.allclose(out["user_pos"], [[102., 100.]])

def test_linear_noise_injects_error():
    cfg = _cfg()
    env = Env(cfg); env.reset(seed=1)
    pred = {"user_pos": np.zeros((4, 2)), "user_vel": np.zeros((4, 2)),
            "hotspot_pos": None, "hotspot_vel": None}
    rng = np.random.default_rng(0)
    from digital_twin_uav_leo.predictors import predict
    out1 = predict("linear_noise", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.1)
    out2 = predict("linear_noise", pred, dt=1.0, cfg=cfg, rng=rng, eps=0.1)
    assert not np.allclose(out1["user_pos"], out2["user_pos"])  # 有随机性

def test_freeze_keeps_positions():
    cfg = _cfg()
    env = Env(cfg); env.reset(seed=1)
    pred = {"user_pos": np.array([[1., 2.]]), "user_vel": np.array([[9., 9.]]),
            "hotspot_pos": None, "hotspot_vel": None}
    from digital_twin_uav_leo.predictors import predict
    out = predict("freeze", pred, dt=1.0, cfg=cfg, rng=None, eps=0.0)
    assert np.allclose(out["user_pos"], [[1., 2.]])
```

- [x] **步骤 2：运行测试确认失败**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：FAIL，`ModuleNotFoundError: digital_twin_uav_leo.predictors`

- [x] **步骤 3：实现 `predictors.py`**

要点：
- 统一入口 `predict(name, pred, dt, cfg, rng, eps)`，原地更新并返回 `pred`。
- `freeze`：不改任何字段。
- `linear`：`user_pos += user_vel * dt`；`hotspot_pos` 存在时用 `mpc_traj.predict_hotspot(obs_like, cfg, 1)`（构造临时 obs dict 传入，注意 hotspot_pos 格式：M=1 是 1D，包一层）；`task_bits` 冻结（规格 §2.2）。
- `linear_noise`：在 linear 基础上，对 `user_pos/hotspot_pos` 加 `rng.normal(0, eps * scale, shape)`，`scale = max(‖user_vel‖*dt, 1.0)`（间隙预期移动距离，规格 §2.2）；不注入 task_bits。
- 用 numpy 向量化，用户数/维度从数组形状推导。

- [x] **步骤 4：运行测试确认通过**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：PASS（5 passed）

---

## Task 3：adaptive_sync.py（误差估计 + 阈值触发）

**文件：**
- 创建：`digital_twin_uav_leo/adaptive_sync.py`
- 测试：追加 `tests/test_dt_env.py`

- [x] **步骤 1：写失败的测试**

```python
def test_probe_error_and_threshold():
    from digital_twin_uav_leo.adaptive_sync import estimate_error, should_resync
    obs_pred = {"user_pos": np.zeros((6, 2)), "hotspot_pos": None}
    obs_true = {"user_pos": np.zeros((6, 2)), "hotspot_pos": None}
    err = estimate_error(obs_pred, obs_true, n_probe=6, rng=np.random.default_rng(0))
    assert err < 1e-9
    assert not should_resync(err, delta=5.0)
    obs_true["user_pos"] = np.full((6, 2), 10.0)
    err = estimate_error(obs_pred, obs_true, n_probe=6, rng=np.random.default_rng(0))
    assert err > 5.0
    assert should_resync(err, delta=5.0)
```

- [x] **步骤 2：运行测试确认失败**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：FAIL，`ModuleNotFoundError: digital_twin_uav_leo.adaptive_sync`

- [x] **步骤 3：实现 `adaptive_sync.py`**

要点：
- `estimate_error(obs_pred, obs_true, n_probe, rng)`：随机抽 `n_probe` 个用户（与热点，若有），返回最大欧氏距离（max over probes）作为当前误差。
- `should_resync(error, delta)`：`error > delta` 返回 True。
- `TwinnedWorld` 在 `sync_mode="adaptive"` 时：每 `probe_m` 个时隙调用 `estimate_error`（孪生 obs vs 物理 obs），超 `delta` 即 `sync(obs_physical)`。在 Task 4 集成时接入。

- [x] **步骤 4：运行测试确认通过**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py -v`
预期：PASS（7 passed）

---

## Task 4：runner.py —— 孪生驱动 episode 评估 + 决策一致率

**文件：**
- 创建：`digital_twin_uav_leo/runner.py`
- 测试：`tests/test_dt_runner.py`

- [x] **步骤 1：写失败的测试**

```python
# tests/test_dt_runner.py
import numpy as np
from uav_leo_experiment.config import make_config
from uav_leo_experiment.env import Env
from uav_leo_experiment.baselines import PostmoveExactPolicy
from digital_twin_uav_leo.runner import run_dt_episode

def test_run_dt_episode_smoke():
    cfg = make_config("v2x_hotspot_hard", ablation="none", seed=1,
                      episodes=2, uavs=1)
    env = Env(cfg); env.reset(seed=1)
    policy = PostmoveExactPolicy()
    rng = np.random.default_rng(0)
    out = run_dt_episode(env, policy, cfg, tau=3, predictor="linear",
                         eps=0.0, sync_mode="fixed", seed=1)
    for k in ("reward", "latency", "energy", "completion", "agree", "syncs", "perfect_reward"):
        assert k in out
    assert 0.0 <= out["completion"] <= 1.0
    assert 0.0 <= out["agree"] <= 1.0
```

- [x] **步骤 2：运行测试确认失败**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_runner.py -v`
预期：FAIL，`ModuleNotFoundError: digital_twin_uav_leo.runner`

- [x] **步骤 3：实现 `runner.py`**

要点：
- `run_dt_episode(env, policy, cfg, tau, predictor, eps, sync_mode, seed, probe_m=2, probe_n=3, delta=10.0)`：
  1. `env.reset(seed)`；`tw = TwinnedWorld(...)`；`tw.sync(env.observation())`；
  2. 每时隙：`tw.step_forward(dt)`（先外推）→ adaptive 模式做探针检查 → `obs_dt = tw.get_obs()` → `action = policy.act(obs_dt, rng, cfg)` → `obs_phys, reward, done, info = env.step(action)` → `tw` 记录物理 obs（供下一次 probe/sync）→ 若 `since_sync >= tau` 或 adaptive 触发则 `tw.sync(obs_phys)`；
  3. 累计 reward、latency（info 中的平均时延）、energy（飞行+计算）、completion（任务级：有任务用户中未超 deadline 比例，参考 `env_fix_audit.md` §14 口径）；
  4. **决策一致率 `agree`**：每时隙用物理真值重算完美决策：`uav_pos_true = obs_phys["uav_pos"]`，构造物理 obs 副本调 `exact_offload_multi(obs_phys_copy, cfg, uav_pos_true)`（`multi_uav.exact_offload_multi`，单 UAV 时 target 含义一致），与 `action["targets"]` 比较一致比例；
  5. 返回 dict：`reward, latency, energy, completion, agree, syncs, perfect_reward`（perfect_reward = 完美信息下同策略的 reward，作为上界参考；实现方式：同 episode 每时隙用物理 obs 直接 act 另跑一条影子轨迹，或用 agree 换算，二选一，选更简单者并在 README 说明口径）。

- [x] **步骤 4：运行测试确认通过**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_runner.py -v`
预期：PASS（1 passed）

- [x] **步骤 5：冒烟核对**：运行 `python -m digital_twin_uav_leo.runner`（加 `if __name__ == "__main__":` 打印单 episode 输出），确认 `tau=1`（每时隙同步）的 reward 与无 DT 基准接近（hard 基准约 -81.65，允许 ±5% 波动，因 seed/协议差异）。若差异大，检查 UAV 位置真值是否被错误替换。

---

## Task 5：run_experiments.py —— RQ1 同步周期扫描（单 UAV）

**文件：**
- 创建：`digital_twin_uav_leo/run_experiments.py`
- 测试：追加 `tests/test_dt_runner.py`（仅 CLI 参数解析冒烟，可选）

- [x] **步骤 1：实现 RQ1 主循环**

要点：
- CLI：`--rq rq1 --scenario hard --methods postmove_exact,current_exact,mpc_traj_h3 --taus 1,2,3,5,10 --episodes 40 --seed0 73`。
- 场景：hard（12 用户）/ stress（16 用户）用 `scenario_spec.MY_SCENARIOS` 覆盖；每方法×τ 跑 `episodes` 集（seed = seed0 + episode_index，与现有 40 集协议一致），聚合均值到 CSV 行：`scenario, method, tau, reward, latency, energy, completion, agree, syncs`。
- 策略名 → 对象映射：`{"postmove_exact": PostmoveExactPolicy(), "current_exact": CurrentExactPolicy(), "mpc_traj_h3": MPCTrajPolicy(horizon=3, offload="exact")}`；名称以 `uav_leo_experiment` 实际类名为准（实现时 grep 确认）。
- 输出：`digital_twin_uav_leo/results/rq1_{scenario}.csv`。
- 每跑完一行打印，便于中途查看。

- [x] **步骤 2：冒烟跑通（1 集）**

运行：`/root/miniconda3/envs/asr_env/bin/python -m digital_twin_uav_leo.run_experiments --rq rq1 --scenario hard --methods postmove_exact --taus 1,3 --episodes 1`
预期：CSV 生成 2 行，无异常。

- [x] **步骤 3：正式跑 RQ1（单 UAV hard + stress）**

运行：`/root/miniconda3/envs/asr_env/bin/python -m digital_twin_uav_leo.run_experiments --rq rq1 --scenario hard --methods postmove_exact,current_exact,mpc_traj_h3 --taus 1,2,3,5,10 --episodes 40`
（stress 同理 `--scenario stress`）
预期：`results/rq1_hard.csv`、`results/rq1_stress.csv` 各 15 行；检查 `tau=1` 行与 Task 4 冒烟一致。

---

## Task 6：RQ2 预测模型/误差 + RQ3 决策器鲁棒性

**文件：** 修改 `digital_twin_uav_leo/run_experiments.py`

- [x] **步骤 1：实现 RQ2 扫描**

- CLI：`--rq rq2 --tau 3 --predictors freeze,linear,linear_noise --eps 0.05,0.1,0.2 --method postmove_exact`（hard，40 集）。
- 输出：`results/rq2_hard.csv`：`predictor, eps, reward, latency, energy, completion, agree, syncs`。
- 注：`freeze/linear` 的 eps 置 0；`linear_noise` 用 eps 注入。

- [x] **步骤 2：实现 RQ3 扫描**

- CLI：`--rq rq3 --methods postmove_exact,current_exact,mpc_traj_h3 --taus 1,3,5,10`（hard，40 集）。
- 输出：`results/rq3_hard.csv`：`method, tau, reward, agree`；重点看 `agree` 下降斜率（谁对误差最敏感）。

- [x] **步骤 3：跑通并核对**

运行：RQ2、RQ3 各 40 集。
预期：RQ2 中 `linear` 优于 `freeze`（外推有价值）；RQ3 中 `agree` 随 τ 下降，`current_exact` 或 `mpc` 与 `postmove_exact` 的鲁棒性差异可见。

---

## Task 7：RQ4 自适应同步 + 多 UAV k3

**文件：** 修改 `digital_twin_uav_leo/run_experiments.py`

- [x] **步骤 1：实现 RQ4（单 UAV）**

- CLI：`--rq rq4 --scenario hard --method postmove_exact --sync_modes fixed,adaptive --taus 1,2,3,5,10 --deltas 5,10,20 --episodes 40`。
- adaptive 用 `probe_m=2, probe_n=3`；统计 `syncs`（平均同步次数）与 reward。
- 输出：`results/rq4_hard.csv`：`sync_mode, tau_or_delta, reward, latency, energy, completion, agree, syncs`。
- 比较口径（规格 §3 RQ4）：adaptive 与"同平均同步次数"的 fixed τ 比（fixed 平均次数由 RQ1 的 `syncs` 列插值或就近匹配）。

- [x] **步骤 2：多 UAV k3（RQ1 + RQ4 各跑一次）**

- 配置：复用 `run_multi_uav.build_config("k3", seed, episodes)`（3 UAV/51 用户/3 热点/2 km）。
- 策略：`PmeoMEcoPolicy()`（PMEO-M-Eco）；`MPCTrajPolicy(horizon=3, offload="exact")`（MPC-M-H3）。
- 注意多 UAV 下 `env.observation()` 的 `hotspot_pos` 是 (M,2)，predictor 与 `TwinnedWorld` 需按形状处理（Task 2 已覆盖 M>1 分支，此处验证）。
- 输出：`results/rq1_k3.csv`（τ 扫描，5 seeds × 10 集，seed ∈ {1,7,42,73,2024}）、`results/rq4_k3.csv`（adaptive vs fixed）。

- [x] **步骤 3：跑通核对**

预期：k3 的 `tau=1` 行 reward 接近 PMEO-M-Eco 基准约 -3590（±5%）；adaptive 的 `syncs` 明显少于 fixed-τ=1 且 reward 损失小。

---

## Task 8：图、汇总表与 README 汇报稿

**文件：**
- 创建：`digital_twin_uav_leo/make_figures.py`、`digital_twin_uav_leo/results/summary.md`、`digital_twin_uav_leo/README.md`

- [x] **步骤 1：实现 `make_figures.py`**

- 图 1：τ vs reward / completion / agree（单 UAV hard，3 条方法线），输出 `results/fig_tau_tradeoff.png`。
- 图 2：predictor/ε vs reward（RQ2），输出 `results/fig_predictor_robustness.png`。
- 图 3：adaptive vs fixed（RQ4，x=平均同步次数，y=reward，或并排柱状），输出 `results/fig_adaptive_sync.png`。
- 复用 `matplotlib`（已装）；风格与 `uav_leo_experiment/make_architecture_multi_uav.py` 一致（DejaVu Sans，220 dpi）。

- [x] **步骤 2：写 `results/summary.md`**

- 每 RQ 一段：问题、设置、核心数字（均值）、一句话结论。
- 表格格式参考 `experiments/uav_leo_v2x_paper_final/paper_results.md`。

- [x] **步骤 3：写 `digital_twin_uav_leo/README.md`（导师汇报入口）**

- 内容：方向一句话、方法与机制示意（引用架构图）、RQ1-4 结果表 + 图、一周结论、与 PMEO 决策顺序故事的衔接、诚实边界（误差注入模拟真实感知误差、任务冻结高估负载）。
- 若 RQ4 无优势，如实报告并给出"工程建议 = 固定 τ 最优值"。

- [x] **步骤 4：全量回归 + 复跑核对**

运行：`/root/miniconda3/envs/asr_env/bin/python -m pytest tests/test_dt_env.py tests/test_dt_runner.py -v`
预期：PASS；再抽查每个 CSV 行数与 README 数字一致。
