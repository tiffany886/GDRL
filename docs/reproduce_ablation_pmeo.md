# GDRL「学习不必要」消融 —— 一页复现说明

> 用途：给老师/审稿人讲清楚「学习组件贡献≈0」这个实验是怎么做的、怎么复现。
> 对应结果表：`experiments/uav_leo_v2x_paper_final/paper_results.md` §3
> （Ablation — is trajectory learning necessary?）与 §3b（decision-order sensitivity）。
> 环境：Python + torch（仓库 `requirements.txt` 对应 conda 环境 `gdrl_gpu`）。

---

## 1. 实验对象：这里的 GDRL 是什么

本项目里的「GDRL」**不是** Cai 等 JSAC 2025 原文的端到端图强化学习，而是我们复现的
**残差轨迹 PPO + 移动后精确卸载**（`uav_leo_experiment/traj_drl.py` 的 `GDRLPolicy`）：

```
move   = clip(专家移动 + PPO 学到的残差 Δ)      ← 只有轨迹是学的
卸载    = 在移动后位置上逐用户精确枚举(argmin)   ← 卸载层没有学
```

即：把「该学的」给神经网络学（轨迹残差），把「能算的」交给精确求解器，
这是一个对学习组件**最有利**的对照配置——如果连这种配置都学不出增益，
就能干净地说明「学习不必要」。

## 2. 消融的两个开关

`run_experiment.py` 提供两个开关，分别关掉 GDRL 的两个学习/求解组件：

| 开关 | 作用 | 得到的策略名 |
|---|---|---|
| `--gdrl_ablate_traj` | 冻结轨迹残差 Δ=0，轨迹退化为纯专家（`predict_tea`） | `gdrl_no_traj` |
| `--gdrl_ablate_offload` | 保留学到的轨迹，但卸载从「精确枚举」换成 TEA 启发式 | `gdrl_no_opt` |

对照策略（用来定位真正的增益来源）：

| 策略 | 说明 | 代码 |
|---|---|---|
| `postmove_exact`（PMEO） | 专家飞 + **移动后**位置精确卸载 | `ExpertExactOffloadPolicy(offload_at="post_move")` |
| `current_exact` | 同一专家飞，但**移动前**位置精确卸载 | `ExpertExactOffloadPolicy(offload_at="current")` |

## 3. 复现步骤

### 3.1 训练 GDRL（残差轨迹 PPO）

```bash
python -m uav_leo_experiment.train_drl --method gdrl --difficulty v2x_hotspot_hard \
    --users 12 --horizon 30 --seed 73 --steps 40000 \
    --output_dir experiments/uav_leo_v2x/drl_models_fixed
```

- 论文用的是 `experiments/uav_leo_v2x_paper_final/models/gdrl_hard.pt`（40k 步，CPU 约 5 分钟）。
- stress 场景同理：`--difficulty v2x_hotspot_stress --users 16`。

### 3.2 评估：学习消融（40 集配对）

```bash
python -m uav_leo_experiment.run_experiment --difficulty v2x_hotspot_hard \
    --ablation none --episodes 40 --horizon 30 --seed 73 --users 12 --leos 4 \
    --gdrl_model experiments/uav_leo_v2x_paper_final/models/gdrl_hard.pt \
    --gdrl_ablate_traj --gdrl_ablate_offload --skip_slow \
    --methods gdrl,gdrl_no_traj,gdrl_no_opt,postmove_exact,current_exact \
    --output_dir experiments/uav_leo_v2x/ablation_repro
```

**预期输出（hard，40 集，与 paper_results.md §3 一致）：**

| 策略 | reward | 与 GDRL 之差 | 含义 |
|---|---|---|---|
| gdrl（完整） | ≈ −81.6 | — | 学了轨迹残差 |
| gdrl_no_traj | ≈ −81.6 | Δ≤0.13（<0.1%） | **去掉轨迹学习，几乎不变** |
| gdrl_no_opt | ≈ −81.6 | Δ=0.00 | **去掉精确卸载（TEA@post-move），完全相同** |
| postmove_exact（PMEO） | ≈ −81.7 | ≈0 | 免训练方法同分 |
| current_exact | ≈ −84.1 | +2.5（37/40，p<0.0001） | 真正的增益在「移动后求解」 |

### 3.3 评估：机制消融（决策顺序，200 集配对）

```bash
python -m uav_leo_experiment.run_experiment --difficulty v2x_hotspot_hard \
    --ablation none --episodes 200 --horizon 30 --seed 73 --users 12 --leos 4 \
    --skip_slow --methods postmove_exact,current_exact \
    --output_dir experiments/uav_leo_v2x/order_repro
```

**预期输出（hard，200 集）：** post-move 比 current-pos 好 `+2.64`，胜场 `189/200`，
配对 t 检验 `p<1e-23`（stress：`+2.93`，`181/200`）。

## 4. 怎么给老师解释这组实验

1. **不是「我们没调好 DRL」**：GDRL 是残差结构，起点就是专家轨迹（Δ=0 时行为=PMEO），
   是对学习最友好的配置；40k 步训完，学到的 Δ 对 reward 贡献 <0.1%。
2. **不是「精确卸载没用」**：把精确卸载换成 TEA 启发式（仍在移动后位置）结果完全一样，
   说明卸载层的关键也不是「精确」，而是「在哪个位置求值」。
3. **增益到底在哪**：只在「移动前 → 移动后」这一个改动上，200 集配对检验稳定显著。
4. **结论一句话**：这个问题的杠杆是决策顺序，不是学习、也不是精确求解器；
   免训练不是方法假设，而是实验结论。

## 5. 常见坑

- `run_experiment.py` 的 `--users`/`--horizon` 默认值不是 hard 场景，必须显式传
  `--users 12 --horizon 30`（stress 用 `--users 16`），否则和论文数字对不上。
- 消融策略名是 `gdrl_no_traj` / `gdrl_no_opt`（不是 `gdrl_ablate_*`），`--methods` 过滤时别写错。
- 训练/评估都建议加 `--seed 73`，与论文保持一致。
