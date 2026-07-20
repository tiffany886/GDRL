# GDRL 实验可视化与结果分析教程

## 1. 运行三种场景实验

在 `C:\Users\Tiffany\Desktop\GDRL\GDRL` 目录下运行：

```powershell
python run_scenarios.py --episodes 600 --amn_epochs 100 --force_retrain_amn --execute
```

运行完成后，每个场景会生成一个结果目录：

```text
experiments/
  small_U3_L8_N8_T100/compare_600ep_T100/
  medium_U6_L12_N12_T100/compare_600ep_T100/
  large_U10_L16_N16_T100/compare_600ep_T100/
```

每个 `compare_600ep_T100` 目录里最重要的是：

```text
baseline_summary.csv
baseline_episodes.csv
*.monitor.csv
*_model.zip
figures/
```

其中 `baseline_summary.csv` 是最终汇总表，`baseline_episodes.csv` 是每个 episode 的 reward 明细。

## 2. 生成单个场景的图

例如 large 场景：

```powershell
python visualize_results.py --compare_dir experiments\large_U10_L16_N16_T100\compare_600ep_T100 --output_dir experiments\large_U10_L16_N16_T100\compare_600ep_T100\figures --latency_file experiments\large_U10_L16_N16_T100\compare_600ep_T100\no_step_latency.npy --dpi 180
```

small / medium 只需要替换目录名：

```powershell
python visualize_results.py --compare_dir experiments\small_U3_L8_N8_T100\compare_600ep_T100 --output_dir experiments\small_U3_L8_N8_T100\compare_600ep_T100\figures --latency_file experiments\small_U3_L8_N8_T100\compare_600ep_T100\no_step_latency.npy --dpi 180

python visualize_results.py --compare_dir experiments\medium_U6_L12_N12_T100\compare_600ep_T100 --output_dir experiments\medium_U6_L12_N12_T100\compare_600ep_T100\figures --latency_file experiments\medium_U6_L12_N12_T100\compare_600ep_T100\no_step_latency.npy --dpi 180
```

生成的图包括：

```text
single_gdrl_reward_curve.png
baseline_reward_curves.png
baseline_reward_bar.png
baseline_latency_bar.png
baseline_reward_latency_tradeoff.png
experiment_dashboard.png
```

## 3. 生成三种场景总览图

运行：

```powershell
python visualize_all_scenarios.py
```

输出目录：

```text
experiments/scenario_overview_figures/
```

生成文件：

```text
all_scenarios_reward_mean.png
all_scenarios_latency_mean.png
all_scenarios_latency_p95.png
all_scenarios_reward_curves.png
gdrl_gap_vs_best_mlp.png
all_scenarios_summary.csv
scenario_rank_table.csv
```

建议报告中优先使用：

```text
gdrl_gap_vs_best_mlp.png
all_scenarios_reward_mean.png
all_scenarios_latency_p95.png
all_scenarios_reward_curves.png
```

## 4. 当前结果总结

三种场景的核心结论：

```text
small:  PPO-MLP 平均 reward 最好，TRPO-MLP 时延最好，GDRL 不占优。
medium: GDRL 平均 reward 最高，但只比 TRPO-MLP 略高；PPO-MLP 时延最好。
large:  TRPO-MLP 平均 reward 最高，GDRL 平均时延和 P95 时延最好。
```

GDRL 相比最佳 MLP 的差距：

```text
small:  reward 低约 597，P95 时延高约 3.80 ms。
medium: reward 高约 10，但 P95 时延高约 1.76 ms。
large:  reward 低约 897，但 P95 时延低约 4.91 ms。
```

因此，当前实验不能简单写成“GDRL 全面优于所有基线”。更准确的表述是：

> GDRL 在中等规模场景中取得最高平均 reward，在大规模场景中取得最低平均时延和最低 P95 时延，说明图结构建模对复杂拓扑下的时延稳定性有帮助；但在 small 和 large 场景的 reward 指标上，GDRL 仍落后于部分 MLP 基线。

## 5. 为什么当前 GDRL 效果不够好

### 5.1 小场景图结构优势不明显

small 场景只有 `U=3, L=8, N=8`，拓扑复杂度不高。MLP 已经能从状态中学习出有效策略，GNN 的结构建模优势不明显，反而增加了训练难度和参数路径。

### 5.2 GDRL 使用了更复杂的两级动作映射

GDRL 不是直接输出最终动作，而是：

```text
TRPO latent action -> AMN -> 离散卸载动作 + 连续资源分配动作
```

这个动作映射网络 AMN 如果训练得不够贴近强化学习期间的真实动作分布，就会引入额外误差。MLP 基线虽然结构简单，但训练目标更直接。

### 5.3 AMN 预训练数据和策略分布不完全一致

当前 AMN 主要用随机 latent action 预训练。真实 TRPO/GDRL 训练后产生的 latent action 分布可能会偏离预训练分布，导致 AMN 映射质量下降，最终影响 reward。

### 5.4 Reward 和 latency 的目标存在冲突

large 场景中 GDRL 的 P95 时延最好，但 reward 不是最高。这说明当前 reward 函数中，GDRL 学到的策略更偏向稳定降低时延，而 TRPO-MLP 可能在资源利用或部分高 reward 决策上更激进。

### 5.5 GNN 特征提取器还比较浅

当前 GFEN 只有两层 GCN，且图节点特征较简单。对于复杂 SAGIN 拓扑，图网络可能没有充分表达链路质量、资源状态、请求类型之间的深层关系。

### 5.6 训练轮数和超参数可能更适合 MLP

TRPO-MLP / PPO-MLP 是稳定基线，结构更简单，600 episodes 已经能比较好收敛。GDRL 参数更多、路径更长，可能需要更多 episode、更细的学习率、KL 约束、AMN 联合训练或分阶段训练。

### 5.7 当前使用 Python 近似信道模型

程序提示：

```text
matlab.engine is unavailable; using Python paper-approx channel model
```

这说明当前没有使用 MATLAB 原始 `groundtospace.m` 信道模型。Python 近似模型可以跑通实验，但可能与论文环境存在偏差，导致复现效果和原论文不完全一致。

## 6. 后续改进建议

优先级建议如下：

1. 增加 GDRL 训练轮数，例如 1000 或 1500 episodes，观察 reward 是否继续上升。
2. 不使用完全随机 latent action 训练 AMN，改成“先训练 GDRL 收集动作，再微调 AMN，再继续训练 GDRL”的循环方式。
3. 对 reward 做归一化或重新调权，让时延、资源利用、成功率之间的权重更清晰。
4. 尝试更强的图网络，例如 GraphSAGE、GAT，或给边加入链路质量/距离特征。
5. 分别固定 AMN、只训练策略网络，检查问题主要来自 GFEN 还是 AMN。
6. 每个场景使用多个随机种子重复实验，报告 mean ± std，避免单次运行波动影响结论。
7. 如果条件允许，安装 MATLAB engine，比较 MATLAB 信道模型和 Python 近似模型下的结果差异。

## 7. 报告中推荐写法

可以写成：

> 实验结果表明，GDRL 并非在所有场景和所有指标上都优于 MLP 基线。在 small 场景中，由于拓扑规模较小，图结构建模的优势不明显，PPO-MLP 和 TRPO-MLP 取得了更好的 reward 或 latency 表现。在 medium 场景中，GDRL 取得最高平均 reward，说明随着拓扑复杂度增加，图特征提取开始发挥作用。在 large 场景中，GDRL 的平均 reward 低于 TRPO-MLP，但平均时延和 P95 时延最低，说明 GDRL 对复杂网络中的时延稳定性更有优势。当前 GDRL reward 表现不足，可能与 AMN 动作映射误差、预训练分布和真实策略分布不一致、GNN 表达能力有限以及训练超参数尚未充分优化有关。

