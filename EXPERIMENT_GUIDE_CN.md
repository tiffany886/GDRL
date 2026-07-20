# GDRL 小/中/大场景实验说明

## 1. 场景设置

当前代码新增了三个预设场景：

| 场景 | 用户 U | LEO 卫星 L | HAPS/无人机 N | 每轮步数 T | 用途 |
|---|---:|---:|---:|---:|---|
| small | 3 | 8 | 8 | 100 | 快速验证流程 |
| medium | 6 | 12 | 12 | 100 | 中等复杂度对比 |
| large | 10 | 16 | 16 | 100 | 当前动作编码范围内较复杂的 SAGIN 场景 |

注意：当前离散动作使用 4 bit 表示 LEO/HAPS 编号，所以 L 和 N 暂时不能超过 16。

## 2. 新目录结构

新实验默认保存到：

```text
experiments/
  small_U3_L8_N8_T100/
    amn/
      autoencoder_dis_small_U3_L8_N8_T100.pth
      autoencoder_con_small_U3_L8_N8_T100.pth
      amn_loss_small_U3_L8_N8_T100.csv
    compare_200ep_T100/
      baseline_summary.csv
      baseline_episodes.csv
      run_config.json
      random.monitor.csv
      trpo_mlp.monitor.csv
      ppo_mlp.monitor.csv
      gdrl.monitor.csv
```

每个场景都有独立 AMN/autoencoder 权重，不再共用旧的：

```text
model_save/autoencoder_dis.pth
model_save/autoencoder_con.pth
```

这样可以避免 U/L/N 改变后出现 size mismatch。

## 3. 推荐训练命令

### 只跑 small 场景

```bash
python compare_baselines.py --scenario small --methods random trpo_mlp ppo_mlp gdrl --episodes 200 --amn_epochs 100
```

### 只跑 medium 场景

```bash
python compare_baselines.py --scenario medium --methods random trpo_mlp ppo_mlp gdrl --episodes 200 --amn_epochs 100
```

### 只跑 large 场景

```bash
python compare_baselines.py --scenario large --methods random trpo_mlp ppo_mlp gdrl --episodes 200 --amn_epochs 100
```

### 强制重新训练当前场景 AMN

如果你怀疑 AMN 没训练好，或者修改了场景/动作编码，使用：

```bash
python compare_baselines.py --scenario medium --methods random trpo_mlp ppo_mlp gdrl --episodes 200 --amn_epochs 200 --force_retrain_amn
```

### 打印小/中/大三条命令

```bash
python run_scenarios.py --episodes 200 --amn_epochs 100
```

### 直接依次执行小/中/大实验

```bash
python run_scenarios.py --episodes 200 --amn_epochs 100 --execute
```

## 4. 生成可视化图

以 small 场景为例：

```bash
python visualize_results.py --compare_dir experiments/small_U3_L8_N8_T100/compare_200ep_T100 --output_dir experiments/small_U3_L8_N8_T100/figures_200ep
```

medium 场景：

```bash
python visualize_results.py --compare_dir experiments/medium_U6_L12_N12_T100/compare_200ep_T100 --output_dir experiments/medium_U6_L12_N12_T100/figures_200ep
```

large 场景：

```bash
python visualize_results.py --compare_dir experiments/large_U10_L16_N16_T100/compare_200ep_T100 --output_dir experiments/large_U10_L16_N16_T100/figures_200ep
```

## 5. 如何判断 AMN 是否训练充分

查看对应场景的：

```text
experiments/场景名/amn/amn_loss_场景名.csv
```

重点看：

- `loss_dis`：离散动作 AMN 重构误差
- `loss_con`：连续动作 AMN 重构误差

如果 loss 一直很大或不下降，说明 AMN 没有学好，GDRL 的动作映射会受影响。

## 6. 实验结论建议

不要只用一个小场景判断 GDRL 是否有效。更合理的写法是：

> 在 small 场景中，PPO-MLP 可能因为问题规模较小而表现较强；随着用户数和 SAGIN 图结构复杂度增加，GDRL 的图特征提取和动作映射机制更有机会体现优势。因此本文进一步设置 small、medium、large 三类场景，比较不同算法在不同规模下的 reward、latency 和稳定性。
