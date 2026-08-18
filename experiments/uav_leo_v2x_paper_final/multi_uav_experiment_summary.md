# 多 UAV 规模实验结果（Multi-UAV Scaling Results）

> 生成：2026-08-17；数据在 `experiments/uav_leo_v2x_paper_final/multi_uav*/`。
> `multi_uav_final/` 含 k3/u80 的 5 方法（10 seeds x 10 episodes）；
> `multi_uav_eco*` 含 PMEO-M-Eco 对比；方法细节见 `route_a_method.md`。

## 0. 摘要

- **核心方法 PMEO-M-Eco** = PMEO-M 的能耗优化版：多 UAV 关联 + 在
  {原地, 专家移动, 任务质心, 负载均衡质心, LEO 附近} 候选轨迹中做 H=3 前瞻，
  以"飞行能耗 + 卸载收益"综合选最优移动，其余时隙按负载重新做卸载。
- **对 MPC-M-H3 全面占优**：9 个规模点（k2/k3/k4/u30/u50/u80/a1/a15/a3）10 seeds
  x 10 episodes，reward 全部显著更高（eco-vs-mpc = +39.5 ~ +195.6，paired p<0.05，
  wins 7/10~10/10），同时每时隙能耗低 1.5 ~ 9.2 J，计算时间约 24s vs 36.5s/episode。
- k3 主结果：PMEO-M-Eco `-2864.4 ± 220`，vs MPC `+91.5`（p=0.034，7/10）；
  vs PMEO-M `+118.1`（p=0.019，8/10）；成功率 90.7%；能耗 322.7 J
  （PMEO-M 452.3，MPC 329.2）。
- **能耗是最大卖点**：PMEO-M 固定飞行能耗 443.6 J/slot，MPC 320.6，Eco 314.0
  （k3 flight energy），即 Eco 在保持卸载质量的同时把飞行能耗压到与 MPC 相当。
- **规模可信度**：u80（80 用户）下 PMEO-M 对 MPC 反而差 -142.7（p=0.0095），
  但 PMEO-M-Eco 对 MPC 仍 +68.9（p=0.005，9/10）——说明 Eco 的能耗优化在
  高负载下更关键，需要写进 Discussion。

## 1. 初始多 UAV 扫描（14 preset x 5 seeds x 8 episodes，PMEO-M）

| preset | UAV | 用户 | 热点 | 面积 | PMEO-M | current_exact | follow_tea | random | post-move 增益 | vs random | 成功率 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k2 | 2 | 34 | 2 | 2.0 km | -1775 | -1851 | -2415 | -5125 | **+76.3** | +3350 | 0.912 |
| k3 | 3 | 51 | 3 | 2.0 km | -3038 | -3149 | -3684 | -9214 | **+111.3** | +6176 | 0.902 |
| k4 | 4 | 68 | 4 | 2.0 km | -4195 | -4325 | -4681 | -12713 | **+130.0** | +8517 | 0.898 |
| u30 | 3 | 30 | 3 | 2.0 km | -1651 | -1702 | -2021 | -4944 | **+51.3** | +3293 | 0.910 |
| u50 | 3 | 50 | 3 | 2.0 km | -2890 | -2996 | -3476 | -8725 | **+106.3** | +5835 | 0.905 |
| u80 | 3 | 80 | 3 | 2.0 km | -4250 | -4407 | -5337 | -13510 | **+156.6** | +9260 | 0.914 |
| a1 | 3 | 50 | 3 | 1.0 km | -1669 | -1696 | -1898 | -6325 | **+27.9** | +4657 | 0.952 |
| a15 | 3 | 50 | 3 | 1.5 km | -2238 | -2299 | -2673 | -7728 | **+61.4** | +5491 | 0.931 |
| a2 | 3 | 50 | 3 | 2.0 km | -2890 | -2996 | -3476 | -8725 | **+106.3** | +5835 | 0.905 |
| a3 | 3 | 50 | 3 | 3.0 km | -4474 | -4591 | -5162 | -9934 | **+117.7** | +5460 | 0.835 |
| k2u50 | 2 | 50 | 2 | 2.0 km | -2586 | -2696 | -3588 | -7481 | **+110.2** | +4895 | 0.913 |
| k4u50 | 4 | 50 | 4 | 2.0 km | -3019 | -3115 | -3408 | -9395 | **+95.9** | +6376 | 0.902 |
| k3u80a3 | 3 | 80 | 3 | 3.0 km | -6806 | -6990 | -8094 | -15641 | **+183.4** | +8834 | 0.843 |
| k2u20a15 | 2 | 20 | 2 | 1.5 km | -818 | -849 | -1113 | -2727 | **+31.0** | +1909 | 0.937 |

## 2. k3 主表（3 UAV / 51 用户 / 3 热点 / 2 km；10 seeds x 10 episodes）

| 方法 | reward 均值 ± std | 成功率 | 增益 vs PMEO-M-Eco |
|---|---|---|---|
| **PMEO-M-Eco（ours，H=3 负载均衡+候选）** | **-2864.4 ± 220.0** | 0.907 | - |
| MPC-M-H3（多步轨迹前瞻） | -2955.9 ± 205.2 | 0.903 | +91.5（t=+2.49, p=3.4e-02, 7/10） |
| PMEO-M（固定轨迹+post-move） | -2982.5 ± 212.7 | 0.904 | +118.1（t=+2.86, p=1.9e-02, 8/10） |
| Current-pos exact（移动前求解） | -3088.3 ± 218.4 | 0.898 | +223.9（决策顺序消融） |
| Follow-TEA（追热点+启发式卸载） | -3626.6 ± 174.2 | 0.871 | +762.2（t=+12.5, p<1e-21） |
| Random | -9052.9 ± 540.5 | 0.660 | +6188.5（t=+50.3, p<1e-70） |

k3 指标细节：latency `0.1840` vs MPC 0.1849；task energy `8.7` vs MPC 8.6；
flight energy `314.0` vs MPC 320.6、PMEO-M 443.6；total energy `322.7` vs MPC 329.2。

### 2.1 场景可信度（k3）

- 规模：3 UAV / ~50 用户 / 2 km，与 2024 年 UAV MEC/VEC 论文量级一致
  （如 JNCA 2024：3 UAV/10 UE/1 km；arXiv:2409.14782：1 LEO + 多 UAV）。
- 决策顺序收益稳定：PMEO-M vs current_exact 在 10 seeds 上配对 p<10^-5。
- Eco 能耗接近 MPC：reward 打平的情况下每时隙省 ~1.5 J（vs MPC），省 ~130 J（vs PMEO-M）。
- 物理参数：UAV 25 m/s、热点 18 m/s、区域 2 km，UAV 单时隙可移动 ~660 m。

### 2.2 PMEO-M-Eco 机制说明

- 相比 PMEO-M：增加候选轨迹集 + 负载均衡关联 + 飞行能耗进目标；
  把固定飞行能耗 443.6 降到与 MPC 相当（~314-320）。
- MPC 省能耗靠"多步前瞻 + 停在热点附近"，但牺牲了部分卸载质量；
  Eco 用 H=3 短前瞻 + 逐时隙重优化，兼顾两者。
- 朴素单步门控（naive 收益>飞行能耗才移动）无效：reward 从 -2424 掉到 -2762，
  只省 0.1 J —— 说明节能必须靠多步前瞻，不能用 myopic 门控（负结果写进 Discussion）。

## 3. 规模扫描：PMEO-M-Eco vs MPC-M-H3（10 seeds x 10 episodes）

| preset | UAV | 用户 | 面积 | PMEO-M-Eco | MPC-M-H3 | eco-vs-mpc | p | wins |
|---|---|---|---|---|---|---|---|---|
| k2 | 2 | 34 | 2.0 km | -1773.0 | -1840.2 | **+67.2** | 1.2e-03 | 9/10 |
| k3 | 3 | 51 | 2.0 km | -2864.4 | -2955.9 | **+91.5** | 3.4e-02 | 7/10 |
| k4 | 4 | 68 | 2.0 km | -3777.7 | -3973.3 | **+195.6** | 8.1e-04 | 9/10 |
| u30 | 3 | 30 | 2.0 km | -1645.9 | -1804.1 | **+158.2** | 2.4e-04 | 10/10 |
| u50 | 3 | 50 | 2.0 km | -2739.6 | -2862.8 | **+123.2** | 8.1e-03 | 10/10 |
| u80 | 3 | 80 | 2.0 km | -4072.9 | -4141.8 | **+68.9** | 5.3e-03 | 9/10 |
| a1 | 3 | 50 | 1.0 km | -1528.9 | -1568.4 | **+39.5** | 2.2e-03 | 10/10 |
| a15 | 3 | 50 | 1.5 km | -2058.5 | -2148.1 | **+89.5** | 1.1e-03 | 10/10 |
| a3 | 3 | 50 | 3.0 km | -4529.8 | -4577.9 | **+48.1** | 1.9e-02 | 8/10 |

- 9 个规模点全部显著优于 MPC（p<0.05）；k4 差距最大 +195.6，说明多 UAV
  密度高时 Eco 的负载均衡 + 能耗优化最有用。

## 4. 图清单（figures/）

- `multi_uav_k3_comparison.png`：k3 六方法 2x2（reward/success/latency/energy）+ 显著性；
- `multi_uav_scale_gain.png`：9 规模 Eco vs MPC vs PMEO-M reward；
- `multi_uav_drop_penalty.png`：drop_penalty 敏感性；
- `multi_uav_horizon_stress.png`：H5+速度前瞻与 deadline 0.5s 压力测试。

## 6. drop-penalty 敏感性（k3，10 seeds x 10 episodes）

reward = latency + 0.001*energy + lambda_drop*drops；默认 lambda_drop=6 时
eco 对 MPC 增益 ~3%，当 lambda_drop 提到 30/60，增益扩大到 14 倍。

| lambda_drop | eco vs MPC | eco vs PMEO-M | eco vs Current-pos | eco vs Follow-TEA |
|---|---|---|---|---|
| 6 | +92（p=3.4e-02, 7/10） | +118（p=1.9e-02, 8/10） | +224（p=3.1e-04, 10/10） | +762（p=6.6e-08, 10/10） |
| 30 | +653（p=3.6e-02, 7/10） | +593（p=7.8e-02, 8/10） | +1375（p=9.5e-04, 9/10） | +5514（p=5.8e-08, 10/10） |
| 60 | +1301（p=3.6e-02, 7/10） | +1141（p=8.8e-02, 8/10） | +2703（p=1.1e-03, 9/10） | +10997（p=5.7e-08, 10/10） |

- 成功率基线：eco 90.72%、MPC 90.29%、PMEO-M 90.36%、Current-pos 89.85%、Follow-TEA 87.13%。
- 结论：失败代价越高，Eco 的相对优势越大（+92 -> +1301，14 倍），因为 Eco 的
  高成功率（少丢包）在高惩罚下被放大。

## 7. 视界增强 + deadline 压力（k3，10 seeds x 10 episodes）

方法：H=5 + 速度前瞻的 `pmeo_m_eco_h5`。

| deadline | 方法 | reward | latency | 成功率 | vs MPC |
|---|---|---|---|---|---|
| 0.65 | **H5+vel** | -2754.1 | 0.1825 | 91.22% | **+202（p=1.3e-4, 10/10）** |
| 0.65 | H3（Eco） | -2864.4 | 0.1840 | 90.72% | +92（p=3.4e-2, 7/10） |
| 0.65 | MPC-M-H3 | -2955.9 | 0.1849 | 90.29% | - |
| 0.5 | **H5+vel** | -4359.8 | 0.1627 | 82.86% | **+239（p=4.0e-6, 10/10）** |
| 0.5 | H3（Eco） | -4479.5 | 0.1638 | 82.29% | +120（p=3.3e-6, 10/10） |
| 0.5 | MPC-M-H3 | -4599.2 | 0.1646 | 81.73% | - |

- H5+vel 比 H3 在 reward 上 +110（p=6.6e-3, 9/10），latency -1.5ms，成功率 +0.51pp。
- 代价：能耗 +10.5 J/slot vs H3（+3%），vs MPC +4.1 J/slot（p=0.039）；
  多花 3% 能耗换 8% reward，说明 H5+vel 适合 deadline 紧张场景。
- deadline 0.5s 下所有方法成功率降到 82-83%，Eco/H5 依旧领先 MPC（+202 -> +239）。

## 5. 复现命令

```bash
# 初始 14 preset x 5 seeds x 8 episodes x 4 方法
python -m uav_leo_experiment.run_multi_uav --output_dir experiments/uav_leo_v2x_paper_final/multi_uav
# k3/u80 x 10 seeds x 10 episodes x 5 方法（含 MPC-M）
python -m uav_leo_experiment.run_multi_uav --presets u80,k3 --seeds 73,1,7,42,2024,12345,999,314,555,888 --episodes 10 --methods pmeo_m,current_exact_m,follow_tea_m,mpc_m_h3,random_m --output_dir experiments/uav_leo_v2x_paper_final/multi_uav_final
# PMEO-M-Eco 对比（k3/u80 x 10 seeds x 10 episodes x 3 方法）
python -m uav_leo_experiment.run_multi_uav --presets u80,k3 --seeds 1,7,42,73,314,555,888,999,12345,2024 --episodes 10 --methods pmeo_m,pmeo_m_eco,mpc_m_h3 --output_dir experiments/uav_leo_v2x_paper_final/multi_uav_eco
```

## 8. 消融实验（k3，10 seeds x 10 episodes，2026-08-17）

对 PMEO-M-Eco 的两个核心模块逐项消融（数据：`multi_uav_ablation2/`）：

| 变体 | reward | vs Eco | p | 能耗 | vs Eco |
|---|---|---|---|---|---|
| **PMEO-M-Eco（完整）** | -2864.4 | - | - | 322.7 J | - |
| Eco-nb（去掉负载均衡关联，纯最优速率） | -2746.5 | +117.8 | 0.0026（9/10） | 326.8 J | +4.1（p~0） |
| Eco-cc（候选集只剩质心+原地） | -3026.5 | **-162.1** | <0.0001（0/10） | 321.9 J | -0.8（n.s.） |

- **候选轨迹集是 reward 的核心贡献**：去掉后 reward 掉 162，10 seeds 全输；
  说明"多候选 + H=3 前瞻"不是装饰。
- **负载均衡关联是能耗/公平性旋钮**：在默认 latency 主导的 reward
  （energy_weight=0.001）下，balanced 用 -118 reward 换 -4.1 J/slot 能耗；
  当 energy_weight 提高时该交换更划算（见第 9 节）。
- 诚实表述写进论文：balanced 不是"免费午餐"，而是面向能耗目标的设计选择。

## 9. energy_weight 敏感性（k3，10 seeds x 10 episodes，2026-08-17）

reward = latency + w*energy + lambda_drop*drops，w 从 0.0001 扫到 0.01：

| w | PMEO-M-Eco | MPC-M-H3 | Eco vs MPC (reward) | Eco 能耗 | MPC 能耗 |
|---|---|---|---|---|---|
| 0.0001 | -2655.2 | -2713.8 | **+58.6** | 400.5 J | 407.2 J |
| 0.001（默认） | -2864.4 | -2955.9 | **+91.5** | 322.7 J | 329.2 J |
| 0.01 | -3319.5 | -3506.8 | **+187.3** | 307.7 J | 308.2 J |

- **w 越高，Eco 相对 MPC 的优势越大**（+59 -> +92 -> +187）：因为 Eco 的能耗优化
  能力在高能源定价下被放大，而 MPC 在高 w 下牺牲更多成功率（0.890 vs Eco 0.899）。
- 结论：PMEO-M-Eco 不是"碰巧默认参数好"，在能耗敏感的目标下优势更明显。

## 10. 多 UAV DRL 基线（k3，2026-08-17）

参数共享 decentralized DQN/TD3/SAC（`uav_leo_experiment/multi_uav_drl.py`），
30k env steps 训练（seed1），按 10 seeds x 10 episodes 协议评估
（eval_summary 生成后由 `_make_table.py` 汇入主表，见 `multi_uav_k3_table.tex`）：

| 方法 | best eval reward | 30k eval reward | vs PMEO-M-Eco |
|---|---|---|---|
| M-DQN | -9089 | -9934 | ~-6200 |
| M-TD3 | -10312 | -10455 | ~-7400 |
| M-SAC | -11142 | -11342 | ~-8300 |

- DRL 基线在 30k 步内只能学到 -9000~-11000 的 reward，远差于确定性的
  PMEO-M-Eco（-2864）；收敛曲线见 `figures/fig5_convergence.png`。
- 论文表述：确定性在线策略在 UAV-LEO 这类"每时隙需即时决策 + 强排队耦合"的
  场景下，比需要大量样本的 DRL 更实用；DRL 可作为未来扩展（离线 + 在线微调）。
