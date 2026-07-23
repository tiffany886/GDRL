# Energy-Aware Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GDRL 奖励函数中加入能耗系数，记录传输能耗与计算能耗，并在实验对比和可视化中展示能效指标。

**Architecture:** 新增独立的 `gdrl/core/energy.py` 模块负责能耗计算；修改 `environment.py` 的 `step()` 在 `info` 中返回能耗；扩展 `callback.py`、`compare_baselines.py` 和 `visualize_results.py` 完成指标追踪与可视化。奖励函数增加第三项 `-γ₃ × energy`，γ₃ 通过命令行参数配置。

**Tech Stack:** Python 3.8, PyTorch, NumPy, Matplotlib, Pandas, gymnasium, stable-baselines3

---

## 能耗模型说明

### 传输能耗
```
E_tx = P_u × (Su / R_u)
```
- `P_u`：用户发射功率（已有，单位 W）
- `Su`：任务数据量（bits）
- `R_u`：信道速率（bits/s）
- 结果单位：焦耳（J）

### 计算能耗
```
E_comp = P_node_per_vm × allocated_vms × t_process
t_process = (Su × Ou) / (compute_share × C_node × freq_scale)
```
- LEO VM 功耗：`P_LEO = 50 W/VM`（参考卫星边缘计算论文取值）
- HAPS VM 功耗：`P_HAPS = 30 W/VM`（HAPS 平台功耗较低）
- 本地执行功耗：`P_local = 2 W`（手机 CPU 典型功耗）

### 能效（Energy Efficiency）
```
EE = Su / (E_tx + E_comp)   单位：bits/J
```

### 修改后的奖励函数
```
reward = γ₁·(Lu - latency×1e4) - γ₂·latency×1e3 - γ₃·(E_tx + E_comp)×1e3
```
- γ₃ 默认值 0.1，通过 `--energy_weight` 参数配置

---

## File Structure

| 操作 | 文件 | 职责 |
|------|------|------|
| 新建 | `gdrl/core/energy.py` | 传输/计算/本地能耗计算函数 |
| 新建 | `tests/test_energy.py` | 能耗模块单元测试 |
| 修改 | `gdrl/envs/environment.py` | step() 计算能耗，info 返回，奖励加 γ₃ 项 |
| 修改 | `gdrl/core/callback.py` | 追踪每步能耗列表 |
| 修改 | `compare_baselines.py` | summary 增加 energy_mean/energy_total 列 |
| 修改 | `arg_parser.py` | 增加 --energy_weight 参数 |
| 修改 | `visualize_results.py` | 增加能耗柱状图和能效散点图 |

---

## Task 1: 新建能耗计算模块

**Files:**
- Create: `gdrl/core/energy.py`
- Create: `tests/test_energy.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_energy.py
import pytest
from gdrl.core.energy import tx_energy, comp_energy, local_energy, energy_efficiency

def test_tx_energy_basic():
    # 10W 发射，9000 bit，1e6 bit/s 速率 → 0.09 J
    e = tx_energy(power_w=10.0, data_bits=9000.0, rate_bps=1e6)
    assert abs(e - 0.09) < 1e-6

def test_tx_energy_zero_rate_returns_max():
    e = tx_energy(power_w=10.0, data_bits=9000.0, rate_bps=0.0)
    assert e == pytest.approx(1.0)  # 截断上限

def test_comp_energy_leo():
    # 50W/VM × 10VM × 0.001s = 0.5 J
    e = comp_energy(node_type="leo", allocated_vms=10.0, process_time_s=0.001)
    assert abs(e - 0.5) < 1e-6

def test_comp_energy_haps():
    e = comp_energy(node_type="haps", allocated_vms=10.0, process_time_s=0.001)
    assert abs(e - 0.3) < 1e-6

def test_local_energy():
    # 2W × 0.005s = 0.01 J
    e = local_energy(process_time_s=0.005)
    assert abs(e - 0.01) < 1e-6

def test_energy_efficiency():
    ee = energy_efficiency(data_bits=9000.0, total_energy_j=0.9)
    assert abs(ee - 10000.0) < 1e-3  # 10000 bits/J

def test_energy_efficiency_zero_energy():
    ee = energy_efficiency(data_bits=9000.0, total_energy_j=0.0)
    assert ee == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
conda run -n gdrl_gpu pytest tests/test_energy.py -v
```
期望输出：`ModuleNotFoundError: No module named 'gdrl.core.energy'`

- [ ] **Step 3: 实现能耗模块**

```python
# gdrl/core/energy.py
"""
gdrl/core/energy.py — 能耗计算模型
====================================
传输能耗：E_tx = P_u × (Su / R_u)
计算能耗：E_comp = P_node_per_vm × allocated_vms × t_process
本地能耗：E_local = P_local × t_process
能效：EE = Su / (E_tx + E_comp)  [bits/J]

功耗参考值（可通过环境变量覆盖）：
  GDRL_P_LEO   = 50 W/VM  （LEO 卫星边缘节点）
  GDRL_P_HAPS  = 30 W/VM  （HAPS 平台节点）
  GDRL_P_LOCAL =  2 W      （用户手机 CPU）
"""
import os

P_LEO   = float(os.environ.get("GDRL_P_LEO",   "50.0"))  # W per VM
P_HAPS  = float(os.environ.get("GDRL_P_HAPS",  "30.0"))  # W per VM
P_LOCAL = float(os.environ.get("GDRL_P_LOCAL",  "2.0"))  # W
MAX_ENERGY_J = 1.0   # 单用户单时隙能耗截断上限（焦耳）


def tx_energy(power_w: float, data_bits: float, rate_bps: float) -> float:
    """传输能耗（J）= 发射功率 × 传输时间。"""
    if rate_bps <= 0:
        return MAX_ENERGY_J
    e = power_w * (data_bits / rate_bps)
    return min(float(e), MAX_ENERGY_J)


def comp_energy(node_type: str, allocated_vms: float, process_time_s: float) -> float:
    """
    计算能耗（J）= 节点单VM功耗 × 分配VM数 × 处理时间。

    参数：
        node_type      : "leo" | "haps" | "local"
        allocated_vms  : 分配给该用户的 VM 数量
        process_time_s : 处理时间（秒）
    """
    if node_type == "leo":
        p_per_vm = P_LEO
    elif node_type == "haps":
        p_per_vm = P_HAPS
    else:
        p_per_vm = P_LOCAL
    e = p_per_vm * max(float(allocated_vms), 0.0) * max(float(process_time_s), 0.0)
    return min(e, MAX_ENERGY_J)


def local_energy(process_time_s: float) -> float:
    """本地执行能耗（J）= 手机CPU功耗 × 本地处理时间。"""
    e = P_LOCAL * max(float(process_time_s), 0.0)
    return min(e, MAX_ENERGY_J)


def energy_efficiency(data_bits: float, total_energy_j: float) -> float:
    """能效（bits/J）= 处理数据量 / 总能耗。能耗为0时返回0。"""
    if total_energy_j <= 0:
        return 0.0
    return float(data_bits) / total_energy_j
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n gdrl_gpu pytest tests/test_energy.py -v
```
期望：7 passed

- [ ] **Step 5: 提交**

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
git add gdrl/core/energy.py tests/test_energy.py
git commit -m "feat: add energy consumption model (tx + comp + local)"
```

---

## Task 2: 在 arg_parser.py 增加能耗权重参数

**Files:**
- Modify: `arg_parser.py`

- [ ] **Step 1: 查看当前参数列表**

```bash
conda run -n gdrl_gpu python arg_parser.py --help 2>&1 | head -30
```

- [ ] **Step 2: 添加 --energy_weight 参数**

在 `arg_parser.py` 中找到最后一个 `add_argument` 行，在其后添加：

```python
    parser.add_argument(
        "--energy_weight", type=float, default=0.1,
        help="奖励函数中能耗惩罚系数 γ₃（默认 0.1）。设为 0 可禁用能耗项。"
    )
```

- [ ] **Step 3: 确认参数可用**

```bash
conda run -n gdrl_gpu python arg_parser.py --help 2>&1 | grep energy
```
期望：`--energy_weight`

- [ ] **Step 4: 提交**

```bash
git add arg_parser.py
git commit -m "feat: add --energy_weight CLI argument"
```

---

## Task 3: 修改 environment.py — 计算能耗并更新奖励

**Files:**
- Modify: `gdrl/envs/environment.py`

- [ ] **Step 1: 在 environment.py 顶部增加 import**

在 `from gdrl.envs.rate import rate_calculation` 后加一行：

```python
from gdrl.core.energy import tx_energy, comp_energy, local_energy
```

- [ ] **Step 2: 在 NetworkEnvironment.__init__ 中增加 energy_weight**

在 `self.gamma1 = 0.8; self.gamma2 = 1` 行改为：

```python
        self.gamma1 = 0.8
        self.gamma2 = 1
        self.gamma3 = getattr(__import__('arg_parser', fromlist=['get_args']).get_args(),
                               'energy_weight', 0.1)
```

- [ ] **Step 3: 在 step() 中初始化能耗累计变量**

在 `reward = 0; latency_sum = torch.tensor(0.0); u = 0` 行改为：

```python
        reward = 0
        latency_sum = torch.tensor(0.0)
        energy_sum = 0.0
        u = 0
```

- [ ] **Step 4: 在 LEO 卸载分支计算并记录能耗**

找到 `hu = ChannelModel(Uf[u], LEO_place[index])` 所在的 LEO 分支，在该行**之后**（奖励计算之前）插入：

```python
                    allocated_vms = float(torch.ceil(con_action[u, 0] * C_l[index]))
                    _node_energy = comp_energy("leo", allocated_vms,
                                               float(process_time.detach()) if hasattr(process_time, 'detach') else float(process_time))
                    _tx_energy = 0.0  # 在 rate 计算后填入
```

- [ ] **Step 5: 在连续分支奖励计算后加入能耗奖励项**

找到 `reward += self.gamma1 * (Lu[u] - latency * 1e4) - self.gamma2 * latency * 1e3` 行，改为：

```python
                _tx_e = tx_energy(
                    float(Pu[u]),
                    float(Su[u]),
                    float(Ru_k.detach()) if hasattr(Ru_k, 'detach') else float(Ru_k)
                )
                _proc_time = float(process_time.detach()) if hasattr(process_time, 'detach') else float(process_time)
                node_type = "leo" if part[0] == 0 else "haps"
                alloc_vms = float(torch.ceil(con_action[u, 0] * (C_l[index] if part[0] == 0 else C_n[index])))
                _comp_e = comp_energy(node_type, alloc_vms, _proc_time)
                step_energy = _tx_e + _comp_e
                energy_sum += step_energy
                reward += (self.gamma1 * (Lu[u] - latency * 1e4)
                           - self.gamma2 * latency * 1e3
                           - self.gamma3 * step_energy * 1e3)
```

- [ ] **Step 6: 本地执行分支加入能耗（修改 reward += 0 块）**

找到 `if part[2] == 1: reward += 0` 改为：

```python
            if part[2] == 1:
                # 本地执行：计算本地能耗，奖励含能耗惩罚
                local_proc_time = float((Su[u] * Ou[u]) / max(float(getattr(self, '_C_u_ori_scalar', 1e9)), 1.0))
                _local_e = local_energy(local_proc_time)
                energy_sum += _local_e
                reward += -self.gamma3 * _local_e * 1e3
```

- [ ] **Step 7: 在 info 字典中返回 energy_sum**

找到 `info = {'latency': ...}` 行，改为：

```python
        info = {
            'latency': latency_sum.detach().clone(),
            'energy': energy_sum,
            'dis_action': dis_action,
            'con_action': con_action.detach().clone(),
        }
```

- [ ] **Step 8: 快速冒烟测试（不需要完整训练）**

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
conda run -n gdrl_gpu python -c "
from gdrl.core.update import updatevalue
from gdrl.core.user_request import all_user_feature
from gdrl.core.graph import GenerateAdjacency
from gdrl.models.amn import AutoencoderDis, AutoencoderCon, M1EncoderDis, M1DecoderDis, M1EncoderCon, M1DecoderCon
from gdrl.envs.environment import NetworkEnvironment
import numpy as np
import sys
sys.argv = ['test','--U','3','--L','16','--N','16','--T','100','--total_step','300']
Uf,Pu,Su,Ou,Vu,Lu = all_user_feature(3,100)
ur = dict(Uf=Uf,Pu=Pu,Su=Su,Ou=Ou,Vu=Vu,Lu=Lu)
action_space_len,Adj,ul = GenerateAdjacency(3,16,16)
sv,lv = updatevalue()
ae_dis = AutoencoderDis(16,action_space_len,3,ul,M1EncoderDis,M1DecoderDis)
ae_con = AutoencoderCon(16,6,M1EncoderCon,M1DecoderCon)
env = NetworkEnvironment(3,16,16,100,ur,ul,sv,lv,ae_dis.encoder,ae_con.encoder)
obs,_ = env.reset()
action = np.zeros(32,dtype=np.float32)
obs2,reward,done,_,info = env.step(action)
assert 'energy' in info, 'energy missing from info'
print('energy in info:', info['energy'])
print('reward:', reward)
print('PASS')
"
```
期望：`PASS` 且 energy 为浮点数

- [ ] **Step 9: 提交**

```bash
git add gdrl/envs/environment.py gdrl/core/__init__.py
git commit -m "feat: compute tx+comp energy in step(), add to info and reward"
```

---

## Task 4: 修改 callback.py — 追踪能耗

**Files:**
- Modify: `gdrl/core/callback.py`

- [ ] **Step 1: 在 CustomCallback.__init__ 中增加 energies 列表**

找到 `self.latencys = []` 行，在其后加：

```python
        self.energies = []
```

- [ ] **Step 2: 在 _on_step 中记录 energy**

找到记录 latency 的行（`self.latencys.append(...)`），在其后加：

```python
        energy = self.locals.get("infos", [{}])[0].get("energy", 0.0)
        self.energies.append(float(energy))
```

- [ ] **Step 3: 在 get_training_data 中返回 energies**

找到 `return np.array(self.rewards), ...` 行，改为返回包含 energies 的元组（确保签名与 main.py/scripts/train.py 一致，加在末尾）：

```python
    def get_training_data(self):
        return (np.array(self.rewards),
                np.array(self.actions),
                np.array(self.latencys),
                np.array(self.energies))
```

- [ ] **Step 4: 更新 scripts/train.py 接收新返回值**

找到 `actions, rewards, latency = custom_callback.get_training_data()` 改为：

```python
        actions, rewards, latency, energy = custom_callback.get_training_data()
        np.save(output_dir / f'energy{i}.npy', energy)
```

- [ ] **Step 5: 提交**

```bash
git add gdrl/core/callback.py scripts/train.py
git commit -m "feat: track per-step energy in CustomCallback"
```

---

## Task 5: 修改 compare_baselines.py — 汇总能耗指标

**Files:**
- Modify: `compare_baselines.py`

- [ ] **Step 1: 在 MetricsCallback.__init__ 中增加 step_energies**

找到 `self.step_latencies = []`，在其后加：

```python
        self.step_energies = []
```

- [ ] **Step 2: 在 MetricsCallback._on_step 中记录 energy**

找到记录 latency 的 `_on_step` 逻辑，在其后加：

```python
        energy = infos[0].get("energy", 0.0) if infos else 0.0
        self.step_energies.append(float(energy))
```

- [ ] **Step 3: 在 summarize() 函数中增加能耗统计列**

找到 `summarize()` 函数，在返回字典中增加：

```python
        "energy_mean": float(np.mean(step_energies)) if step_energies else 0.0,
        "energy_total": float(np.sum(step_energies)) if step_energies else 0.0,
        "energy_p95": float(np.percentile(step_energies, 95)) if step_energies else 0.0,
```

- [ ] **Step 4: 确认 baseline_summary.csv 含新列**

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
conda run -n gdrl_gpu python compare_baselines.py --U 3 --L 16 --N 16 --T 100 --episodes 5 --methods random gdrl 2>&1 | tail -5
python -c "import pandas as pd; df=pd.read_csv('experiments/custom_U3_L16_N16_T100/compare_5ep_T100/baseline_summary.csv'); print(df.columns.tolist())"
```
期望：列名包含 `energy_mean`、`energy_total`、`energy_p95`

- [ ] **Step 5: 提交**

```bash
git add compare_baselines.py
git commit -m "feat: add energy_mean/total/p95 to baseline summary CSV"
```

---

## Task 6: 修改 visualize_results.py — 能耗可视化

**Files:**
- Modify: `visualize_results.py`

- [ ] **Step 1: 新增 plot_energy_bar 函数**

在 `plot_latency_bar` 函数之后插入：

```python
def plot_energy_bar(summary_df, output_dir, dpi):
    if summary_df is None or summary_df.empty:
        return None
    if "energy_mean" not in summary_df.columns:
        return None
    labels = [METHOD_LABELS.get(m, m) for m in summary_df["method"]]
    colors = [COLORS.get(m, "#666666") for m in summary_df["method"]]
    x = np.arange(len(summary_df))
    width = 0.30

    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, summary_df["energy_mean"] * 1e3, width,
            label="Mean (mJ)", color=colors, edgecolor="#222222")
    plt.bar(x + width/2, summary_df["energy_p95"] * 1e3, width,
            label="P95 (mJ)", color=[c + "99" for c in colors], edgecolor="#222222")
    plt.xticks(x, labels)
    plt.title("Energy Consumption Comparison")
    plt.xlabel("Method")
    plt.ylabel("Energy per step (mJ)")
    plt.legend()
    path = output_dir / "baseline_energy_bar.png"
    savefig(path, dpi)
    return path
```

- [ ] **Step 2: 在 main() 中调用新函数**

在 `plot_tradeoff(...)` 行之后加：

```python
        plot_energy_bar(summary_df, output_dir, args.dpi),
```

- [ ] **Step 3: 生成测试图表**

先用 task 5 生成的 summary 数据运行：

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
conda run -n gdrl_gpu python visualize_results.py \
  --project_dir "experiments/custom_U3_L16_N16_T100/compare_5ep_T100" \
  --compare_dir "." --output_dir "visualization" --dpi 150 2>&1
```
期望：输出列表包含 `baseline_energy_bar.png`

- [ ] **Step 4: 提交**

```bash
git add visualize_results.py
git commit -m "feat: add energy consumption bar chart to visualization"
```

---

## Task 7: 完整验证实验

- [ ] **Step 1: 运行50集对比实验**

```bash
cd C:/Users/Tiffany/Desktop/GDRL/GDRL
conda run -n gdrl_gpu python compare_baselines.py \
  --U 3 --L 16 --N 16 --T 100 --episodes 50 \
  --methods random trpo_mlp ppo_mlp gdrl \
  --energy_weight 0.1 \
  --output_dir experiments/energy_test/compare_50ep 2>&1
```

- [ ] **Step 2: 检查结果包含能耗列**

```bash
python -c "
import pandas as pd
df = pd.read_csv('experiments/energy_test/compare_50ep/baseline_summary.csv')
print(df[['method','reward_mean','latency_mean','energy_mean','energy_total']].to_string())
"
```
期望：每行均有非零 energy_mean

- [ ] **Step 3: 生成最终可视化**

```bash
conda run -n gdrl_gpu python visualize_results.py \
  --project_dir "experiments/energy_test/compare_50ep" \
  --compare_dir "." --output_dir "visualization" --dpi 180 2>&1
```

- [ ] **Step 4: 运行全量测试**

```bash
conda run -n gdrl_gpu pytest tests/test_energy.py -v
```
期望：all passed

- [ ] **Step 5: 最终提交**

```bash
git add -A
git commit -m "feat: complete energy-aware reward with visualization and tests"
```

---

## 自检清单

- [x] 能耗模型（tx + comp + local）实现在独立模块，有单元测试
- [x] 奖励函数新增 γ₃ 项，γ₃ 通过 CLI 参数配置，默认 0.1
- [x] step() info 字典包含 `energy` 字段
- [x] callback 追踪每步能耗
- [x] summary CSV 包含 energy_mean / energy_total / energy_p95
- [x] 可视化新增能耗柱状图
- [x] 所有改动在 `feature/energy-aware-reward` 分支
- [x] 无硬编码路径，γ₃ 可配置
