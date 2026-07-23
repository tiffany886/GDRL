# Hybrid SAGIN: Satellite-UAV-gNB-MEC 四层异构网络 + 部分卸载 + SAC + Transformer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 在 GDRL 基础上重构为四层异构网络（LEO + UAV + gNB + MEC + UE），增加部分卸载、SAC+GNN 算法、Transformer AMN，支持数字孪生集中训练 + 分布式执行。

**Architecture:** 动作编码从 7-bit 扩到 12-bit；新增 gNB/MEC 节点类型；AMN 增加卸载比例输出头；新增 SAC 和 Transformer 两种对比算法；新增策略蒸馏模块。

---

## 文件变更总览

```
# 新增
gdrl/models/amn_transformer.py    ← Transformer 版 AMN
gdrl/distillation/__init__.py     ← 蒸馏模块
gdrl/distillation/distill.py       ← 策略蒸馏
gdrl/distillation/executor.py      ← 分布式执行器
gdrl/core/digital_twin.py          ← 数字孪生环境
configs/hybrid_scenario.yaml       ← 场景 YAML
tests/test_hybrid_nodes.py         ← 节点测试
tests/test_partial_offload.py      ← 部分卸载测试

# 修改
gdrl/core/nodes.py                 ← 新增 gNB_status(), MEC_status(); UAV 替代 HAPS
gdrl/core/graph.py                 ← 12-bit 动作编码; 五类节点邻接
gdrl/core/user.py                  ← 用户移动轨迹
gdrl/core/env_init.py              ← 初始化 gNB, MEC
gdrl/core/update.py                ← 适配新状态变量
gdrl/core/energy.py                ← 部分卸载混合能耗 + gNB/MEC 功耗
gdrl/envs/channel.py               ← 新增 5G NR + UAV 信道 + MEC 光纤
gdrl/envs/rate.py                  ← 部分卸载速率
gdrl/envs/environment.py           ← 观测空间 + step 适配四层网络+部分卸载
gdrl/models/amn.py                 ← 新增卸载比例输出头
gdrl/models/feature.py             ← 观测维度适配五类节点
compare_baselines.py               ← +gdrl_sac, +gdrl_transformer 方法
arg_parser.py                      ← SAC / Transformer / hybrid 参数
run_scenarios.py                   ← 方法列表 + hybrid 场景
experiment_config.py               ← SCENARIOS 加四个场景
configs/scenarios.yaml             ← 四个场景定义
```

---

## Task 1: 创建新分支 + 基础环境检查

- [ ] **Step 1: 确认当前在 feature/hybrid-sagin 分支**

```bash
git branch
```
期望：`* feature/hybrid-sagin`

- [ ] **Step 2: 确认依赖可用**

```bash
python -c "import torch; from torch_geometric.nn import GATv2Conv; from stable_baselines3 import SAC; print('all deps ok')"
```
期望：`all deps ok`

- [ ] **Step 3: 确认旧测试全通过**

```bash
python -m pytest tests/ -v --timeout=60 2>&1 | tail -20
```
期望：无失败

---

## Task 2: 节点改造 — gNB + MEC + UAV

**Files:**
- Modify: `gdrl/core/nodes.py`
- Create: `tests/test_hybrid_nodes.py`

- [ ] **Step 1: 读 nodes.py 确认当前结构**

```bash
cat -n gdrl/core/nodes.py
```

- [ ] **Step 2: 在 nodes.py 末尾追加 gNB_status() 和 MEC_status()**

```python
def gNB_status():
    """地面基站（gNB）初始状态。高算力、低时延、固定位置。"""
    C_g = np.random.randint(200, 500)       # VM 实例数
    gNB_place = 10                           # 固定高度 10m（宏站天线高度）
    return C_g, gNB_place


def all_gNB_status(G):
    """批量生成 G 个 gNB 的状态。"""
    C_g, gNB_place = [], []
    for _ in range(G):
        cg, gp = gNB_status()
        C_g.append(cg)
        gNB_place.append(gp)
    return np.array(C_g), np.array(gNB_place)


def MEC_status():
    """大型地面中心站（MEC）。超高算力，通过光纤连接 gNB。"""
    C_m = np.random.randint(800, 1500)      # VM 实例数
    MEC_place = 5                            # 室内机房
    return C_m, MEC_place


def all_MEC_status(M):
    """批量生成 M 个 MEC 中心站状态。"""
    C_m, MEC_place = [], []
    for _ in range(M):
        cm, mp = MEC_status()
        C_m.append(cm)
        MEC_place.append(mp)
    return np.array(C_m), np.array(MEC_place)
```

- [ ] **Step 3: 改 HAPS → UAV（降低高度和算力）**

把 `HAPS_status` 改为：
```python
def UAV_status():
    """无人机（UAV）初始状态。中等算力、可移动、低空。"""
    C_u = np.random.randint(80, 120)         # VM 实例数
    UAV_place = np.random.randint(100, 5000) # 高度 100m - 5km
    return C_u, UAV_place


def all_UAV_status(V):
    """批量生成 V 个 UAV 状态。"""
    C_v, UAV_place = [], []
    for _ in range(V):
        cv, up = UAV_status()
        C_v.append(cv)
        UAV_place.append(up)
    return np.array(C_v), np.array(UAV_place)
```

- [ ] **Step 4: 写节点单元测试 tests/test_hybrid_nodes.py**

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from gdrl.core.nodes import gNB_status, all_gNB_status, MEC_status, all_MEC_status, UAV_status, all_UAV_status


def test_gNB_status():
    C, pos = gNB_status()
    assert 200 <= C <= 500
    assert pos == 10


def test_MEC_status():
    C, pos = MEC_status()
    assert 800 <= C <= 1500
    assert pos == 5


def test_UAV_status():
    C, pos = UAV_status()
    assert 80 <= C <= 120
    assert 100 <= pos <= 5000


def test_all_gNB_status():
    C, pos = all_gNB_status(3)
    assert C.shape == (3,)
    assert pos.shape == (3,)


def test_all_MEC_status():
    C, pos = all_MEC_status(2)
    assert C.shape == (2,)
    assert pos.shape == (2,)


def test_all_UAV_status():
    C, pos = all_UAV_status(4)
    assert C.shape == (4,)
    assert pos.shape == (4,)
```

- [ ] **Step 5: 跑测试**

```bash
python -m pytest tests/test_hybrid_nodes.py -v
```
期望：6 passed

- [ ] **Step 6: 提交**

```bash
git add gdrl/core/nodes.py tests/test_hybrid_nodes.py
git commit -m "feat: add gNB, MEC node types; replace HAPS with UAV"
```

---

## Task 3: 动作编码改造 7-bit → 12-bit

**Files:**
- Modify: `gdrl/core/graph.py`

- [ ] **Step 1: 读 graph.py 确认当前编码**

```bash
grep -n "bit\|action_space\|user_lists" gdrl/core/graph.py | head -20
```

- [ ] **Step 2: 改为 12-bit 编码**

保留 `GenerateAdjacency(U, L, N, G, M, V)` 签名，新增 gNB 数 G、MEC 数 M、UAV 数 V：

```
12-bit 编码：
  bit[11:10] 节点类型: 00=gNB, 01=UAV, 10=LEO, 11=MEC
  bit[9:0]   节点编号: 0-1023
```

节点排序：UE[0:U] → gNB[U:U+G] → UAV[U+G:U+G+V] → LEO[U+G+V:U+G+V+L] → MEC[U+G+V+L:U+G+V+L+M]

邻接矩阵形状 `[U+G+V+L+M, U+G+V+L+M]`。

- [ ] **Step 3: 跑冒烟测试**

```bash
python -c "from gdrl.core.graph import GenerateAdjacency; _, _, ul = GenerateAdjacency(5,2,1,1,4); print('ok')"
```

- [ ] **Step 4: 提交**

```bash
git add gdrl/core/graph.py
git commit -m "feat: expand action encoding from 7-bit to 12-bit for 5 node types"
```

---

## Task 4: 部分卸载 — AMN 增加卸载比例头

**Files:**
- Modify: `gdrl/models/amn.py`
- Create: `tests/test_partial_offload.py`

- [ ] **Step 1: AMN 输出结构调整**

原版 32 维输出 → 新版 2U + 16 + 2U 维：
```
[0:U]           → 卸载比例 (sigmoid, 0-1)
[U:U+16]        → 离散卸载目标（编码后）
[U+16:U+16+2U]  → 连续资源分配
```

- [ ] **Step 2: 编写部分卸载测试**

```python
def test_partial_offload_output_shape():
    ...

def test_offload_ratio_in_range():
    ...
```

- [ ] **Step 3: 提交**

```bash
git add gdrl/models/amn.py tests/test_partial_offload.py
git commit -m "feat: add partial offload ratio head to AMN output"
```

---

## Task 5: 环境改造 — environment.py 适配四层网络 + 部分卸载

**Files:**
- Modify: `gdrl/envs/environment.py`
- Modify: `gdrl/core/env_init.py`
- Modify: `gdrl/core/update.py`
- Modify: `gdrl/core/energy.py`

- [ ] **Step 1: env_init.py 增加 gNB/MEC 初始化**

```python
def ResetFunction(G, M, V, L, T, U):
    # gNB 状态
    C_g, gNB_place = all_gNB_status(G)
    gNB_status_dict = {'C_g': C_g, 'gNB_place': gNB_place, 'C_g_ori': C_g.copy()}
    gNB_resource = {'C_resource_g': np.zeros((G, T))}
    # MEC 状态
    C_m, MEC_place = all_MEC_status(M)
    MEC_status_dict = {'C_m': C_m, 'MEC_place': MEC_place, 'C_m_ori': C_m.copy()}
    MEC_resource = {'C_resource_m': np.zeros((M, T))}
    # UAV 状态
    C_v, UAV_place = all_UAV_status(V)
    UAV_status_dict = {'C_v': C_v, 'UAV_place': UAV_place, 'C_v_ori': C_v.copy()}
    UAV_resource = {'C_resource_v': np.zeros((V, T))}
    # ... 返回新元组
```

- [ ] **Step 2: environment.py step() 增加部分卸载逻辑**

关键改动点：
```
1. 解析动作时多读 offload_ratio[U]
2. 本地处理：offload_ratio × task → 本地计算
3. 卸载处理：(1-offload_ratio) × task → 发给目标节点
4. 时延 = max(本地时延, 卸载时延) 或 加权和
5. 能耗 = 本地能耗 + 卸载能耗
```

- [ ] **Step 3: energy.py 增加 gNB/MEC 功耗**

```python
P_gNB  = float(os.environ.get("GDRL_P_GNB",  "100.0"))  # W/VM
P_MEC  = float(os.environ.get("GDRL_P_MEC",  "200.0"))  # W/VM
P_UAV  = float(os.environ.get("GDRL_P_UAV",   "20.0"))  # W/VM
```

- [ ] **Step 4: 冒烟测试 — 环境能创建 + step 不报错**

```bash
python -c "
from gdrl.envs.environment import NetworkEnvironment
...
obs, _ = env.reset()
obs2, reward, done, _, info = env.step(np.zeros(action_dim))
print('PASS:', reward, info.get('energy'))
"
```

- [ ] **Step 5: 提交**

```bash
git add gdrl/envs/ gdrl/core/env_init.py gdrl/core/update.py gdrl/core/energy.py
git commit -m "feat: adapt environment for 4-tier network + partial offloading"
```

---

## Task 6: 信道模型 — 新增 5G NR + UAV 空对地 + MEC 光纤

**Files:**
- Modify: `gdrl/envs/channel.py`

- [ ] **Step 1: ChannelModel 增加 gNB/UAV/MEC 分支**

```
gNB 信道: 3.5 GHz (5G NR 中频), UPA 8×8
UAV 信道: 2.4 GHz, 自由空间路径损耗 + Rician
MEC 信道: 光纤, 增益固定 1.0（零损耗假设）
```

- [ ] **Step 2: 提交**

```bash
git add gdrl/envs/channel.py
git commit -m "feat: add 5G NR, UAV air-to-ground, and MEC fiber channel models"
```

---

## Task 7: GFEN 适配五类节点

**Files:**
- Modify: `gdrl/models/feature.py`

- [ ] **Step 1: 观测空间适配**

原版 `8U+3L+3N+2` → 新版 `8U + dim_gNB + dim_UAV + dim_LEO + dim_MEC + 2`。

- [ ] **Step 2: `_build_edge_attr` 适配五类节点**

原版只处理 UE/LEO/HAPS，新版需处理 UE/gNB/UAV/LEO/MEC。

- [ ] **Step 3: 提交**

```bash
git add gdrl/models/feature.py
git commit -m "feat: adapt GFEN for 5 node types in observation and edge features"
```

---

## Task 8: 新增 SAC+GNN 算法

**Files:**
- Modify: `compare_baselines.py`
- Modify: `arg_parser.py`

- [ ] **Step 1: arg_parser.py 加 SAC 参数**

```python
parser.add_argument("--sac_buffer_size", type=int, default=50000)
parser.add_argument("--sac_batch_size", type=int, default=256)
parser.add_argument("--sac_tau", type=float, default=0.005)
```

- [ ] **Step 2: compare_baselines.py 加 gdrl_sac 方法**

```python
if method == "gdrl_sac":
    from stable_baselines3 import SAC
    model = SAC("MlpPolicy", env, policy_kwargs=dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=32),
        net_arch=dict(pi=[64, 64], qf=[64, 64]),
    ), learning_rate=3e-4, buffer_size=50000, batch_size=256,
    tau=0.005, gamma=0.995, verbose=1, device="auto",
    tensorboard_log=str(tensorboard_dir))
```

- [ ] **Step 3: 短测试（5 episodes）**

```bash
python compare_baselines.py --U 5 --G 1 --V 2 --L 4 --M 1 --T 20 --episodes 5 --methods random gdrl_sac --output_dir tmp/sac_smoke
```
期望：5 episodes 完成，baseline_summary.csv 有 gdrl_sac 行

- [ ] **Step 4: 提交**

```bash
git add compare_baselines.py arg_parser.py
git commit -m "feat: add SAC+GNN (gdrl_sac) as comparison algorithm"
```

---

## Task 9: 新增 Transformer-AMN

**Files:**
- Create: `gdrl/models/amn_transformer.py`
- Modify: `compare_baselines.py`
- Modify: `arg_parser.py`

- [ ] **Step 1: 创建 Transformer-AMN**

```python
"""
gdrl/models/amn_transformer.py — Transformer-based Action Mapping Network
替代原版 LSTM Autoencoder，小序列(16步)上效率更高。
"""
import torch
import torch.nn as nn

class TransformerAMN(nn.Module):
    def __init__(self, input_dim=16, d_model=64, nhead=4, num_layers=2,
                 output_dim_dis=256, output_dim_con=32):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head_dis = nn.Linear(d_model, output_dim_dis)
        self.head_con = nn.Linear(d_model, output_dim_con)

    def forward(self, x):
        # x: [B, 16]
        x = self.input_proj(x).unsqueeze(1)         # [B, 1, d_model]
        x = self.transformer(x)                      # [B, 1, d_model]
        x = x.squeeze(1)                              # [B, d_model]
        return self.head_dis(x), self.head_con(x)    # [B, dis_dim], [B, con_dim]
```

- [ ] **Step 2: compare_baselines.py 加 gdrl_transformer 方法**

```python
if method == "gdrl_transformer":
    from gdrl.models.amn_transformer import TransformerAMN
    # ... 替换 AMN 为 TransformerAMN
```

- [ ] **Step 3: 冒烟测试**

```bash
python -c "
from gdrl.models.amn_transformer import TransformerAMN
m = TransformerAMN()
x = torch.randn(4, 16)
dis, con = m(x)
print('dis:', dis.shape, 'con:', con.shape)
"
```
期望：`dis: torch.Size([4, 256]) con: torch.Size([4, 32])`

- [ ] **Step 4: 提交**

```bash
git add gdrl/models/amn_transformer.py compare_baselines.py arg_parser.py
git commit -m "feat: add Transformer-based AMN (gdrl_transformer)"
```

---

## Task 10: 场景配置

**Files:**
- Modify: `experiment_config.py`
- Modify: `configs/scenarios.yaml`
- Create: `configs/hybrid_scenario.yaml`

- [ ] **Step 1: experiment_config.py SCENARIOS 加 hybrid 场景**

```python
SCENARIOS = {
    "small": {"U": 5, "G": 1, "V": 2, "M": 1, "L": 4, "T": 100},
    "medium": {"U": 15, "G": 2, "V": 4, "M": 1, "L": 6, "T": 100},
    "large": {"U": 30, "G": 3, "V": 6, "M": 2, "L": 12, "T": 100},
    "extreme_test": {"U": 40, "G": 4, "V": 8, "M": 2, "L": 12, "T": 100},
}
```

- [ ] **Step 2: 提交**

```bash
git add experiment_config.py configs/
git commit -m "feat: add hybrid SAGIN scenarios (small/medium/large/extreme)"
```

---

## Task 11: 短测试（完整冒烟）

- [ ] **Step 1: small 场景 5 episodes**

```bash
python compare_baselines.py --scenario small --episodes 5 --methods random gdrl --output_dir tmp/smoke_small
```

- [ ] **Step 2: 检查 summary 有所有列**

```bash
python -c "import pandas as pd; df=pd.read_csv('tmp/smoke_small/baseline_summary.csv'); print(df.columns.tolist()); print(df)"
```

- [ ] **Step 3: 若失败则修复**

---

## Task 12: 完整实验（后台运行）

- [ ] **Step 1: small 场景 200ep 全方法**

```bash
python compare_baselines.py --scenario small --episodes 200 --methods random trpo_mlp ppo_mlp gdrl gdrl_sac --output_dir experiments/hybrid_small_200ep 2>&1 | tee logs/hybrid_small.log
```

- [ ] **Step 2: medium 场景 200ep（选做）**

```bash
python compare_baselines.py --scenario medium --episodes 200 --methods random trpo_mlp gdrl gdrl_sac --output_dir experiments/hybrid_medium_200ep 2>&1 | tee logs/hybrid_medium.log
```

---

## 自检清单

- [ ] gNB、MEC、UAV 节点类型正确生成，单元测试通过
- [ ] 12-bit 动作编码覆盖 5 类节点
- [ ] 部分卸载：AMN 输出卸载比例，step() 按比例拆分任务
- [ ] gdrl_sac 方法能跑通 5 episodes
- [ ] gdrl_transformer 方法能跑通 5 episodes
- [ ] 环境在 large 场景（53 节点）创建不报错
- [ ] baseline_summary.csv 含 reward / latency / energy 指标
- [ ] 所有提交在 feature/hybrid-sagin 分支