# GAT + Dynamic Edge Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace GCNConv with GATv2Conv and add dynamic edge features (node load ratio + inter-node distance) in the GFEN, enabling UE to dynamically attend to low-latency, low-congestion LEO/HAPS nodes.

**Architecture:** Surgical change confined to `gdrl/models/feature.py` (GNN layers) and `gdrl/core/graph.py` (edge attr generation). The observation space and action space are unchanged. An ablation flag `--use_gat` allows GCN vs GAT comparison. Edge features: static (normalized inter-node distance as propagation delay proxy) concatenated with dynamic (per-edge destination-node load ratio extracted from raw obs).

**Tech Stack:** PyTorch Geometric GATv2Conv, stable-baselines3 BaseFeaturesExtractor, PyTorch, pytest

---

## File Map

| File | Change |
|------|--------|
| `gdrl/models/feature.py` | Replace GCNConv to GATv2Conv; add `_build_edge_attr` for dynamic edge features; load static `edge_attr.pt` |
| `gdrl/core/graph.py` | Add `compute_edge_attr(edge_index_np, U_place, LEO_place, HAPS_place, U, L, N)` returning [E, 1] normalized-distance tensor |
| `scripts/main.py` | Call `compute_edge_attr` and save `edge_attr.pt` alongside `edge_index.pt` |
| `arg_parser.py` | Add `--use_gat` boolean flag (default True) |
| `tests/test_gat_feature.py` | New test file: output shape, edge_attr shape, ablation flag |

---

### Task 1: Add --use_gat flag to arg_parser.py

**Files:**
- Modify: `arg_parser.py` (after `--energy_weight` block, around line 49)
- Test: `tests/test_gat_feature.py` (new file, flag tests only)

- [ ] **Step 1: Read arg_parser.py to confirm insertion point**

Run: `grep -n "energy_weight" arg_parser.py`

- [ ] **Step 2: Add the flag after --energy_weight**

In `arg_parser.py`, after the `--energy_weight` argument block, add:

```python
    parser.add_argument(
        "--use_gat", action="store_true", default=True,
        help="使用 GATv2Conv 替代 GCNConv（默认开启）。传 --no-use_gat 可还原 GCN 做消融对比。"
    )
    parser.add_argument(
        "--no-use_gat", dest="use_gat", action="store_false"
    )
```

- [ ] **Step 3: Create tests/test_gat_feature.py with flag tests**

```python
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_use_gat_flag_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "4", "--N", "4"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is True


def test_use_gat_flag_disabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "4", "--N", "4", "--no-use_gat"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is False
```

- [ ] **Step 4: Run tests**

```
conda run -n gdrl_gpu python -m pytest tests/test_gat_feature.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add arg_parser.py tests/test_gat_feature.py
git commit -m "feat: add --use_gat ablation flag to arg_parser"
```

---

### Task 2: Add compute_edge_attr to graph.py

Compute normalized inter-node distance for each directed edge. Distance proxy: |pos_src - pos_dst| / max_dist. Node positions: UE uses U_place, LEO uses LEO_place, HAPS uses HAPS_place. Node index order: UE 0..U-1, LEO U..U+L-1, HAPS U+L..U+L+N-1.

**Files:**
- Modify: `gdrl/core/graph.py` (add function after GenerateAdjacency)
- Test: `tests/test_gat_feature.py` (add edge_attr shape test)

- [ ] **Step 1: Read graph.py to find insertion point**

Run: `tail -5 gdrl/core/graph.py`

- [ ] **Step 2: Add compute_edge_attr to graph.py**

After the closing of `GenerateAdjacency`, append:

```python
def compute_edge_attr(edge_index_np, U_place, LEO_place, HAPS_place, U, L, N):
    """
    为每条有向边计算归一化节点间距离（传播延迟代理），shape [E, 1]。

    参数：
        edge_index_np : np.ndarray [2, E]，有向边 (src, dst) 列表
        U_place       : array-like [U]，用户 1D 坐标
        LEO_place     : array-like [L]，LEO 卫星 1D 坐标
        HAPS_place    : array-like [N]，HAPS 平台 1D 坐标
        U, L, N       : 各层节点数

    返回：
        torch.FloatTensor [E, 1]，归一化距离，范围 [0, 1]
    """
    import numpy as np
    import torch

    all_pos = np.concatenate([
        np.asarray(U_place, dtype=float).reshape(-1),
        np.asarray(LEO_place, dtype=float).reshape(-1),
        np.asarray(HAPS_place, dtype=float).reshape(-1),
    ])  # [U+L+N]

    src = edge_index_np[0]
    dst = edge_index_np[1]
    dist = np.abs(all_pos[src] - all_pos[dst]).astype(np.float32)  # [E]

    max_dist = float(dist.max()) if dist.max() > 0 else 1.0
    dist_norm = (dist / max_dist).reshape(-1, 1)  # [E, 1]

    return torch.tensor(dist_norm, dtype=torch.float32)
```

- [ ] **Step 3: Add shape test to test_gat_feature.py**

Append to `tests/test_gat_feature.py`:

```python
import numpy as np
import torch


def test_compute_edge_attr_shape():
    from gdrl.core.graph import GenerateAdjacency, compute_edge_attr
    U, L, N = 3, 4, 4
    _, Adj, _ = GenerateAdjacency(U, L, N)
    rows, cols = np.where(Adj > 0)
    edge_index_np = np.stack([rows, cols], axis=0)
    E = edge_index_np.shape[1]

    rng = np.random.default_rng(0)
    U_place = rng.uniform(0, 1, U)
    LEO_place = rng.uniform(0, 1, L)
    HAPS_place = rng.uniform(0, 1, N)

    ea = compute_edge_attr(edge_index_np, U_place, LEO_place, HAPS_place, U, L, N)
    assert ea.shape == (E, 1), f"expected ({E}, 1), got {ea.shape}"
    assert float(ea.min()) >= 0.0
    assert float(ea.max()) <= 1.0 + 1e-6
```

- [ ] **Step 4: Run tests**

```
conda run -n gdrl_gpu python -m pytest tests/test_gat_feature.py::test_compute_edge_attr_shape -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add gdrl/core/graph.py tests/test_gat_feature.py
git commit -m "feat: add compute_edge_attr() for static distance-based edge features"
```

---

### Task 3: Save edge_attr.pt in main.py

After edge_index.pt is saved in main.py, also compute and save edge_attr.pt using the node positions that are already in scope.

**Files:**
- Modify: `scripts/main.py`

- [ ] **Step 1: Find edge_index.pt save in main.py**

Run: `grep -n "edge_index\|edge_attr\|GenerateAdjacency\|Adj_Matrix\|torch.save" scripts/main.py`

- [ ] **Step 2: Insert edge_attr.pt save after edge_index.pt save**

Find the block that builds `edge_index` from `Adj_Matrix` and calls `torch.save`. Immediately after the `torch.save(edge_index, 'edge_index.pt')` line, add:

```python
# 保存静态边特征（归一化节点间距离）
from gdrl.core.graph import compute_edge_attr as _cef
_ei_np = np.stack(np.where(Adj_Matrix > 0), axis=0)  # [2, E]
_edge_attr = _cef(
    _ei_np,
    user_status['U_place'], LEO_status['LEO_place'], HAPS_status['HAPS_place'],
    args.U, args.L, args.N,
)
torch.save(_edge_attr, 'edge_attr.pt')
```

NOTE: Adjust variable names (`user_status`, `LEO_status`, `HAPS_status`) to match the actual variable names in main.py that hold `U_place`, `LEO_place`, `HAPS_place`.

- [ ] **Step 3: Verify no import errors**

```
conda run -n gdrl_gpu python -c "import scripts.main" 2>&1 | head -5
```

or: `conda run -n gdrl_gpu python scripts/main.py --help`

Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add scripts/main.py
git commit -m "feat: save edge_attr.pt (static distance features) alongside edge_index.pt"
```

---

### Task 4: Replace GCNConv with GATv2Conv in feature.py

This is the core change. The existing GCNConv layers are replaced with GATv2Conv. The `_build_edge_attr` helper extracts dynamic load-ratio features from the (log-scaled) observation tensor and concatenates them with static distance features loaded from edge_attr.pt.

**Files:**
- Modify: `gdrl/models/feature.py`
- Test: `tests/test_gat_feature.py` (add forward-pass shape test)

Observation vector layout (for reference in _build_edge_attr):
```
[Uf(U), Pu(U), Su(U), Ou(U), Vu(U), Lu(U),   <- indices 0 .. 6U-1
 C_l(L), C_n(N), A_u(1), A_u_ori(1),           <- 6U .. 6U+L+N+1
 C_u_ori(U), C_l_ori(L), C_n_ori(N),           <- 6U+L+N+2 .. 7U+2L+2N+1
 U_place(U), LEO_place(L), HAPS_place(N)]       <- 7U+2L+2N+2 .. 8U+3L+3N+1
```
con_state = last 2*(U+L+N) elements = [C_u_ori..HAPS_place], reshaped to [B, U+L+N, 2]
  node feat dim-0 = original capacity per node
  node feat dim-1 = position per node
var_state = first obs_dim - 2*(U+L+N) elements
  C_l_curr at var_state[:, 6U : 6U+L]
  C_n_curr at var_state[:, 6U+L : 6U+L+N]

- [ ] **Step 1: Verify GATv2Conv availability**

```
conda run -n gdrl_gpu python -c "from torch_geometric.nn import GATv2Conv; print('ok')"
```

Expected: `ok`

- [ ] **Step 2: Add import for GATv2Conv**

In `gdrl/models/feature.py`, change:

```python
from torch_geometric.nn import GCNConv
```

to:

```python
from torch_geometric.nn import GCNConv, GATv2Conv
```

- [ ] **Step 3: Add self._L and self._N storage in __init__**

In `__init__`, after `self.output_dim = args.U + args.L + args.N`, add:

```python
        self._L = args.L
        self._N = args.N
```

- [ ] **Step 4: Replace GNN init block**

Replace this block in `__init__`:

```python
        # ── GCN 图神经网络部分（处理拓扑结构）
        self.node_norm = nn.LayerNorm(2)
        self.gnn1 = GCNConv(2, 16)       # 第一层 GCN：节点特征 2→16
        self.gnn2 = GCNConv(16, 1)       # 第二层 GCN：节点特征 16→1（聚合）
        self.graph_norm = nn.LayerNorm(self.output_dim)
```

with:

```python
        # ── GNN 部分：GATv2Conv（默认）或 GCNConv（--no-use_gat 消融）
        _use_gat = getattr(args, 'use_gat', True)
        _EDGE_DIM = 2          # 静态距离(1) + 动态负载比例(1)
        _GAT_HEADS = 4
        self.node_norm = nn.LayerNorm(2)
        if _use_gat:
            self.gnn1 = GATv2Conv(2, 4, heads=_GAT_HEADS, concat=True,
                                  dropout=0.1, add_self_loops=False,
                                  edge_dim=_EDGE_DIM)
            # gnn1 输出：[B, N, 16]（4 heads × 4 dims）
            self.gat_mid_norm = nn.LayerNorm(_GAT_HEADS * 4)
            self.gnn2 = GATv2Conv(_GAT_HEADS * 4, 1, heads=1, concat=False,
                                  dropout=0.0, add_self_loops=False,
                                  edge_dim=_EDGE_DIM)
        else:
            self.gnn1 = GCNConv(2, 16)
            self.gnn2 = GCNConv(16, 1)
            self.gat_mid_norm = nn.LayerNorm(16)
        self._use_gat = _use_gat
        self._edge_dim = _EDGE_DIM
        self.graph_norm = nn.LayerNorm(self.output_dim)

        # 静态边特征（传播距离），由 main.py 运行时保存
        import os as _os
        _ea_path = 'edge_attr.pt'
        if _os.path.exists(_ea_path):
            self.register_buffer('edge_attr', torch.load(_ea_path).float())
        else:
            self.edge_attr = None
```

- [ ] **Step 5: Add _build_edge_attr helper method before forward()**

Add this method inside the class, before `def forward(`:

```python
    def _build_edge_attr(self, scaled_obs, edge_index, batch_size):
        """
        构建 [B, E, 2] 边特征：
          dim-0: 静态归一化距离（edge_attr.pt）
          dim-1: 动态目标节点负载比例（从 log-scale 观测提取）

        如果 edge_attr.pt 未加载，返回 None。
        """
        if self.edge_attr is None:
            return None

        E = self.edge_attr.shape[0]
        device = scaled_obs.device
        U = self.output_dim - self._L - self._N
        L = self._L
        N = self._N

        # 静态边特征 [E, 1] -> [B, E, 1]
        static_ea = self.edge_attr.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        # 动态特征：目标节点当前负载（log-scale 下的代理值）
        var_state = scaled_obs[:, :-2 * self.output_dim]   # [B, var_dim]
        cl_curr = var_state[:, 6 * U: 6 * U + L]          # [B, L]
        cn_curr = var_state[:, 6 * U + L: 6 * U + L + N]  # [B, N]

        con_state_raw = scaled_obs[:, -2 * self.output_dim:].reshape(
            batch_size, self.output_dim, 2)
        cl_ori = con_state_raw[:, U: U + L, 0]            # [B, L]
        cn_ori = con_state_raw[:, U + L: U + L + N, 0]    # [B, N]

        leo_load = cl_curr / (cl_ori.abs() + 1e-6)        # [B, L]
        haps_load = cn_curr / (cn_ori.abs() + 1e-6)       # [B, N]
        ue_load = torch.zeros(batch_size, U, device=device)
        node_load = torch.cat([ue_load, leo_load, haps_load], dim=1)  # [B, U+L+N]

        dst = edge_index[1]                                 # [E]
        dynamic_ea = node_load[:, dst].unsqueeze(2)        # [B, E, 1]

        return torch.cat([static_ea, dynamic_ea], dim=2)   # [B, E, 2]
```

- [ ] **Step 6: Replace GNN forward block**

Replace this block in `forward()`:

```python
        # 3. GCN 图特征提取路径
        con_state = self.node_norm(con_state)
        con_output = self.gnn1(con_state, edge_index)                    # 2→16
        con_output = F.relu(self._safe_activations(con_output))
        con_output = F.dropout(con_output, training=self.training)
        con_output = self.gnn2(con_output, edge_index).squeeze(2)        # 16→1
        con_output = self.graph_norm(self._safe_activations(con_output))
```

with:

```python
        # 3. GNN 图特征提取路径（GAT 或 GCN，由 _use_gat 控制）
        con_state = self.node_norm(con_state)
        edge_attr = self._build_edge_attr(x, edge_index, batch_size)
        if self._use_gat:
            con_output = self.gnn1(con_state, edge_index, edge_attr)     # 2->16
            con_output = F.elu(self._safe_activations(con_output))
            con_output = self.gat_mid_norm(con_output)
            con_output = self.gnn2(con_output, edge_index, edge_attr).squeeze(2)  # 16->1
        else:
            con_output = self.gnn1(con_state, edge_index)                # 2->16
            con_output = F.relu(self._safe_activations(con_output))
            con_output = F.dropout(con_output, training=self.training)
            con_output = self.gnn2(con_output, edge_index).squeeze(2)    # 16->1
        con_output = self.graph_norm(self._safe_activations(con_output))
```

- [ ] **Step 7: Add forward-pass shape tests to test_gat_feature.py**

Append to `tests/test_gat_feature.py`:

```python
import torch
from gymnasium import spaces


def _make_extractor(tmp_path, monkeypatch, U=3, L=4, N=4, use_gat=True):
    import os
    os.chdir(tmp_path)

    from gdrl.core.graph import GenerateAdjacency, compute_edge_attr
    _, Adj, _ = GenerateAdjacency(U, L, N)
    rows, cols = np.where(Adj > 0)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    edge_index_np = np.stack([rows, cols], axis=0)

    rng = np.random.default_rng(42)
    ea = compute_edge_attr(
        edge_index_np,
        rng.uniform(0, 1, U), rng.uniform(0, 1, L), rng.uniform(0, 1, N),
        U, L, N,
    )
    torch.save(edge_index, 'edge_index.pt')
    torch.save(ea, 'edge_attr.pt')

    flag = "--use_gat" if use_gat else "--no-use_gat"
    monkeypatch.setattr("sys.argv", [
        "prog", f"--U={U}", f"--L={L}", f"--N={N}", flag
    ])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    from gdrl.models import feature as feat_mod
    reload(feat_mod)

    obs_dim = 8 * U + 3 * L + 3 * N + 2
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    return feat_mod.CustomFeaturesExtractor(obs_space, features_dim=32), obs_dim


def test_gat_output_shape(tmp_path, monkeypatch):
    extractor, obs_dim = _make_extractor(tmp_path, monkeypatch, use_gat=True)
    obs = torch.randn(2, obs_dim)
    out = extractor(obs)
    assert out.shape == (2, 32)


def test_gcn_output_shape(tmp_path, monkeypatch):
    extractor, obs_dim = _make_extractor(tmp_path, monkeypatch, use_gat=False)
    obs = torch.randn(2, obs_dim)
    out = extractor(obs)
    assert out.shape == (2, 32)
```

- [ ] **Step 8: Run all tests**

```
conda run -n gdrl_gpu python -m pytest tests/test_gat_feature.py -v
```

Expected: all 6 tests pass

- [ ] **Step 9: Commit**

```bash
git add gdrl/models/feature.py tests/test_gat_feature.py
git commit -m "feat: replace GCNConv with GATv2Conv + dynamic edge features in GFEN"
```

---

### Task 5: Integration smoke test

Verify a short training run completes without errors.

**Files:** no changes, only verification

- [ ] **Step 1: Delete stale edge_index.pt and edge_attr.pt if present**

```
rm -f edge_index.pt edge_attr.pt
```

- [ ] **Step 2: Run 5-episode GAT smoke test**

```
conda run -n gdrl_gpu python scripts/main.py --U 3 --L 4 --N 4 --T 20 --num_episodes 5 --use_gat
```

Expected: 5 episodes complete, no shape or NaN errors

- [ ] **Step 3: Run 5-episode GCN ablation smoke test**

```
conda run -n gdrl_gpu python scripts/main.py --U 3 --L 4 --N 4 --T 20 --num_episodes 5 --no-use_gat
```

Expected: 5 episodes complete

- [ ] **Step 4: Commit any fixes**

```bash
git add -p
git commit -m "fix: integration fixes for GAT smoke test"
```

---

## Batch Dimension Note

If GATv2Conv raises a shape error for edge_attr=[B,E,2], fall back to flat per-batch: in `_build_edge_attr`, return `edge_attr_out.reshape(batch_size * E, 2)` and tile edge_index. This fix goes inside `_build_edge_attr` only.

## Ablation Experiment

After implementation, compare GAT vs GCN over 1200 episodes:

```bash
conda run -n gdrl_gpu python compare_baselines.py --scenario custom_U3_L16_N16_T100 \
    --episodes 1200 --energy_weight 0.01 \
    --output_dir experiments/gat_compare_1200ep --use_gat

conda run -n gdrl_gpu python compare_baselines.py --scenario custom_U3_L16_N16_T100 \
    --episodes 1200 --energy_weight 0.01 \
    --output_dir experiments/gcn_compare_1200ep --no-use_gat
```
