"""DAG templates and generator (spec §3.1, W2.1).

Templates: 2–3 种（轻/重），节点 4–5、深度 3–4；示例骨架
采集 -> 检测 -> 融合 -> 决策。

Work unit semantics:
- 应用实例 = 一张 DAG，绑定到终端 n，输入数据在 n。
- 源节点（采集）的 ``in_bits`` 需从终端上行；非源节点的
  ``in_bits`` = 各前驱 ``out_bits`` 之和（跨设备时需传输）。
- 节点 ``cycles`` 为所需计算周期；``out_bits`` 为其输出载荷。

Sizes are sampled deterministically from the caller-provided RNG
(uniform 0.6–1.4x around template means).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class DAGNode:
    index: int
    name: str
    cycles: float          # compute cycles required
    in_bits: float         # bits that must reach the executing device before start
    out_bits: float        # payload forwarded to successors after execution
    parents: Tuple[int, ...]
    children: List[int] = field(default_factory=list)


@dataclass
class DAG:
    template_id: int
    name: str
    nodes: List[DAGNode]

    @property
    def sink(self) -> int:
        # decision node = last index by construction (topologically last)
        return max(n.index for n in self.nodes)

    @property
    def total_cycles(self) -> float:
        return sum(n.cycles for n in self.nodes)

    def topological_order(self) -> List[int]:
        """Kahn's algorithm; used by tests and as a safety check."""
        indeg = {n.index: len(n.parents) for n in self.nodes}
        queue = [n.index for n in self.nodes if not n.parents]
        order = []
        while queue:
            v = queue.pop(0)
            order.append(v)
            for c in self.nodes[v].children:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("DAG has a cycle")
        return order


# template: name, source input mean, node (name, cycles_mean, out_bits_mean),
#           edges (parent, child)
TEMPLATES = [
    {
        "name": "chain-light",
        "source_in_mean": 8.0e5,
        "nodes": [
            ("采集", 7.0e8, 3.0e5),
            ("检测", 6.0e8, 2.5e5),
            ("融合", 5.0e8, 2.0e5),
            ("决策", 6.0e8, 1.5e5),
        ],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
    {
        "name": "fork-join",
        "source_in_mean": 1.0e6,
        "nodes": [
            ("采集", 8.0e8, 4.0e5),
            ("检测A", 1.5e9, 3.0e5),
            ("检测B", 1.8e9, 3.0e5),
            ("融合", 7.0e8, 2.0e5),
            ("决策", 8.0e8, 1.5e5),
        ],
        "edges": [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)],
    },
    {
        "name": "chain-heavy",
        "source_in_mean": 1.2e6,
        "nodes": [
            ("采集", 1.6e9, 4.8e5),
            ("检测", 1.4e9, 4.0e5),
            ("融合", 1.1e9, 3.2e5),
            ("决策", 1.4e9, 2.4e5),
        ],
        "edges": [(0, 1), (1, 2), (2, 3)],
    },
]


def _clamp_positive(v: float) -> float:
    return max(float(v), 1.0)


def generate_dag(template_id: int, rng: np.random.Generator,
                 cycles_scale: float = 1.0, bits_scale: float = 1.0) -> DAG:
    """Sample one concrete DAG instance from the given template."""
    spec = TEMPLATES[template_id]
    factor = rng.uniform(0.6, 1.4)
    nodes = []
    for i, (name, cycles_mean, out_mean) in enumerate(spec["nodes"]):
        nodes.append(
            DAGNode(
                index=i,
                name=name,
                cycles=_clamp_positive(cycles_mean * factor * cycles_scale),
                in_bits=0.0,
                out_bits=_clamp_positive(out_mean * factor * bits_scale),
                parents=(),
            )
        )
    for parent, child in spec["edges"]:
        nodes[parent].children.append(child)
        nodes[child].parents = nodes[child].parents + (parent,)
    nodes[0].in_bits = _clamp_positive(spec["source_in_mean"] * factor * bits_scale)
    for i in range(1, len(nodes)):
        nodes[i].in_bits = _clamp_positive(
            sum(nodes[p].out_bits for p in nodes[i].parents)
        )
    return DAG(template_id=template_id, name=spec["name"], nodes=nodes)
