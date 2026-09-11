import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from docs.GDRL.gdrl.core.graph import GenerateAdjacency, GenerateAdjacency_hybrid, _decode_action_12bit


def test_generate_adjacency_backward_compat():
    """原版 3 参数接口仍然工作。"""
    a, Adj, ul = GenerateAdjacency(3, 16, 16)
    assert isinstance(a, int)
    assert Adj.shape == (3 + 16 + 16, 3 + 16 + 16)
    assert isinstance(ul, dict)
    assert len(ul) == 3


def test_hybrid_adjacency_shape():
    """Hybrid 5 层图邻接矩阵形状正确。"""
    U, G, V, L, M = 5, 2, 4, 8, 1
    a, Adj, ul, node_order = GenerateAdjacency_hybrid(U, G, V, L, M)
    total = U + G + V + L + M
    assert Adj.shape == (total, total)
    assert len(ul) == U
    assert node_order["UE"] == 0
    assert node_order["gNB"] == U
    assert node_order["UAV"] == U + G
    assert node_order["LEO"] == U + G + V
    assert node_order["MEC"] == U + G + V + L


def test_hybrid_action_space_nonzero():
    """Action space 应该大于 0。"""
    a, Adj, ul, _ = GenerateAdjacency_hybrid(5, 2, 4, 8, 1)
    assert a > 0
    for u in ul:
        assert len(ul[u]) > 0


def test_decode_action_12bit():
    """12-bit 动作解码正确。"""
    # gNB 直连非本地，节点 3
    bits = "00" + "0" + "0" + bin(3)[2:].zfill(8)
    action_int = int(bits, 2)
    node_type, is_indirect, is_local, node_idx = _decode_action_12bit(action_int)
    assert node_type == "00"
    assert is_indirect is False
    assert is_local is False
    assert node_idx == 3

    # MEC 间接本地，节点 5
    bits = "11" + "1" + "1" + bin(5)[2:].zfill(8)
    action_int = int(bits, 2)
    node_type, is_indirect, is_local, node_idx = _decode_action_12bit(action_int)
    assert node_type == "11"
    assert is_indirect is True
    assert is_local is True
    assert node_idx == 5


def test_hybrid_symmetric_adjacency():
    """邻接矩阵应该是对称的。"""
    a, Adj, ul, _ = GenerateAdjacency_hybrid(5, 2, 4, 8, 1)
    assert np.allclose(Adj, Adj.T)


def test_hybrid_large_scale():
    """Large 场景 (U=30) 能生成。"""
    U, G, V, L, M = 30, 3, 6, 12, 2
    a, Adj, ul, node_order = GenerateAdjacency_hybrid(U, G, V, L, M)
    total = U + G + V + L + M
    assert Adj.shape == (total, total)
    assert total == 53
    assert a > 100  # 足够大的动作空间