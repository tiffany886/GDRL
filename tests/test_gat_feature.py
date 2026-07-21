import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_use_gat_flag_default(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "8", "--N", "8"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is True


def test_use_gat_flag_disabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--U", "3", "--L", "8", "--N", "8", "--no-use_gat"])
    from importlib import reload
    import arg_parser
    reload(arg_parser)
    args = arg_parser.get_args()
    assert args.use_gat is False


import numpy as np
import torch


def test_compute_edge_attr_shape():
    from gdrl.core.graph import GenerateAdjacency, compute_edge_attr
    U, L, N = 3, 8, 8
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
