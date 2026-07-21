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


import torch
from gymnasium import spaces


def _make_extractor(tmp_path, monkeypatch, U=3, L=8, N=8, use_gat=True):
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
