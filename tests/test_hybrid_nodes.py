import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from docs.GDRL.gdrl.core.nodes import (
    LEO_status, all_LEO_status,
    UAV_status, all_UAV_status,
    gNB_status, all_gNB_status,
    MEC_status, all_MEC_status,
)


def test_LEO_status():
    C, pos = LEO_status()
    assert 150 <= C <= 200
    assert 300e3 <= pos <= 500e3


def test_all_LEO_status():
    C, pos, ori = all_LEO_status(4)
    assert C.shape == (4,)
    assert pos.shape == (4,)
    assert ori.shape == (4,)


def test_UAV_status():
    C, pos = UAV_status()
    assert 80 <= C <= 120
    assert 100 <= pos <= 5000


def test_all_UAV_status():
    C, pos, ori = all_UAV_status(3)
    assert C.shape == (3,)
    assert pos.shape == (3,)
    assert ori.shape == (3,)


def test_gNB_status():
    C, pos = gNB_status()
    assert 200 <= C <= 500
    assert pos == 10


def test_all_gNB_status():
    C, pos, ori = all_gNB_status(2)
    assert C.shape == (2,)
    assert pos.shape == (2,)
    assert ori.shape == (2,)


def test_MEC_status():
    C, pos = MEC_status()
    assert 800 <= C <= 1500
    assert pos == 5


def test_all_MEC_status():
    C, pos, ori = all_MEC_status(2)
    assert C.shape == (2,)
    assert pos.shape == (2,)
    assert ori.shape == (2,)