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

def test_tx_energy_clamp():
    # Very high power * very high data → should clamp to MAX_ENERGY_J=1.0
    e = tx_energy(power_w=100.0, data_bits=1e9, rate_bps=1.0)
    assert e == pytest.approx(1.0)

def test_comp_energy_unknown_node_type_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Unknown node_type"):
        comp_energy(node_type="gpu_cluster", allocated_vms=1.0, process_time_s=0.001)
