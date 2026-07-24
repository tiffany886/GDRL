import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# 必须在导入 gdrl.models.amn 之前设置 sys.argv，避免 feature.py 的 get_args() 报错
sys.argv = ['test', '--U', '3', '--L', '4', '--N', '4']
import torch
import numpy as np
# 直接从 amn.py 导入，绕过 __init__.py 触发 feature.py
import importlib.util
_spec = importlib.util.spec_from_file_location("amn_mod", os.path.join(os.path.dirname(__file__), "..", "gdrl", "models", "amn.py"))
_amn_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_amn_mod)
M1EncoderPartial = _amn_mod.M1EncoderPartial
M1DecoderPartial = _amn_mod.M1DecoderPartial
AutoencoderPartial = _amn_mod.AutoencoderPartial
M1EncoderDis = _amn_mod.M1EncoderDis
M1DecoderDis = _amn_mod.M1DecoderDis
AutoencoderDis = _amn_mod.AutoencoderDis


def test_encoder_partial_output():
    """M1EncoderPartial 输出卸载比例 + 离散动作。"""
    U = 3
    action_space_len = 20
    encoder = M1EncoderPartial(16, action_space_len, U)
    x = torch.randn(4, 16)
    user_lists = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8, 9]}
    dis_action, offload_ratio = encoder(x, user_lists, differentiable=False)
    assert dis_action.shape == (4, U)
    assert offload_ratio.shape == (4, U)
    # offload_ratio 应该在 [0, 1]
    assert float(offload_ratio.min()) >= 0.0
    assert float(offload_ratio.max()) <= 1.0


def test_encoder_partial_differentiable():
    """STE 可微模式下，offload_ratio 保持可微。"""
    U = 3
    action_space_len = 20
    encoder = M1EncoderPartial(16, action_space_len, U)
    x = torch.randn(4, 16)
    user_lists = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8, 9]}
    dis_action, offload_ratio = encoder(x, user_lists, differentiable=True)
    assert dis_action.shape == (4, U)
    assert offload_ratio.shape == (4, U)
    # offload_ratio 应该有梯度
    assert offload_ratio.requires_grad or x.requires_grad


def test_autoencoder_partial_forward():
    """AutoencoderPartial forward 不报错。"""
    U = 3
    action_space_len = 20
    user_lists = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8, 9]}
    ae = AutoencoderPartial(16, action_space_len, U, user_lists,
                            M1EncoderPartial, M1DecoderPartial)
    x = torch.randn(4, 16)
    output = ae(x)
    assert output.shape == (4, 16)  # 重构输入


def test_backward_compat_dis_encoder():
    """原版 M1EncoderDis 仍然工作。"""
    U = 3
    action_space_len = 20
    encoder = M1EncoderDis(16, action_space_len)
    x = torch.randn(4, 16)
    user_lists = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8, 9]}
    output = encoder(x, user_lists, differentiable=False)
    assert output.shape == (4, U)


def test_backward_compat_autoencoder_dis():
    """原版 AutoencoderDis 仍然工作。"""
    U = 3
    action_space_len = 20
    user_lists = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7, 8, 9]}
    ae = AutoencoderDis(16, action_space_len, U, user_lists,
                        M1EncoderDis, M1DecoderDis)
    x = torch.randn(4, 16)
    output = ae(x)
    assert output.shape == (4, 16)