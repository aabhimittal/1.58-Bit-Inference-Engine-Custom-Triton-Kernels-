"""Activation quantization + W1.58-A8 reference tests (CPU; runs anywhere)."""

import torch
import pytest

from bitnet.quantize import absmean_quantize, activation_quant, pack_ternary
from kernels.bitnet_kernel import (
    bitnet_matmul_a8,
    ternary_matmul_a8_reference,
)


def test_activation_quant_is_int8_and_close():
    torch.manual_seed(0)
    x = torch.randn(4, 512)
    x_int8, inv_scale = activation_quant(x)
    assert x_int8.dtype == torch.int8
    assert int(x_int8.abs().max()) <= 127
    assert inv_scale.shape == (4, 1)
    rel = (x_int8.float() * inv_scale - x).norm() / x.norm()
    assert rel < 0.05  # int8 per-token should track activations closely


def test_activation_quant_per_token_scale():
    # rows with very different magnitudes must get independent scales
    x = torch.cat([torch.randn(1, 64), 100.0 * torch.randn(1, 64)], dim=0)
    _, inv_scale = activation_quant(x)
    assert inv_scale[1] > 10 * inv_scale[0]


def test_a8_reference_matches_manual_integer_matmul():
    torch.manual_seed(1)
    x = torch.randn(3, 256)
    w = torch.randn(40, 256)
    ternary, gamma = absmean_quantize(w)
    packed, _ = pack_ternary(ternary)

    y = ternary_matmul_a8_reference(x, packed, gamma, k=256)

    x_int8, inv_scale = activation_quant(x)
    expected = (x_int8.float() * inv_scale) @ (ternary * gamma).t()
    assert torch.allclose(y, expected, atol=1e-4, rtol=1e-4)


def test_a8_dispatch_uses_reference_on_cpu():
    torch.manual_seed(2)
    x = torch.randn(2, 128)
    ternary, gamma = absmean_quantize(torch.randn(16, 128))
    packed, _ = pack_ternary(ternary)
    y = bitnet_matmul_a8(x, packed, gamma, k=128)  # CPU path
    ref = ternary_matmul_a8_reference(x, packed, gamma, k=128)
    assert torch.equal(y, ref)


def test_a8_bias():
    torch.manual_seed(3)
    x = torch.randn(2, 64)
    ternary, gamma = absmean_quantize(torch.randn(8, 64))
    packed, _ = pack_ternary(ternary)
    bias = torch.randn(8)
    y = bitnet_matmul_a8(x, packed, gamma, k=64, bias=bias)
    ref = ternary_matmul_a8_reference(x, packed, gamma, k=64, bias=bias)
    assert torch.allclose(y, ref, atol=1e-5)


def test_bitlinear_a8_forward_shape_and_scheme():
    import torch.nn as nn
    from bitnet.bitlinear import BitLinear

    lin = nn.Linear(128, 64)
    bit = BitLinear.from_linear(lin, quantize_activations=True)
    assert bit.quantize_activations is True
    assert "W1.58-A8" in repr(bit)
    out = bit(torch.randn(2, 5, 128))
    assert out.shape == (2, 5, 64)
