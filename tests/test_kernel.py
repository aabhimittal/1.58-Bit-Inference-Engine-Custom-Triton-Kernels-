"""Triton kernel correctness tests (GPU-only; skipped without CUDA + Triton)."""

import pytest
import torch

from bitnet.quantize import absmean_quantize, pack_ternary
from kernels.bitnet_kernel import (
    HAS_TRITON,
    bitnet_matmul,
    ternary_matmul_reference,
)

pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="requires CUDA + Triton (run on a Kaggle T4)",
)


def _make(n, k, seed=0, per_row=False):
    torch.manual_seed(seed)
    w = torch.randn(n, k)
    ternary, gamma = absmean_quantize(w, per_row=per_row)
    packed, _ = pack_ternary(ternary)
    return packed.cuda(), gamma.cuda(), ternary, gamma


@pytest.mark.parametrize("m", [1, 4, 32, 128])
@pytest.mark.parametrize("n,k", [(256, 512), (512, 2048), (1024, 4096)])
def test_kernel_matches_reference(m, n, k):
    packed, gamma_c, _, _ = _make(n, k, seed=m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    y = bitnet_matmul(x, packed, gamma_c, k=k)
    ref = ternary_matmul_reference(x, packed, gamma_c, k=k)
    # fp16 accumulation -> compare with a tolerance scaled to the contraction depth
    torch.testing.assert_close(y.float(), ref.float(), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("m", [1, 64])
def test_kernel_per_row_scale_and_bias(m):
    n, k = 512, 1024
    packed, gamma_c, _, _ = _make(n, k, seed=7, per_row=True)
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    y = bitnet_matmul(x, packed, gamma_c, k=k, bias=bias)
    ref = ternary_matmul_reference(x, packed, gamma_c, k=k, bias=bias)
    torch.testing.assert_close(y.float(), ref.float(), atol=1e-1, rtol=1e-2)


def test_bitlinear_forward_matches_dequant():
    import torch.nn as nn
    from bitnet.bitlinear import BitLinear

    lin = nn.Linear(1024, 512).cuda().half()
    bit = BitLinear.from_linear(lin).cuda()
    x = torch.randn(2, 1024, device="cuda", dtype=torch.float16)
    y = bit(x)
    ref = x @ bit.dequantized_weight().cuda().t().half() + bit.bias.half()
    torch.testing.assert_close(y.float(), ref.float(), atol=1e-1, rtol=1e-2)
