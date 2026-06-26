"""Pack/unpack round-trip and quantization reference tests (CPU; runs anywhere)."""

import torch
import pytest

from bitnet.quantize import (
    absmean_quantize,
    pack_ternary,
    unpack_ternary,
    dequantize,
)


@pytest.mark.parametrize("shape", [(8, 16), (5, 7), (128, 4096), (3, 1)])
def test_pack_unpack_roundtrip(shape):
    torch.manual_seed(0)
    ternary = torch.randint(-1, 2, shape).float()  # {-1,0,1}
    packed, k_padded = pack_ternary(ternary)
    assert packed.dtype == torch.uint8
    assert k_padded % 4 == 0 and k_padded >= shape[1]
    restored = unpack_ternary(packed, k=shape[1])
    assert torch.equal(restored, ternary)


def test_pack_pads_k_to_multiple_of_pad_to():
    ternary = torch.randint(-1, 2, (4, 30)).float()
    packed, k_padded = pack_ternary(ternary, pad_to=64)
    assert k_padded == 64
    assert packed.shape == (4, 64 // 4)
    # padding decodes to ternary 0 and is dropped when slicing back to K
    assert torch.equal(unpack_ternary(packed, k=30), ternary)


def test_pack_rejects_out_of_range():
    bad = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    with pytest.raises(ValueError):
        pack_ternary(bad)


def test_absmean_quantize_is_ternary_and_close():
    torch.manual_seed(1)
    w = torch.randn(64, 256)
    ternary, gamma = absmean_quantize(w)
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert gamma.numel() == 1
    approx = ternary * gamma
    # absmean ternary should track the sign structure and rough magnitude
    rel = (approx - w).norm() / w.norm()
    assert rel < 0.95


def test_absmean_per_row_scale_shape():
    w = torch.randn(32, 128)
    ternary, gamma = absmean_quantize(w, per_row=True)
    assert gamma.shape == (32, 1)


def test_dequantize_matches_unpack_times_gamma():
    torch.manual_seed(2)
    w = torch.randn(16, 320)
    ternary, gamma = absmean_quantize(w)
    packed, _ = pack_ternary(ternary)
    deq = dequantize(packed, gamma, k=320)
    assert torch.equal(deq, ternary * gamma)


def test_reference_matmul_matches_dense():
    """The packed-ternary reference path equals an explicit dense ternary matmul."""
    from kernels.bitnet_kernel import ternary_matmul_reference

    torch.manual_seed(3)
    w = torch.randn(48, 200)
    ternary, gamma = absmean_quantize(w)
    packed, _ = pack_ternary(ternary)
    x = torch.randn(4, 200)
    y = ternary_matmul_reference(x, packed, gamma, k=200)
    expected = x @ (ternary * gamma).t()
    assert torch.allclose(y, expected, atol=1e-4, rtol=1e-4)
