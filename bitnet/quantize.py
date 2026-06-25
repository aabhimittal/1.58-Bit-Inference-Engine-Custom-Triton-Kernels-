"""Ternary (1.58-bit) quantization and 2-bit packing utilities.

BitNet b1.58 represents each weight with a value in ``{-1, 0, 1}`` ("1.58 bit" =
log2(3)) plus a single floating-point scale ``gamma`` shared across the tensor (or
across each output row). This module provides:

* :func:`absmean_quantize` - the BitNet "absmean" weight quantizer.
* :func:`pack_ternary` / :func:`unpack_ternary` - pack four ternary values into one
  ``uint8`` (2 bits each) and the exact inverse.
* :func:`dequantize` - reconstruct the floating-point weight for reference math.

Packing layout
--------------
A weight matrix ``W`` has shape ``[N, K]`` (``out_features``, ``in_features``). We pack
*along K* so four consecutive ternary codes become one byte::

    code = ternary + 1            # {-1,0,1} -> {0,1,2}, 2 bits each
    byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)

giving ``packed`` of shape ``[N, K_padded // 4]`` (``K`` is zero-padded up to a
multiple of ``pad_to``; the padding code 1 -> weight 0 is a no-op in the matmul).
The weight then occupies 2 bits/value instead of 16, an ~8x reduction in the HBM
traffic that dominates autoregressive decode on a T4.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Ternary value -1/0/1 is stored as the unsigned 2-bit code value+1 (0/1/2).
CODE_OFFSET = 1
BITS_PER_WEIGHT = 2
WEIGHTS_PER_BYTE = 8 // BITS_PER_WEIGHT  # 4


def absmean_quantize(
    weight: torch.Tensor, per_row: bool = False, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` to ternary ``{-1,0,1}`` using BitNet's absmean rule.

    Args:
        weight: float tensor of shape ``[N, K]``.
        per_row: if ``True`` use one scale per output row (shape ``[N, 1]``);
            otherwise a single per-tensor scale (shape ``[1]``).
        eps: floor on the scale to avoid division by zero for all-zero rows.

    Returns:
        ``(ternary, gamma)`` where ``ternary`` is a float tensor in ``{-1,0,1}`` with
        the same shape as ``weight`` and ``gamma`` is the scale such that
        ``ternary * gamma`` approximates ``weight``.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")
    w = weight.float()
    if per_row:
        gamma = w.abs().mean(dim=1, keepdim=True).clamp_min(eps)  # [N, 1]
    else:
        gamma = w.abs().mean().clamp_min(eps).reshape(1)  # [1]
    ternary = torch.round(w / gamma).clamp_(-1, 1)
    return ternary, gamma


def pack_ternary(
    ternary: torch.Tensor, pad_to: int = WEIGHTS_PER_BYTE
) -> Tuple[torch.Tensor, int]:
    """Pack a ternary ``[N, K]`` tensor into ``uint8`` of shape ``[N, K_padded // 4]``.

    Args:
        ternary: tensor with values in ``{-1, 0, 1}`` (any dtype).
        pad_to: ``K`` is zero-padded up to a multiple of this value. Must itself be a
            multiple of 4 so packing stays byte-aligned. Use the kernel's ``BLOCK_K``
            to guarantee the kernel never reads past the buffer.

    Returns:
        ``(packed_uint8, K_padded)``.
    """
    if ternary.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(ternary.shape)}")
    if pad_to % WEIGHTS_PER_BYTE != 0:
        raise ValueError(f"pad_to must be a multiple of {WEIGHTS_PER_BYTE}")

    n, k = ternary.shape
    k_padded = ((k + pad_to - 1) // pad_to) * pad_to
    codes = (ternary.round().to(torch.int64) + CODE_OFFSET)  # {0,1,2}
    if (codes < 0).any() or (codes > 2).any():
        raise ValueError("ternary values must lie in {-1, 0, 1}")

    if k_padded != k:
        pad = torch.full(
            (n, k_padded - k), CODE_OFFSET, dtype=codes.dtype, device=codes.device
        )  # CODE_OFFSET == ternary 0
        codes = torch.cat([codes, pad], dim=1)

    codes = codes.reshape(n, k_padded // WEIGHTS_PER_BYTE, WEIGHTS_PER_BYTE)
    shifts = torch.arange(
        0, 8, BITS_PER_WEIGHT, device=codes.device, dtype=codes.dtype
    )  # [0,2,4,6]
    packed = (codes << shifts).sum(dim=-1).to(torch.uint8)
    return packed, k_padded


def unpack_ternary(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Inverse of :func:`pack_ternary`, returning a float ternary ``[N, K]`` tensor.

    Args:
        packed: ``uint8`` tensor of shape ``[N, K_padded // 4]``.
        k: the original (unpadded) ``K`` to slice back to.
    """
    if packed.dim() != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(packed.shape)}")
    p = packed.to(torch.int64)
    shifts = torch.arange(0, 8, BITS_PER_WEIGHT, device=p.device, dtype=p.dtype)
    codes = (p.unsqueeze(-1) >> shifts) & 0b11  # [N, K/4, 4] -> {0,1,2}
    n = packed.shape[0]
    ternary = (codes.reshape(n, -1) - CODE_OFFSET).float()  # {-1,0,1}
    return ternary[:, :k]


def dequantize(packed: torch.Tensor, gamma: torch.Tensor, k: int) -> torch.Tensor:
    """Reconstruct the float weight ``[N, K]`` from packed codes and scale.

    Used as the reference (golden) path the Triton kernel is checked against and as the
    CPU/large-batch fallback inside :class:`~bitnet.bitlinear.BitLinear`.
    """
    return unpack_ternary(packed, k) * gamma.to(torch.float32)
