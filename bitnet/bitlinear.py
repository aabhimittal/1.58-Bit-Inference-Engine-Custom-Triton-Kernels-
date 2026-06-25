"""``BitLinear`` - a drop-in ``nn.Linear`` replacement backed by the ternary kernel."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import WEIGHTS_PER_BYTE, absmean_quantize, pack_ternary, dequantize

# Pad K to a multiple of this when packing so the Triton kernel's largest BLOCK_K never
# reads past the packed buffer. Must be a multiple of WEIGHTS_PER_BYTE.
PACK_PAD = 512
assert PACK_PAD % WEIGHTS_PER_BYTE == 0


class BitLinear(nn.Module):
    """Linear layer whose weight is stored as packed ternary ``{-1,0,1}`` + a scale.

    The forward pass calls the custom Triton kernel (:func:`kernels.bitnet_matmul`) on
    CUDA and transparently falls back to a dequantize-then-``F.linear`` path on CPU or
    when Triton is unavailable, so the layer is always numerically correct.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        per_row: bool = False,
        device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_row = per_row

        k_padded = ((in_features + PACK_PAD - 1) // PACK_PAD) * PACK_PAD
        # Non-trainable packed weight + scale (this layer is inference-only).
        self.register_buffer(
            "packed_weight",
            torch.zeros(
                (out_features, k_padded // WEIGHTS_PER_BYTE),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "gamma",
            torch.ones(
                (out_features, 1) if per_row else (1,),
                dtype=torch.float32,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros(out_features, dtype=torch.float32, device=device)
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, per_row: bool = False) -> "BitLinear":
        """Quantize and pack an existing ``nn.Linear`` into a ``BitLinear``."""
        out_features, in_features = linear.weight.shape
        layer = cls(
            in_features,
            out_features,
            bias=linear.bias is not None,
            per_row=per_row,
            device=linear.weight.device,
        )
        ternary, gamma = absmean_quantize(linear.weight.data, per_row=per_row)
        packed, _ = pack_ternary(ternary, pad_to=PACK_PAD)
        layer.packed_weight.copy_(packed)
        layer.gamma.copy_(gamma.reshape(layer.gamma.shape).to(torch.float32))
        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.to(torch.float32))
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Import here so importing bitnet never hard-requires Triton.
        from kernels.bitnet_kernel import bitnet_matmul

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        bias = self.bias if self.bias is not None else None
        y = bitnet_matmul(
            x2d, self.packed_weight, self.gamma, self.in_features, bias=bias
        )
        return y.reshape(*orig_shape[:-1], self.out_features)

    def dequantized_weight(self) -> torch.Tensor:
        """Return the reconstructed float weight ``[out, in]`` (for inspection/tests)."""
        return dequantize(self.packed_weight, self.gamma, self.in_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, per_row={self.per_row}, packed_ternary=2bit"
        )
