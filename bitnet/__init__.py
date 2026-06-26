"""1.58-bit (ternary) inference engine with custom Triton kernels.

Public API:
    - quantize: absmean ternary quantization + 2-bit packing helpers
    - BitLinear: drop-in nn.Linear replacement backed by a packed-ternary Triton kernel
    - model_utils: load a BitNet model and swap its Linear layers for BitLinear
"""

from .quantize import (
    absmean_quantize,
    pack_ternary,
    unpack_ternary,
    dequantize,
)
from .bitlinear import BitLinear

__all__ = [
    "absmean_quantize",
    "pack_ternary",
    "unpack_ternary",
    "dequantize",
    "BitLinear",
]
