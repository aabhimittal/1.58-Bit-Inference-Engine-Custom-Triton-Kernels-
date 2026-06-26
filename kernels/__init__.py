"""Custom Triton kernels for packed-ternary (1.58-bit) matmul."""

from .bitnet_kernel import (
    HAS_TRITON,
    bitnet_matmul,
    ternary_matmul_reference,
)

__all__ = ["HAS_TRITON", "bitnet_matmul", "ternary_matmul_reference"]
