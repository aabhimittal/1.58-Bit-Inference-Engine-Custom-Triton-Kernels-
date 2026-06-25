"""Triton kernels for packed-ternary ("1.58-bit") matmul ``y = x @ Wᵀ``.

The weight ``W [N, K]`` is stored as ternary ``{-1, 0, 1}`` packed four-per-byte
(see :mod:`bitnet.quantize`) plus a scale ``gamma``. Two kernels are provided:

* ``_bitnet_gemv_kernel`` - the **decode path** (small ``M``). It unpacks the ternary
  codes in registers and accumulates with ``tl.where`` add/subtract -- the literal
  "addition-only multiplication": ``w==1`` adds the activation, ``w==-1`` subtracts it,
  ``w==0`` skips it. No floating-point multiply by the weight is performed. The win on a
  T4 is bandwidth: only ``K/4`` bytes of weight are streamed instead of ``2K`` (fp16).

* ``_bitnet_matmul_kernel`` - the **prefill / large-M path**. It unpacks ternary into an
  fp16 operand and uses ``tl.dot`` (tensor cores) for throughput. Multiplying by a
  ``{-1,0,1}`` operand is still just add/sub/skip, but materialising it lets the MAC
  pipeline stay busy when ``M`` is large.

The Python entry point :func:`bitnet_matmul` dispatches to the GEMV kernel for small
``M`` and the GEMM kernel otherwise, and falls back to a pure-torch reference when
Triton/CUDA is unavailable so callers always get correct results.
"""

from __future__ import annotations

from typing import Optional

import torch

from bitnet.quantize import WEIGHTS_PER_BYTE, dequantize

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on machines without Triton
    HAS_TRITON = False

# Small-M cutoff below which the bandwidth-bound GEMV kernel is preferred over tl.dot.
GEMV_MAX_M = 8


def ternary_matmul_reference(
    x: torch.Tensor,
    packed: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Golden reference: dequantize the packed weight and run a normal matmul.

    Shapes: ``x [M, K]``, ``packed [N, K_padded//4]`` uint8, output ``[M, N]``.
    """
    weight = dequantize(packed, gamma, k).to(x.dtype)  # [N, K]
    y = x @ weight.t()
    if bias is not None:
        y = y + bias
    return y


if HAS_TRITON:

    def _gemv_configs():
        return [
            triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w)
            for bn in (64, 128, 256)
            for bk in (256, 512)
            for w in (4, 8)
        ]

    @triton.autotune(configs=_gemv_configs(), key=["N", "K"])
    @triton.jit
    def _bitnet_gemv_kernel(
        x_ptr,
        packed_ptr,
        gamma_ptr,
        bias_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_ym,
        stride_yn,
        PER_ROW_SCALE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
        byte_col = offs_k // WEIGHTS_PER_BYTE  # which byte within the block
        shift = (offs_k % WEIGHTS_PER_BYTE) * 2  # bit offset within that byte
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < K
            x = tl.load(
                x_ptr + pid_m * stride_xm + k_idx * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)  # [BLOCK_K]

            p_ptr = (
                packed_ptr
                + offs_n[:, None] * stride_pn
                + (k0 // WEIGHTS_PER_BYTE + byte_col[None, :]) * stride_pk
            )
            p = tl.load(
                p_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0
            ).to(tl.int32)  # [BLOCK_N, BLOCK_K] (same byte read 4x)
            w = ((p >> shift[None, :]) & 0b11) - 1  # {-1,0,1}

            xb = x[None, :]
            # addition-only: +x where w==1, -x where w==-1, nothing where w==0
            contrib = tl.where(w == 1, xb, 0.0) - tl.where(w == -1, xb, 0.0)
            acc += tl.sum(contrib, axis=1)

        if PER_ROW_SCALE:
            gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        else:
            gamma = tl.load(gamma_ptr).to(tl.float32)
        acc = acc * gamma

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)

        tl.store(
            y_ptr + pid_m * stride_ym + offs_n * stride_yn,
            acc,
            mask=n_mask,
        )

    def _gemm_configs():
        return [
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
            )
            for bm in (32, 64)
            for bn in (64, 128)
            for bk in (32, 64)
            for w in (4,)
            for s in (2, 3)
        ]

    @triton.autotune(configs=_gemm_configs(), key=["M", "N", "K"])
    @triton.jit
    def _bitnet_matmul_kernel(
        x_ptr,
        packed_ptr,
        gamma_ptr,
        bias_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_pn,
        stride_pk,
        stride_ym,
        stride_yn,
        PER_ROW_SCALE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        byte_col = offs_k // WEIGHTS_PER_BYTE
        shift = (offs_k % WEIGHTS_PER_BYTE) * 2
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float16)  # [BLOCK_M, BLOCK_K]

            # unpack weight as [BLOCK_K, BLOCK_N] for tl.dot
            p_ptr = (
                packed_ptr
                + offs_n[None, :] * stride_pn
                + (k0 // WEIGHTS_PER_BYTE + byte_col[:, None]) * stride_pk
            )
            p = tl.load(
                p_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0
            ).to(tl.int32)
            w = (((p >> shift[:, None]) & 0b11) - 1).to(tl.float16)  # {-1,0,1}
            acc += tl.dot(x, w)

        if PER_ROW_SCALE:
            gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        else:
            gamma = tl.load(gamma_ptr).to(tl.float32)
        acc = acc * gamma[None, :]

        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[
                None, :
            ]

        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


def bitnet_matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    gamma: torch.Tensor,
    k: int,
    bias: Optional[torch.Tensor] = None,
    force_reference: bool = False,
) -> torch.Tensor:
    """Compute ``y = (x @ Wᵀ) + bias`` for packed-ternary ``W``.

    Args:
        x: ``[M, K]`` activations (fp16/fp32) on CUDA for the kernel path.
        packed: ``[N, K_padded//4]`` uint8 packed ternary weight.
        gamma: scale, shape ``[1]`` (per-tensor) or ``[N]``/``[N,1]`` (per-row).
        k: original (unpadded) ``K``.
        bias: optional ``[N]`` bias.
        force_reference: skip the kernel and use the torch reference (for testing).

    Returns:
        ``y`` of shape ``[M, N]`` in ``x``'s dtype.
    """
    if force_reference or not HAS_TRITON or not x.is_cuda:
        return ternary_matmul_reference(x, packed, gamma, k, bias)

    assert x.dim() == 2, "x must be 2D [M, K]"
    m, k_in = x.shape
    n = packed.shape[0]
    assert k_in == k, f"x has K={k_in} but weight expects K={k}"

    gamma_flat = gamma.reshape(-1).to(torch.float32).contiguous()
    per_row = gamma_flat.numel() == n
    if not per_row:
        assert gamma_flat.numel() == 1, "gamma must have 1 or N elements"

    bias_t = bias.contiguous() if bias is not None else x.new_zeros(1)
    has_bias = bias is not None

    x = x.contiguous()
    packed = packed.contiguous()
    y = x.new_empty((m, n), dtype=torch.float32)

    # K must be padded to a multiple of WEIGHTS_PER_BYTE for the packing; the kernels
    # mask the K tail, so any padding stored in `packed` is simply ignored here.
    if m <= GEMV_MAX_M:
        grid = lambda meta: (m, triton.cdiv(n, meta["BLOCK_N"]))
        _bitnet_gemv_kernel[grid](
            x, packed, gamma_flat, bias_t, y,
            m, n, k,
            x.stride(0), x.stride(1),
            packed.stride(0), packed.stride(1),
            y.stride(0), y.stride(1),
            PER_ROW_SCALE=per_row, HAS_BIAS=has_bias,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]),
            triton.cdiv(n, meta["BLOCK_N"]),
        )
        _bitnet_matmul_kernel[grid](
            x, packed, gamma_flat, bias_t, y,
            m, n, k,
            x.stride(0), x.stride(1),
            packed.stride(0), packed.stride(1),
            y.stride(0), y.stride(1),
            PER_ROW_SCALE=per_row, HAS_BIAS=has_bias,
        )
    return y.to(x.dtype)
