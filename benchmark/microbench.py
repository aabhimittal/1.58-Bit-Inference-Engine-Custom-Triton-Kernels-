"""Synthetic microbenchmark: custom ternary kernel vs fp16 ``F.linear`` (cuBLAS).

Run on a GPU (Kaggle T4)::

    python benchmark/microbench.py

It reports, per shape and batch size ``M``: latency, the custom kernel's speedup over
fp16, achieved weight bandwidth, and the numerical error vs the dequantized reference.
The headline is the decode path (``M=1``), where reading 2-bit weights instead of fp16
removes the HBM bottleneck.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from bitnet.quantize import absmean_quantize, pack_ternary
from kernels.bitnet_kernel import HAS_TRITON, bitnet_matmul, ternary_matmul_reference

# (N out_features, K in_features) - representative LLM projection shapes.
SHAPES = [(2048, 2048), (4096, 4096), (5120, 5120), (4096, 11008)]
BATCHES = [1, 2, 8, 64]


def _time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run():
    if not (HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("This benchmark requires CUDA + Triton (run on a Kaggle T4).")

    print(f"{'N x K':>14} {'M':>4} {'fp16 ms':>9} {'bit ms':>9} "
          f"{'speedup':>8} {'bit GB/s':>9} {'max err':>9}")
    print("-" * 70)

    for (n, k) in SHAPES:
        w = torch.randn(n, k)
        ternary, gamma = absmean_quantize(w)
        packed = pack_ternary(ternary)[0].cuda()
        gamma_c = gamma.cuda()
        w_fp16 = (ternary * gamma).to(torch.float16).cuda()  # dense baseline weight

        for m in BATCHES:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)

            fp16_ms = _time(lambda: F.linear(x, w_fp16))
            bit_ms = _time(lambda: bitnet_matmul(x, packed, gamma_c, k=k))

            # weight bytes moved by the custom kernel = 2 bits/value = N*K/4 bytes
            bit_bytes = n * k / 4
            bit_gbps = bit_bytes / (bit_ms * 1e-3) / 1e9

            y = bitnet_matmul(x, packed, gamma_c, k=k)
            ref = ternary_matmul_reference(x, packed, gamma_c, k=k)
            max_err = (y.float() - ref.float()).abs().max().item()

            print(f"{n:>6}x{k:<7} {m:>4} {fp16_ms:>9.3f} {bit_ms:>9.3f} "
                  f"{fp16_ms / bit_ms:>7.2f}x {bit_gbps:>9.1f} {max_err:>9.4f}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
