"""Roofline / bandwidth analysis for the ternary kernels, with JSON + PNG export.

For the decode GEMV, each output needs the whole weight matrix once, so the op is
purely memory bound: ideal time = weight_bytes / peak_HBM_bandwidth. Ternary weights are
2 bits/value, so their ideal time is ~8x below the fp16 baseline. This script measures
how close each kernel gets to its roofline and saves a plot.

Run on a GPU (Kaggle T4)::

    python benchmark/roofline.py --out results/
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from bitnet.quantize import absmean_quantize, pack_ternary
from kernels.bitnet_kernel import (
    HAS_TRITON,
    bitnet_matmul,
    bitnet_matmul_a8,
)

# T4 spec: ~320 GB/s HBM. Override with --peak-bw for other GPUs.
DEFAULT_PEAK_BW_GBPS = 320.0
SHAPES = [(2048, 2048), (4096, 4096), (5120, 5120), (4096, 11008), (8192, 8192)]


def _time(fn, iters=100, warmup=20):
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


def run(out_dir: str, peak_bw: float):
    if not (HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("Roofline requires CUDA + Triton (run on a Kaggle T4).")
    os.makedirs(out_dir, exist_ok=True)

    results = {"peak_bw_gbps": peak_bw, "device": torch.cuda.get_device_name(0), "rows": []}
    print(f"{'N x K':>14} {'fp16 ms':>9} {'A16 ms':>8} {'A8 ms':>8} "
          f"{'A16 x':>6} {'A8 x':>6} {'A16 %roof':>10} {'A8 %roof':>9}")
    print("-" * 82)

    for (n, k) in SHAPES:
        ternary, gamma = absmean_quantize(torch.randn(n, k))
        packed = pack_ternary(ternary)[0].cuda()
        gamma_c = gamma.cuda()
        w_fp16 = (ternary * gamma).to(torch.float16).cuda()
        x = torch.randn(1, k, device="cuda", dtype=torch.float16)  # decode M=1

        fp16_ms = _time(lambda: F.linear(x, w_fp16))
        a16_ms = _time(lambda: bitnet_matmul(x, packed, gamma_c, k=k))
        a8_ms = _time(lambda: bitnet_matmul_a8(x, packed, gamma_c, k=k))

        # roofline: ideal ms to stream the weight from HBM
        fp16_ideal = (n * k * 2) / (peak_bw * 1e9) * 1e3
        ter_ideal = (n * k / 4) / (peak_bw * 1e9) * 1e3
        row = {
            "n": n, "k": k,
            "fp16_ms": fp16_ms, "a16_ms": a16_ms, "a8_ms": a8_ms,
            "a16_speedup": fp16_ms / a16_ms, "a8_speedup": fp16_ms / a8_ms,
            "a16_pct_roofline": 100 * ter_ideal / a16_ms,
            "a8_pct_roofline": 100 * ter_ideal / a8_ms,
            "fp16_pct_roofline": 100 * fp16_ideal / fp16_ms,
        }
        results["rows"].append(row)
        print(f"{n:>6}x{k:<7} {fp16_ms:>9.3f} {a16_ms:>8.3f} {a8_ms:>8.3f} "
              f"{row['a16_speedup']:>5.2f}x {row['a8_speedup']:>5.2f}x "
              f"{row['a16_pct_roofline']:>9.1f}% {row['a8_pct_roofline']:>8.1f}%")

    json_path = os.path.join(out_dir, "roofline.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {json_path}")
    _plot(results, os.path.join(out_dir, "roofline.png"))


def _plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    labels = [f"{r['n']}x{r['k']}" for r in results["rows"]]
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 0.2 for i in xs], [r["a16_speedup"] for r in results["rows"]],
           width=0.4, label="W1.58-A16")
    ax.bar([i + 0.2 for i in xs], [r["a8_speedup"] for r in results["rows"]],
           width=0.4, label="W1.58-A8")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="fp16 baseline")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("speedup vs fp16 F.linear (decode, M=1)")
    ax.set_title(f"1.58-bit GEMV speedup on {results['device']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--peak-bw", type=float, default=DEFAULT_PEAK_BW_GBPS)
    args = p.parse_args()
    run(args.out, args.peak_bw)
