# The "1.58-Bit" Inference Engine — Custom Triton Kernels

A from-scratch demonstration that **BitNet b1.58 ternary weights `{-1, 0, 1}`** can be
packed into `int8` containers and multiplied with a **custom Triton kernel** that beats
stock PyTorch on a free Kaggle **T4** GPU — by attacking the real bottleneck of
autoregressive decoding: **memory bandwidth**.

## The idea (and an honest framing)

BitNet b1.58 stores every weight as one of three values `{-1, 0, 1}` ("1.58 bit" =
`log₂3`) plus a single floating-point scale `γ`. Multiplying by such a weight is
trivial: `+1 →` add the activation, `−1 →` subtract it, `0 →` skip — **no multiplier
needed** ("addition-only multiplication").

A T4 has **no native 2-bit math**, so we can't get free FLOPs from the format. What we
*can* get is **8× less weight traffic**: each weight moves as 2 bits instead of fp16's
16. During decode (batch size 1) the matmul is a memory-bound GEMV dominated by reading
the weight matrix, so cutting weight bytes 8× directly raises tokens/sec. That is the
thesis of this project, and the benchmarks report it honestly — including the cases
(large-batch prefill GEMM) where cuBLAS tensor cores still win.

## How it works

1. **Quantize** weights to ternary with BitNet's *absmean* rule — `bitnet/quantize.py`
   (`γ = mean(|W|)`, `W_tern = round(W/γ).clamp(-1,1)`).
2. **Pack** four ternary codes into one `uint8` (2 bits each, `code = value + 1`),
   packed along `K` → `[N, K/4]` — `bitnet/quantize.py`.
3. **Multiply** with a custom Triton kernel — `kernels/bitnet_kernel.py`:
   - *Decode path (small M):* unpack codes in registers and accumulate with
     `tl.where` add/subtract — the literal addition-only kernel, optimized for
     weight bandwidth.
   - *Prefill path (large M):* unpack into an fp16 operand and use `tl.dot`
     (tensor cores) for throughput.
4. **Drop-in layer** `BitLinear` (`bitnet/bitlinear.py`) wraps the kernel with a
   `from_linear()` constructor and a correct dequantize+`F.linear` fallback for
   CPU / no-Triton.
5. **Integrate & benchmark** against a real BitNet model — `bitnet/model_utils.py`,
   `benchmark/`.

## Repository layout

```
bitnet/quantize.py        absmean ternary quant + 2-bit pack/unpack
bitnet/bitlinear.py       BitLinear: packed-ternary nn.Linear replacement
bitnet/model_utils.py     load BitNet, swap Linear->BitLinear, tokens/sec timer
kernels/bitnet_kernel.py  Triton GEMV (add-only) + GEMM (tl.dot) kernels
benchmark/microbench.py   synthetic kernel vs fp16 F.linear
benchmark/end_to_end.py   real model tokens/sec: fp16 vs custom kernel
tests/test_pack.py        pack/unpack + quant reference (CPU, runs anywhere)
tests/test_kernel.py      Triton vs reference correctness (GPU only)
notebooks/bitnet_158_triton.ipynb   the Kaggle walkthrough
```

## Run it on Kaggle (T4)

1. New Notebook → Settings → Accelerator: **GPU T4 ×2**, Internet: **On**.
2. Add this repo (clone in a cell or upload as a Dataset).
3. Open `notebooks/bitnet_158_triton.ipynb` and Run All, or from a cell:

```python
!pip install -q transformers accelerate            # torch + triton preinstalled
!python benchmark/microbench.py                    # synthetic kernel benchmark
!python benchmark/end_to_end.py --model 1bitLLM/bitnet_b1_58-large
```

Swap `--model microsoft/bitnet-b1.58-2B-4T` for the heavier official checkpoint.

## Tests

```bash
pytest tests/test_pack.py     # CPU: pack/unpack + quant math (runs anywhere)
pytest tests/test_kernel.py   # GPU: Triton kernel vs reference (needs CUDA+Triton)
```

## Results (fill in from your T4 run)

Synthetic GEMV (`microbench.py`), decode path `M=1`:

| Shape (N×K) | fp16 ms | 1.58-bit ms | speedup | bit GB/s | max err |
|-------------|--------:|------------:|--------:|---------:|--------:|
| 4096×4096   |   _TBD_ |       _TBD_ |   _TBD_ |    _TBD_ |   _TBD_ |
| 4096×11008  |   _TBD_ |       _TBD_ |   _TBD_ |    _TBD_ |   _TBD_ |

End-to-end (`end_to_end.py`), greedy decode, batch=1:

| Model | baseline tok/s | 1.58-bit tok/s | speedup |
|-------|---------------:|---------------:|--------:|
| 1bitLLM/bitnet_b1_58-large | _TBD_ | _TBD_ | _TBD_ |

> Note: large-batch (prefill) GEMM may not beat cuBLAS on a T4 — expected, since the
> win here is decode-time bandwidth, not raw FLOPs.

## References

- BitNet b1.58: *The Era of 1-bit LLMs* (Ma et al., 2024).
- Microsoft BitNet models on the Hugging Face Hub (`microsoft/bitnet-b1.58-2B-4T`),
  and the `1bitLLM` paper reproductions.
- OpenAI Triton.
