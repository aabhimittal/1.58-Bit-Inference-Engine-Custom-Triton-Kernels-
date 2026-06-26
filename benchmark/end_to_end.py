"""End-to-end tokens/sec: a real BitNet model with fp16 Linear vs packed-ternary BitLinear.

Run on a GPU (Kaggle T4)::

    python benchmark/end_to_end.py --model 1bitLLM/bitnet_b1_58-large --tokens 128

Loads the model twice (or reuses + re-swaps), measures greedy-decode throughput for the
stock fp16 layers vs the custom Triton kernel, and prints tokens/sec for both.
"""

from __future__ import annotations

import argparse

import torch

from bitnet.model_utils import (
    DEFAULT_MODEL_ID,
    load_model,
    replace_linears,
    tokens_per_second,
)


def run(model_id: str, tokens: int, prompt: str):
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA (run on a Kaggle T4).")

    print(f"Loading baseline model: {model_id}")
    model, tok = load_model(model_id)
    base_tps = tokens_per_second(model, tok, prompt=prompt, max_new_tokens=tokens)
    print(f"  baseline (fp16 Linear): {base_tps:6.2f} tok/s")

    n = replace_linears(model)
    print(f"Swapped {n} Linear layers -> packed-ternary BitLinear (Triton kernel)")
    bit_tps = tokens_per_second(model, tok, prompt=prompt, max_new_tokens=tokens)
    print(f"  custom  (1.58-bit kernel): {bit_tps:6.2f} tok/s")

    print(f"\nSpeedup: {bit_tps / base_tps:.2f}x   (decode, batch=1)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompt", default="The future of efficient AI inference is")
    args = p.parse_args()
    run(args.model, args.tokens, args.prompt)
