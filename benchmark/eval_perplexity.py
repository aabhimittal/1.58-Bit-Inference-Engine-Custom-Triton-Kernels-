"""Perplexity check: does ternary (+ optional int8-activation) quantization hold up?

Compares the perplexity of the stock fp16 model against the packed-ternary BitLinear
model on a small text sample. A modest perplexity increase (not a blow-up) is the sign
that the quantization + custom kernel are numerically sound end to end.

Run on a GPU (Kaggle T4)::

    python benchmark/eval_perplexity.py --model 1bitLLM/bitnet_b1_58-large --a8
"""

from __future__ import annotations

import argparse

import torch

from bitnet.model_utils import DEFAULT_MODEL_ID, load_model, replace_linears

SAMPLE_TEXT = (
    "Artificial intelligence has progressed rapidly over the past decade. "
    "Large language models can now write code, summarize documents, and answer "
    "questions across many domains. Making these models cheaper to run is one of "
    "the central engineering challenges of the field, and low-bit quantization is a "
    "promising path toward that goal."
)


@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str = "cuda") -> float:
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    out = model(input_ids, labels=input_ids)
    return float(torch.exp(out.loss).item())


def run(model_id: str, a8: bool, text: str):
    if not torch.cuda.is_available():
        raise SystemExit("This eval requires CUDA (run on a Kaggle T4).")

    model, tok = load_model(model_id)
    base_ppl = perplexity(model, tok, text)
    print(f"baseline (fp16)          perplexity: {base_ppl:8.3f}")

    scheme = "W1.58-A8" if a8 else "W1.58-A16"
    n = replace_linears(model, quantize_activations=a8)
    bit_ppl = perplexity(model, tok, text)
    print(f"{scheme} ({n} layers)  perplexity: {bit_ppl:8.3f}")
    print(f"\nperplexity ratio (bit / fp16): {bit_ppl / base_ppl:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--a8", action="store_true", help="use W1.58-A8 (int8 activations)")
    p.add_argument("--text", default=SAMPLE_TEXT)
    args = p.parse_args()
    run(args.model, args.a8, args.text)
