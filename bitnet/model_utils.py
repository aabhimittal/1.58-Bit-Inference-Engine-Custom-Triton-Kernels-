"""Helpers to load a BitNet model and swap its ``nn.Linear`` layers for ``BitLinear``."""

from __future__ import annotations

import time
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from .bitlinear import BitLinear

# Default to the small reproduction of the BitNet b1.58 paper (~0.7B), which fits a free
# T4 comfortably. Switch to "microsoft/bitnet-b1.58-2B-4T" for the heavier official run.
DEFAULT_MODEL_ID = "1bitLLM/bitnet_b1_58-large"

# Linear sub-layers worth replacing (the large projections); lm_head/embeddings are left
# in full precision for output quality.
DEFAULT_TARGETS: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Load a causal-LM and its tokenizer from the Hugging Face Hub.

    Returns ``(model, tokenizer)``. Requires ``transformers`` (and usually
    ``trust_remote_code=True`` for BitNet checkpoints that ship custom modeling code).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device).eval()
    return model, tokenizer


def _iter_named_linears(
    module: nn.Module, targets: Iterable[str]
) -> Iterable[Tuple[nn.Module, str, nn.Linear]]:
    targets = tuple(targets)
    for parent_name, parent in module.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear) and child_name in targets:
                yield parent, child_name, child


def replace_linears(
    model: nn.Module,
    targets: Iterable[str] = DEFAULT_TARGETS,
    per_row: bool = False,
) -> int:
    """In-place swap every targeted ``nn.Linear`` for a packed-ternary ``BitLinear``.

    Returns the number of layers replaced.
    """
    count = 0
    for parent, child_name, linear in list(_iter_named_linears(model, targets)):
        bit_layer = BitLinear.from_linear(linear, per_row=per_row)
        bit_layer.to(linear.weight.device)
        setattr(parent, child_name, bit_layer)
        count += 1
    return count


@torch.no_grad()
def tokens_per_second(
    model: nn.Module,
    tokenizer,
    prompt: str = "The future of efficient AI inference is",
    max_new_tokens: int = 128,
    warmup: int = 1,
    device: str = "cuda",
) -> float:
    """Greedy-decode ``max_new_tokens`` and return tokens/second (decode throughput)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = out.shape[1] - inputs["input_ids"].shape[1]
    return generated / elapsed
