"""High-level one-call API + CLI for the 1.58-bit inference engine.

Example (Python)::

    from bitnet.engine import BitNetEngine
    eng = BitNetEngine.from_pretrained("1bitLLM/bitnet_b1_58-large", activation_quant=True)
    print(eng.generate("The future of efficient AI inference is", max_new_tokens=64))

Example (CLI)::

    python -m bitnet.engine --model 1bitLLM/bitnet_b1_58-large --a8 \\
        --prompt "Hello" --max-new-tokens 64 --benchmark
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch

from .model_utils import (
    DEFAULT_MODEL_ID,
    DEFAULT_TARGETS,
    load_model,
    replace_linears,
    tokens_per_second,
)


@dataclass
class BitNetEngine:
    """Loads a causal-LM, swaps its projections for packed-ternary ``BitLinear``, and
    exposes ``generate`` / ``benchmark`` with the custom Triton kernels."""

    model: object
    tokenizer: object
    device: str = "cuda"
    num_replaced: int = 0
    scheme: str = "W1.58-A16"

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        activation_quant: bool = False,
        per_row: bool = False,
        targets=DEFAULT_TARGETS,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        swap: bool = True,
    ) -> "BitNetEngine":
        """Load ``model_id`` and (optionally) replace its Linear layers with BitLinear."""
        model, tok = load_model(model_id, dtype=dtype, device=device)
        n = 0
        if swap:
            n = replace_linears(
                model,
                targets=targets,
                per_row=per_row,
                quantize_activations=activation_quant,
            )
        return cls(
            model=model,
            tokenizer=tok,
            device=device,
            num_replaced=n,
            scheme="W1.58-A8" if activation_quant else "W1.58-A16",
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def benchmark(
        self, prompt: str = "The future of efficient AI inference is", max_new_tokens: int = 128
    ) -> float:
        """Return decode throughput (tokens/sec) for the current model."""
        return tokens_per_second(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
        )


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="1.58-bit inference engine")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--prompt", default="The future of efficient AI inference is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--a8", action="store_true", help="use W1.58-A8 (int8 activations)")
    p.add_argument("--per-row", action="store_true", help="per-output-row weight scale")
    p.add_argument("--sample", action="store_true", help="sample instead of greedy")
    p.add_argument("--benchmark", action="store_true", help="also print tokens/sec")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("The engine requires CUDA (run on a Kaggle T4).")

    eng = BitNetEngine.from_pretrained(
        args.model, activation_quant=args.a8, per_row=args.per_row
    )
    print(f"[{eng.scheme}] replaced {eng.num_replaced} Linear layers\n")
    text = eng.generate(
        args.prompt, max_new_tokens=args.max_new_tokens, do_sample=args.sample
    )
    print(text)
    if args.benchmark:
        tps = eng.benchmark(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
        print(f"\ndecode throughput: {tps:.2f} tok/s")


if __name__ == "__main__":
    main()
