"""Measure real decode throughput with Hugging Face transformers.

Output is a JSON list consumed by ``llm-roofline compare``. Each entry:
{"model", "batch", "context", "dtype", "decode_tok_s", "ttft_ms", "hf_id"}.

    python bench/measure_decode.py --hf-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --model tinyllama-1.1b --batches 1 2 4 8 --context 512 --out measured.json

Method: prefill ``context`` tokens, then time N greedy decode steps with the KV
cache. Warmup is discarded and the median of repeats is reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def measure(model, batch: int, context: int, steps: int, repeats: int) -> tuple[float, float]:
    dev = next(model.parameters()).device
    vocab = model.config.vocab_size
    ids = torch.randint(100, vocab - 100, (batch, context), device=dev)
    tok_s, ttft = [], []
    for r in range(repeats + 1):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True)
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            t0 = time.perf_counter()
            for _ in range(steps):
                out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            t_dec = time.perf_counter() - t0
        if r == 0:
            continue  # warmup
        tok_s.append(batch * steps / t_dec)
        ttft.append(t_prefill * 1e3)
    return statistics.median(tok_s), statistics.median(ttft)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model", required=True, help="matching key in llm_roofline.MODELS")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DT))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="measured.json")
    a = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(a.hf_id, torch_dtype=DT[a.dtype]).cuda().eval()
    results = []
    for b in a.batches:
        tok_s, ttft = measure(model, b, a.context, a.steps, a.repeats)
        results.append({"model": a.model, "hf_id": a.hf_id, "batch": b, "context": a.context,
                        "dtype": a.dtype, "decode_tok_s": tok_s, "ttft_ms": ttft})
        print(f"batch={b:3d}  decode={tok_s:8.1f} tok/s  prefill={ttft:7.1f} ms")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
