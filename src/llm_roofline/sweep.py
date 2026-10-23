"""Design-space sweeps over model, hardware, batch size, context and precision."""

from __future__ import annotations

import csv
import io
import itertools

from .hardware import Hardware
from .models import ModelConfig
from .roofline import decode, prefill


def sweep(models: list[ModelConfig], hws: list[Hardware], batches: list[int],
          contexts: list[int], w_dtype: str = "fp16", kv_dtype: str = "fp16",
          mfu: float = 1.0, mbu: float = 1.0) -> list[dict]:
    rows: list[dict] = []
    for m, hw, b, c in itertools.product(models, hws, batches, contexts):
        d = decode(m, hw, b, c, w_dtype, kv_dtype, mfu, mbu)
        p = prefill(m, hw, b, c, w_dtype, kv_dtype, mfu=mfu, mbu=mbu)
        rows.append({
            "model": m.name, "hardware": hw.name, "batch": b, "context": c,
            "w_dtype": w_dtype,
            "decode_ai": round(d.arithmetic_intensity, 2),
            "ridge": round(d.ridge_point, 1),
            "decode_bound": d.bound,
            "decode_tok_s": round(d.tokens_per_s, 1),
            "prefill_bound": p.bound,
            "prefill_ms": round(p.time_s * 1e3, 2),
            "mem_gb": round(d.mem_needed_bytes / 1e9, 2),
            "fits": d.fits,
        })
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue()
