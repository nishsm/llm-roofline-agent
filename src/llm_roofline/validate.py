"""Compare model predictions with measured decode throughput.

Rigor rule: fit ONE bandwidth-efficiency factor on batch-1 measurements and
report error on the other batch sizes, so the reported error is held out.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .hardware import Hardware
from .models import MODELS
from .roofline import decode


def compare(measured_path: str | Path, hw: Hardware, fit_on_batch: int = 1) -> dict:
    entries = json.loads(Path(measured_path).read_text())
    preds = []
    for e in entries:
        dtype = e.get("dtype", "fp16")
        r = decode(MODELS[e["model"]], hw, e["batch"], e["context"],
                   w_dtype=dtype, kv_dtype=e.get("kv_dtype", dtype))
        preds.append((e, r.tokens_per_s, r.bound))

    fit = [e["decode_tok_s"] / p for e, p, _ in preds if e["batch"] == fit_on_batch]
    eff = statistics.median(fit) if fit else 1.0

    def err(meas: float, pred: float) -> float:
        return abs(pred - meas) / meas * 100.0

    rows = []
    for e, p, bound in preds:
        meas = e["decode_tok_s"]
        rows.append({
            "model": e["model"], "batch": e["batch"], "context": e["context"],
            "measured_tok_s": round(meas, 1),
            "ideal_pred_tok_s": round(p, 1),
            "ideal_err_pct": round(err(meas, p), 1),
            "calibrated_pred_tok_s": round(p * eff, 1),
            "calibrated_err_pct": round(err(meas, p * eff), 1),
            "bound": bound,
            "held_out": e["batch"] != fit_on_batch,
        })
    held = [r["calibrated_err_pct"] for r in rows if r["held_out"]]
    return {
        "fitted_efficiency": round(eff, 3),
        "fit_on_batch": fit_on_batch,
        "mean_abs_err_pct_held_out": round(statistics.mean(held), 1) if held else None,
        "rows": rows,
    }
