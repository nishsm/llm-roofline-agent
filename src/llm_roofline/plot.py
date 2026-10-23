"""Roofline plot. Needs matplotlib (pip install 'llm-roofline-agent[plot]')."""

from __future__ import annotations

from .hardware import Hardware
from .models import ModelConfig
from .roofline import decode, prefill


def roofline_plot(model: ModelConfig, hw: Hardware, path: str,
                  batches=(1, 4, 16, 64), context: int = 2048, w_dtype: str = "fp16") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [10 ** (i / 20) for i in range(-20, 61)]
    roof = [min(hw.total_flops, x * hw.total_bw) for x in xs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(xs, roof, color="black", label=f"{hw.name} roofline")
    for b in batches:
        d = decode(model, hw, b, context, w_dtype)
        ax.scatter(d.arithmetic_intensity, d.flops / d.time_s, label=f"decode B={b}")
    p = prefill(model, hw, 1, context, w_dtype)
    ax.scatter(p.arithmetic_intensity, p.flops / p.time_s, marker="s", label="prefill B=1")
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("attainable FLOP/s")
    ax.set_title(f"{model.name} on {hw.name}, context {context}, {w_dtype}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
