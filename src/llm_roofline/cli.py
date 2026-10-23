"""Command line interface: sweep, hetero, compare, plot, list."""

from __future__ import annotations

import argparse
import json
import sys

from .hardware import PROFILES, load_hardware
from .models import MODELS
from .roofline import decode_hetero
from .sweep import sweep, to_csv, to_markdown
from .validate import compare


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llm-roofline", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list built-in models and hardware profiles")

    sp = sub.add_parser("sweep", help="sweep batch size and context across models and hardware")
    sp.add_argument("--model", nargs="+", default=["llama-3.2-3b"], choices=sorted(MODELS))
    sp.add_argument("--hw", nargs="+", default=["h100_sxm"], help="profile name or JSON path")
    sp.add_argument("--batch", type=_ints, default=[1, 4, 16, 64])
    sp.add_argument("--context", type=_ints, default=[512, 2048, 8192])
    sp.add_argument("--dtype", default="fp16")
    sp.add_argument("--kv-dtype", default=None)
    sp.add_argument("--mfu", type=float, default=1.0)
    sp.add_argument("--mbu", type=float, default=1.0)
    sp.add_argument("--csv", action="store_true")

    hp = sub.add_parser("hetero", help="what-if: attention on one device, FFN on another")
    hp.add_argument("--model", default="llama-3.1-8b", choices=sorted(MODELS))
    hp.add_argument("--attn-hw", default="h100_sxm")
    hp.add_argument("--ffn-hw", default="sram_accelerator_estimate")
    hp.add_argument("--ffn-devices", type=int, default=1, help="number of FFN devices")
    hp.add_argument("--batch", type=int, default=8)
    hp.add_argument("--context", type=int, default=4096)
    hp.add_argument("--link-gbps", type=float, default=400.0, help="link bandwidth, GB/s")
    hp.add_argument("--no-overlap", action="store_true")

    cp = sub.add_parser("compare", help="compare predictions with measured decode throughput")
    cp.add_argument("measured", help="JSON list from bench/measure_decode.py")
    cp.add_argument("--hw", required=True)
    cp.add_argument("--fit-on-batch", type=int, default=1)

    pp = sub.add_parser("plot", help="save a roofline plot (needs matplotlib)")
    pp.add_argument("--model", default="llama-3.2-3b", choices=sorted(MODELS))
    pp.add_argument("--hw", default="h100_sxm")
    pp.add_argument("--out", default="roofline.png")

    a = ap.parse_args(argv)

    if a.cmd == "list":
        print("models:  ", ", ".join(sorted(MODELS)))
        print("hardware:", ", ".join(sorted(PROFILES)))
        return 0

    if a.cmd == "sweep":
        rows = sweep([MODELS[m] for m in a.model], [load_hardware(h) for h in a.hw],
                     a.batch, a.context, a.dtype, a.kv_dtype or a.dtype, a.mfu, a.mbu)
        print(to_csv(rows) if a.csv else to_markdown(rows))
        return 0

    if a.cmd == "hetero":
        from dataclasses import replace

        ffn = replace(load_hardware(a.ffn_hw), num_devices=a.ffn_devices)
        res = decode_hetero(MODELS[a.model], load_hardware(a.attn_hw), ffn, a.batch,
                            a.context, a.link_gbps * 1e9, overlap=not a.no_overlap)
        print(json.dumps(res, indent=2))
        return 0

    if a.cmd == "compare":
        print(json.dumps(compare(a.measured, load_hardware(a.hw), a.fit_on_batch), indent=2))
        return 0

    if a.cmd == "plot":
        from .plot import roofline_plot

        roofline_plot(MODELS[a.model], load_hardware(a.hw), a.out)
        print(f"wrote {a.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
