"""Measure achievable GPU ceilings (matmul FLOP/s and memory bandwidth) on THIS machine.

Writes a hardware JSON usable as ``--hw my_gpu.json``. Laptop GPUs throttle, so
run this on the same power plan you use for the decode benchmark.

    python bench/measure_peaks.py --name rtx4060_laptop_measured --out my_gpu.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def _bench(fn, iters: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="measured_gpu")
    ap.add_argument("--out", default="my_gpu.json")
    ap.add_argument("--n", type=int, default=4096, help="square matmul size")
    ap.add_argument("--mb", type=int, default=1024, help="buffer size for bandwidth test")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    dev = torch.device("cuda")
    x = torch.randn(a.n, a.n, device=dev, dtype=torch.float16)
    y = torch.randn(a.n, a.n, device=dev, dtype=torch.float16)
    t = _bench(lambda: x @ y, 50)
    flops = 2.0 * a.n ** 3 / t

    buf = torch.empty(a.mb * 1024 * 1024 // 2, device=dev, dtype=torch.float16)
    dst = torch.empty_like(buf)
    t = _bench(lambda: dst.copy_(buf), 50)
    bw = 2.0 * buf.numel() * 2 / t  # read + write

    props = torch.cuda.get_device_properties(0)
    hw = {
        "name": a.name,
        "peak_flops": flops,
        "mem_bw": bw,
        "mem_capacity": float(props.total_memory),
        "num_devices": 1,
        "source": f"measured on {props.name} with bench/measure_peaks.py",
    }
    with open(a.out, "w") as f:
        json.dump(hw, f, indent=2)
    print(json.dumps(hw, indent=2))


if __name__ == "__main__":
    main()
