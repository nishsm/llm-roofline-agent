"""Hardware profiles.

IMPORTANT: peak numbers are inputs, not facts this repo vouches for.

* ``h100_sxm`` uses NVIDIA's published dense FP16/BF16 peak and HBM3 bandwidth.
* ``rtx4060_laptop`` is an APPROXIMATE PLACEHOLDER. Laptop power limits change
  real clocks, so measure your own ceilings with ``bench/measure_peaks.py`` and
  pass the resulting JSON file via ``--hw``.
* ``sram_accelerator_estimate`` is a rough what-if built from third-party
  reporting on SRAM-heavy inference chips, not vendor data. Replace it with
  numbers you can source before drawing conclusions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops: float        # dense FLOP/s for the compute dtype, per device
    mem_bw: float            # bytes/s, per device
    mem_capacity: float      # bytes, per device
    num_devices: int = 1     # weights and KV assumed evenly sharded
    source: str = ""

    @property
    def total_flops(self) -> float:
        return self.peak_flops * self.num_devices

    @property
    def total_bw(self) -> float:
        return self.mem_bw * self.num_devices

    @property
    def total_capacity(self) -> float:
        return self.mem_capacity * self.num_devices

    @property
    def ridge_point(self) -> float:
        """FLOPs per byte where the roofline turns from memory- to compute-bound."""
        return self.total_flops / self.total_bw

    @classmethod
    def from_json(cls, path: str | Path) -> "Hardware":
        return cls(**json.loads(Path(path).read_text()))


PROFILES: dict[str, Hardware] = {
    h.name: h
    for h in [
        Hardware("h100_sxm", peak_flops=989e12, mem_bw=3.35e12, mem_capacity=80e9,
                 source="NVIDIA H100 SXM datasheet, dense FP16/BF16"),
        Hardware("rtx4060_laptop", peak_flops=15e12, mem_bw=256e9, mem_capacity=8e9,
                 source="APPROXIMATE PLACEHOLDER, measure with bench/measure_peaks.py"),
        Hardware("sram_accelerator_estimate", peak_flops=1.2e15, mem_bw=150e12,
                 mem_capacity=0.5e9, num_devices=1,
                 source="third-party estimate for an SRAM-heavy inference chip, not vendor data"),
    ]
}


def load_hardware(name_or_path: str) -> Hardware:
    if name_or_path in PROFILES:
        return PROFILES[name_or_path]
    path = Path(name_or_path)
    if path.exists():
        return Hardware.from_json(path)
    raise KeyError(f"unknown hardware {name_or_path!r}; known: {sorted(PROFILES)} or a JSON path")
