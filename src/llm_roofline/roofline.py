"""Analytical roofline model for decoder-only LLM inference.

Assumptions (all first-order, stated so they can be challenged):

* FLOPs count multiply-accumulates as 2 FLOPs. Norms, RoPE and softmax FLOPs
  are ignored; they are small next to the matmuls.
* Decode reads every weight once per step regardless of batch size, plus the
  full KV cache for each sequence in the batch.
* Attention is modeled as fused (FlashAttention style): the S x S score matrix
  never goes to DRAM, so only Q/K/V/KV-cache traffic is counted.
* Activations between layers are assumed to stay on chip and are ignored.
* ``time = max(flops / (peak_flops * mfu), bytes / (mem_bw * mbu))``.
  ``mfu`` and ``mbu`` are efficiency factors in (0, 1]; fitting them to real
  measurements is the point of ``llm-roofline compare``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import Hardware
from .models import ModelConfig

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5,
}


@dataclass(frozen=True)
class PhaseResult:
    phase: str
    batch: int
    seq: int
    flops: float
    bytes: float
    arithmetic_intensity: float
    ridge_point: float
    bound: str                # "memory" or "compute"
    t_compute_s: float
    t_memory_s: float
    time_s: float
    tokens_per_s: float
    mem_needed_bytes: float
    fits: bool


def kv_cache_bytes(model: ModelConfig, batch: int, seq: int, kv_dtype: str = "fp16") -> float:
    """K and V for every layer: 2 * layers * batch * seq * kv_heads * head_dim * bytes."""
    return 2.0 * model.layers * batch * seq * model.kv_dim * DTYPE_BYTES[kv_dtype]


def _finish(phase: str, hw: Hardware, model: ModelConfig, batch: int, seq: int,
            flops: float, nbytes: float, tokens: float, w_dtype: str, kv_dtype: str,
            kv_seq: int, mfu: float, mbu: float) -> PhaseResult:
    t_c = flops / (hw.total_flops * mfu)
    t_m = nbytes / (hw.total_bw * mbu)
    time_s = max(t_c, t_m)
    ai = flops / nbytes
    mem_needed = model.total_params * DTYPE_BYTES[w_dtype] + kv_cache_bytes(model, batch, kv_seq, kv_dtype)
    return PhaseResult(
        phase=phase, batch=batch, seq=seq, flops=flops, bytes=nbytes,
        arithmetic_intensity=ai, ridge_point=hw.ridge_point,
        bound="compute" if t_c >= t_m else "memory",
        t_compute_s=t_c, t_memory_s=t_m, time_s=time_s,
        tokens_per_s=tokens / time_s, mem_needed_bytes=mem_needed,
        fits=mem_needed <= hw.total_capacity,
    )


def decode(model: ModelConfig, hw: Hardware, batch: int, context: int,
           w_dtype: str = "fp16", kv_dtype: str = "fp16",
           mfu: float = 1.0, mbu: float = 1.0) -> PhaseResult:
    """One decode step: every sequence in the batch emits one token at ``context``."""
    wb = DTYPE_BYTES[w_dtype]
    attn_flops = model.layers * 4.0 * batch * model.heads * model.head_dim * context
    flops = 2.0 * batch * model.matmul_params + attn_flops
    nbytes = (model.matmul_params * wb
              + kv_cache_bytes(model, batch, context, kv_dtype)   # read history
              + kv_cache_bytes(model, batch, 1, kv_dtype))        # write new token
    return _finish("decode", hw, model, batch, context, flops, nbytes,
                   tokens=batch, w_dtype=w_dtype, kv_dtype=kv_dtype,
                   kv_seq=context, mfu=mfu, mbu=mbu)


def prefill(model: ModelConfig, hw: Hardware, batch: int, prompt_len: int,
            w_dtype: str = "fp16", kv_dtype: str = "fp16", causal: bool = True,
            mfu: float = 1.0, mbu: float = 1.0) -> PhaseResult:
    """Process ``prompt_len`` tokens per sequence. LM head runs on the last token only."""
    wb = DTYPE_BYTES[w_dtype]
    causal_factor = 0.5 if causal else 1.0
    layer_flops = 2.0 * batch * prompt_len * model.layer_params
    head_flops = 2.0 * batch * model.lm_head_params
    attn_flops = (model.layers * 4.0 * batch * model.heads * model.head_dim
                  * prompt_len ** 2 * causal_factor)
    flops = layer_flops + head_flops + attn_flops
    nbytes = (model.matmul_params * wb
              + kv_cache_bytes(model, batch, prompt_len, kv_dtype))   # write KV
    return _finish("prefill", hw, model, batch, prompt_len, flops, nbytes,
                   tokens=batch * prompt_len, w_dtype=w_dtype, kv_dtype=kv_dtype,
                   kv_seq=prompt_len, mfu=mfu, mbu=mbu)


def decode_hetero(model: ModelConfig, attn_hw: Hardware, ffn_hw: Hardware,
                  batch: int, context: int, link_bw: float,
                  link_latency_s: float = 5e-6, w_dtype: str = "fp16",
                  kv_dtype: str = "fp16", act_dtype: str = "fp16",
                  overlap: bool = True) -> dict:
    """First-order what-if: attention + KV cache on ``attn_hw``, FFN on ``ffn_hw``.

    Per layer, activations cross the link twice (to the FFN device and back).
    With ``overlap`` the step time is the slowest of the three stages (an ideal
    ping-pong pipeline); without it the stages add. This is a toy model for
    exploring where the bottleneck moves, not a prediction of any real system.
    """
    wb = DTYPE_BYTES[w_dtype]
    attn_weight_params = model.layers * model.attn_params_per_layer + model.lm_head_params
    ffn_weight_params = model.layers * model.mlp_params_per_layer

    attn_flops = (2.0 * batch * attn_weight_params
                  + model.layers * 4.0 * batch * model.heads * model.head_dim * context)
    attn_bytes = (attn_weight_params * wb
                  + kv_cache_bytes(model, batch, context, kv_dtype)
                  + kv_cache_bytes(model, batch, 1, kv_dtype))
    ffn_flops = 2.0 * batch * ffn_weight_params
    ffn_bytes = ffn_weight_params * wb

    t_attn = max(attn_flops / attn_hw.total_flops, attn_bytes / attn_hw.total_bw)
    t_ffn = max(ffn_flops / ffn_hw.total_flops, ffn_bytes / ffn_hw.total_bw)
    comm_bytes = 2.0 * model.layers * batch * model.hidden * DTYPE_BYTES[act_dtype]
    t_comm = comm_bytes / link_bw + 2.0 * model.layers * link_latency_s

    step = max(t_attn, t_ffn, t_comm) if overlap else t_attn + t_ffn + t_comm
    stages = {"attention": t_attn, "ffn": t_ffn, "link": t_comm}
    ffn_fits = ffn_weight_params * wb <= ffn_hw.total_capacity
    return {
        "t_attention_s": t_attn, "t_ffn_s": t_ffn, "t_link_s": t_comm,
        "step_time_s": step, "tokens_per_s": batch / step,
        "bottleneck": max(stages, key=stages.get),
        "ffn_weights_fit_on_ffn_device": ffn_fits,
    }
