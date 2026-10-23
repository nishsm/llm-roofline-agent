import json

import pytest

from llm_roofline import MODELS, PROFILES, decode, decode_hetero, prefill
from llm_roofline.roofline import kv_cache_bytes
from llm_roofline.validate import compare

H100 = PROFILES["h100_sxm"]
GB = 1e9


def test_h100_ridge_point_matches_published_ratio():
    # 989 TFLOPS / 3.35 TB/s is about 295 FLOPs per byte
    assert H100.ridge_point == pytest.approx(295.2, rel=0.01)


@pytest.mark.parametrize("name,expected", [
    ("llama-3.1-8b", 8.03e9), ("llama-3.2-3b", 3.21e9),
    ("llama-3.2-1b", 1.24e9), ("tinyllama-1.1b", 1.10e9),
])
def test_param_counts_match_published_sizes(name, expected):
    assert MODELS[name].total_params == pytest.approx(expected, rel=0.02)


def test_kv_cache_matches_hand_calculation():
    # 2 * 32 layers * 8192 tokens * 8 kv heads * 128 dim * 2 bytes = 1 GiB
    got = kv_cache_bytes(MODELS["llama-3.1-8b"], 1, 8192, "fp16")
    assert got == 2 * 32 * 8192 * 8 * 128 * 2 == 1_073_741_824


def test_batch1_decode_is_memory_bound_with_intensity_near_one():
    r = decode(MODELS["llama-3.1-8b"], H100, batch=1, context=512)
    assert r.bound == "memory"
    assert 0.9 < r.arithmetic_intensity < 1.5
    assert r.arithmetic_intensity < r.ridge_point


def test_long_prefill_is_compute_bound():
    r = prefill(MODELS["llama-3.1-8b"], H100, batch=1, prompt_len=2048)
    assert r.bound == "compute"


def test_batching_raises_decode_throughput_while_memory_bound():
    m = MODELS["llama-3.2-3b"]
    t = [decode(m, H100, b, 512).tokens_per_s for b in (1, 4, 16)]
    assert t[0] < t[1] < t[2]


def test_decode_time_is_weight_bytes_over_bandwidth_at_batch1_short_context():
    m = MODELS["llama-3.1-8b"]
    r = decode(m, H100, 1, 1)
    expected = m.matmul_params * 2 / H100.total_bw
    assert r.time_s == pytest.approx(expected, rel=0.01)


def test_int4_weights_speed_up_memory_bound_decode_but_not_prefill():
    m = MODELS["llama-3.1-8b"]
    d16 = decode(m, H100, 1, 512, "fp16").tokens_per_s
    d4 = decode(m, H100, 1, 512, "int4").tokens_per_s
    assert d4 > 3.0 * d16
    assert prefill(m, H100, 1, 4096, "int4").bound == "compute"


def test_memory_fit_flag():
    small_gpu = PROFILES["rtx4060_laptop"]
    assert not decode(MODELS["llama-3.1-8b"], small_gpu, 1, 512).fits
    assert decode(MODELS["llama-3.2-1b"], small_gpu, 1, 512).fits


def test_hetero_overlap_never_slower_than_serial():
    m = MODELS["llama-3.1-8b"]
    ffn = PROFILES["sram_accelerator_estimate"]
    kw = dict(batch=8, context=4096, link_bw=400 * GB)
    a = decode_hetero(m, H100, ffn, overlap=True, **kw)
    b = decode_hetero(m, H100, ffn, overlap=False, **kw)
    assert a["step_time_s"] <= b["step_time_s"]
    assert a["bottleneck"] in {"attention", "ffn", "link"}


def test_compare_holds_out_batches_and_recovers_known_efficiency(tmp_path):
    """Synthetic measurements generated at exactly 70% of the ideal prediction."""
    hw = PROFILES["rtx4060_laptop"]
    m = MODELS["tinyllama-1.1b"]
    entries = []
    for b in (1, 2, 4, 8):
        ideal = decode(m, hw, b, 512).tokens_per_s
        entries.append({"model": m.name, "batch": b, "context": 512,
                        "decode_tok_s": 0.7 * ideal})
    p = tmp_path / "m.json"
    p.write_text(json.dumps(entries))
    out = compare(p, hw, fit_on_batch=1)
    assert out["fitted_efficiency"] == pytest.approx(0.7, rel=1e-6)
    assert out["mean_abs_err_pct_held_out"] < 0.5
