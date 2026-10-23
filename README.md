# llm-roofline-agent

First-principles **roofline performance modeling for LLM inference**: given a model, a hardware profile, batch size, context length and precision, predict whether prefill and decode are compute-bound or memory-bound and how fast they run, then **check the prediction against real GPU measurements**.

The goal is a performance model you can trust because its error is measured, not assumed, and a place to explore design-space questions such as "what moves if decode attention and FFN run on different hardware?"

## Status

| Phase | What | State |
|---|---|---|
| 1 | Analytical model, hardware profiles, sweep / hetero / compare / plot CLI, unit tests | done (14 tests pass) |
| 2 | Measure real prefill/decode on an RTX 4060 laptop GPU, report held-out model error | **not run yet** (scripts in `bench/`) |
| 3 | LangGraph agent that runs sweeps, flags model-vs-measurement disagreement, proposes follow-up experiments and writes a report | planned |

No measured results are published here yet. Numbers will be added to this README only after Phase 2 is actually run.

## The model

For each phase the model counts FLOPs and DRAM bytes, then

```
arithmetic_intensity = FLOPs / bytes
ridge_point          = peak_FLOPs / peak_bandwidth
time                 = max(FLOPs / (peak_FLOPs * mfu), bytes / (bandwidth * mbu))
```

* **Decode** reads every weight once per step plus the KV cache of every sequence, so at small batch its intensity is about 1 FLOP/byte, far below any GPU's ridge point: it is memory-bound. Batching raises intensity; lower-precision weights cut bytes.
* **Prefill** processes many tokens per weight read, so it is compute-bound at realistic prompt lengths.
* **Heterogeneous what-if** (`hetero`): attention + KV cache on one device, FFN on another, activations crossing a link twice per layer, with optional ideal overlap. A toy model for locating bottlenecks, not a prediction of any real system.

Assumptions are listed in `src/llm_roofline/roofline.py` (fused attention, activations stay on chip, norms/RoPE ignored).

## Hardware profiles

`h100_sxm` uses NVIDIA's published dense FP16 peak and memory bandwidth. `rtx4060_laptop` is an **approximate placeholder**: run `bench/measure_peaks.py` to measure your own ceilings and pass the JSON via `--hw`. `sram_accelerator_estimate` is a rough what-if from third-party reporting, **not vendor data**; replace it with sourced numbers before drawing conclusions.

## Usage

```bash
pip install -e '.[dev,plot]'
pytest

llm-roofline list
llm-roofline sweep --model llama-3.1-8b --hw h100_sxm --batch 1,32,256 --context 2048
llm-roofline hetero --batch 32 --context 4096 --ffn-devices 8 --link-gbps 400
llm-roofline plot --model llama-3.2-3b --hw h100_sxm --out roofline.png
```

## Validating against your own GPU (Phase 2)

```bash
pip install -e '.[bench]'
python bench/measure_peaks.py --name rtx4060_laptop_measured --out my_gpu.json
python bench/measure_decode.py --hf-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --model tinyllama-1.1b --batches 1 2 4 8 --context 512 --out measured.json
llm-roofline compare measured.json --hw my_gpu.json
```

`compare` fits a single efficiency factor on batch-1 measurements and reports error on the **held-out** batch sizes, so the headline error is not fit to itself. Report ideal and calibrated error side by side.

## Limitations

* Single-GPU, dense decoder-only transformers with SwiGLU MLPs. No MoE, no tensor/pipeline parallel communication model, no speculative decoding yet.
* Efficiency is one scalar; real kernels vary with shape. Expect the largest error at small batch and long context.
* HF `transformers` eager decode is not an optimized runtime; it measures a baseline, not a ceiling.

## Related work

Microsoft's [Vidur](https://github.com/microsoft/vidur) is a far more complete LLM inference simulator (operator profiling plus predictive models). This repo is a small, transparent, first-principles companion, not a replacement.

## License

MIT
