# NVFP4 Decode Optimization: VDR=8 + Unrolled Kernel

**Date:** 2026-07-15
**Build:** 6a954bd73 (b10035)
**GPU:** NVIDIA GeForce RTX 5090 (sm120 Blackwell, 32 GiB VRAM)
**CPU:** Intel Core Ultra 9 285K (24 P-cores), gcc-15.2

## Changes

- `ggml/src/ggml-cuda/vecdotq.cuh`: Added `vec_dot_nvfp4_q8_1_blackwell` — fully unrolled
  decode kernel with VDR=8 (up from VDR=4). Processes all 4 sub-blocks of each NVFP4
  block in straight-line code, eliminating thread-pair split overhead.
- `ggml/src/ggml-cuda/mmvq.cu`: Dispatch to Blackwell variant via `#if defined(BLACKWELL_MMA_AVAILABLE)`.

## Results (3 reps, fa=0)

| Model | Quant | pp128 (t/s) | tg128 (t/s) |
|---|---|---|---|
| Qwen3.6-35B-A3B MoE | NVFP4 **old** | 2869 | **184.41** |
| Qwen3.6-35B-A3B MoE | NVFP4 **new** | 2873 | **190.14** |
| Qwen3.6-35B-A3B MoE | Q4_K_S | 2392 | **258.64** |

**Decode improvement: +3.1%** (184.4 -> 190.1 t/s).

## Raw llama-bench

```
| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   0 |           pp128 |      2873.37 ± 34.49 |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   0 |           tg128 |        190.14 ± 0.09 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   0 |           pp128 |      2392.33 ± 80.70 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   0 |           tg128 |        258.64 ± 2.73 |
```

## Bug Fix

Initial version used `int4` (128-bit) vectorized loads from `block_nvfp4::qs`,
which starts at offset 4 within a 36-byte struct — only 4-byte aligned.
128-bit loads require 16-byte alignment on NVIDIA GPUs, causing a CUDA fault.
Fixed by using individual `int` (32-bit) loads with fully unrolled LUT + dp4a
computation.
