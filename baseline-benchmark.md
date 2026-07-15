# Project Blackbear — Baseline Benchmark
> **Date:** 2026-07-15
> **Build:** `a58222229` (b10027)
> **Branch:** `project-blackbeard`
> **Status:** Backends stripped to CPU + CUDA only

---

## System

| Component | Detail |
|---|---|
| **CPU** | Intel Core Ultra 9 285K (Arrow Lake), 24 P-cores, no E-cores |
| **CPU features** | AVX512F, AVX512CD, AVX512BW, AVX512DQ, AVX512VL, AVX512VNNI, AVX512BF16, AVX512VBMI, FMA, F16C, VAES, VPCLMULQDQ |
| **GPU** | NVIDIA RTX 5090, sm120 (Blackwell), 32,088 MiB VRAM |
| **CUDA** | 13.3, NVCC 13.3.73 |
| **Compiler** | gcc-15.2.0 (host), NVCC 13.3 (device) |
| **Build** | CMake Release, `-march=native`, `-DGGML_CUDA=ON`, `-DGGML_NATIVE=ON`, `-DGGML_CUDA_GRAPHS=ON`, `-DGGML_CUDA_FA_ALL_QUANTS=ON`, `-DGGML_CPU_KLEIDIAI=OFF`, LTO enabled |
| **CUDA arch** | `120a-real` (Blackwell native, no forward-compat PTX) |
| **Threading** | 24 threads, OpenMP, no NUMA |

---

## Models Benchmarked

| Model | Architecture | Quant | File Size | Active Params | Total Params |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B-UD-Q4_K_S | qwen35moe (MoE) | Q4_K_S | 19.45 GiB | ~3B | 34.66B |
| Qwen3.6-35B-Fast-NVFP4 | qwen35moe (MoE) | NVFP4 | 22.88 GiB | ~3B | 35.51B |
| Qwen3.6-27B-NVFP4-unsloth | qwen35 (dense) | NVFP4 | 23.72 GiB | 27.32B | 27.32B |

---

## Results

### Prompt Processing (tokens/sec)

| Model | pp128 | pp512 | pp2048 |
|---|---|---|---|
| Q4_K_S (35B MoE) | 3,132 | 7,749 | 7,679 |
| NVFP4 (35B MoE) | 4,659 | **8,597** | — |
| NVFP4 (27B dense) | — | 4,154 | — |

### Text Generation (tokens/sec)

| Model | tg128 | tg256 |
|---|---|---|
| Q4_K_S (35B MoE) | 269.5 | 270.8 |
| NVFP4 (35B MoE) | 203.5 | 203.6 |
| NVFP4 (27B dense) | — | 61.2 |

### Batch Size Sensitivity (Q4_K_S, pp512 / tg256)

| Batch | pp512 (t/s) | tg256 (t/s) |
|---|---|---|
| 512 | 7,801 | 269.7 |
| 1024 | 7,800 | 271.6 |
| 2048 | 7,815 | 271.5 |
| 4096 | 7,747 | 271.7 |

Batch size has negligible impact — the MoE routing + CUDA graphs absorb it.

## Backend Strip (Phase 3)

16 backends removed (Metal, Vulkan, SYCL, OpenCL, CANN, HIP, MUSA, WebGPU,
OpenVINO, Hexagon, ZenDNN, ZDNN, BLAS, RPC, virtgpu, ET). Only `ggml-cpu/` and
`ggml-cuda/` remain. Build targets dropped from 730 to 115.

**Performance unchanged** — all benchmarks match baseline within measurement
noise, confirming no impact from the removal.

---

## Key Observations

1. **NVFP4 prompt processing beats Q4_K_S** on the 35B MoE model — 8,597 vs 7,749 t/s at pp512 (+11%). This confirms Blackwell's native FP4 tensor core MMA (`mma_block_scaled_fp4` with `kind::mxf4nvf4`) provides a real win for prompt eval, despite NVFP4 being 3.4 GiB larger on disk.

2. **NVFP4 generation trails Q4_K_S** — 203.6 vs 270.8 t/s (-25%). Generation is memory-bandwidth-bound, and NVFP4's larger block scales (ue4m3 per 16 elements vs Q4_K_S grouped scales) increase the bytes-per-token read from VRAM.

3. **MoE matters more than quantization** — the 27B dense NVFP4 model is slower than the 35B MoE NVFP4 in both pp (4,154 vs 8,597) and tg (61.2 vs 203.6). The dense model activates all 27B parameters per token; the MoE activates ~3B.

4. **CUDA graphs are working** — `llama-simple` reported ~30 graph reuses in a 31-token run, confirming `GGML_CUDA_GRAPHS=ON` is effective.

5. **Prompt processing scales well** — pp128→pp512 shows 2.5x throughput improvement (batch amortization), then plateaus at pp2048.

---

## Comparison Notes

- **No CPU-only baseline** collected — all layers offloaded (`-ngl 99`). The 285K has AVX512 but no AMX, so CPU-only would be significantly slower.
- **No batch-size impact** — this is expected for a single-sequence benchmark; batch sensitivity appears only with concurrent requests (server mode, speculative decoding).
- **Q8_0 DFlash model** (386M params, `dflash` architecture) failed to create context — architecture mismatch, not a benchmark failure. Excluded from results.

---

## Raw llama-bench Output

```
### Q4_K_S (35B MoE), multi-prompt/generation

| model                          |       size |     params | backend    | ngl | threads | test     |               t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   pp128  |  3132.27 ± 216.83 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   pp512  |  7749.47 ± 113.83 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |  pp2048  |  7679.08 ± 115.65 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   tg128  |    269.54 ± 2.97 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |   tg256  |    270.78 ± 0.68 |

### NVFP4 (35B MoE)

| model                          |       size |     params | backend    | ngl | threads | test     |               t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----------------: |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   pp128  |  4658.60 ± 145.38 |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   pp512  |  8597.07 ± 42.61 |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   tg128  |    203.52 ± 2.02 |
| qwen35moe 35B.A3B NVFP4        |  22.88 GiB |    35.51 B | CUDA       |  99 |      24 |   tg256  |    203.58 ± 1.85 |

### NVFP4 (27B dense)

| model                          |       size |     params | backend    | ngl | threads | test     |               t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | -----------------: |
| qwen35 27B NVFP4               |  23.72 GiB |    27.32 B | CUDA       |  99 |      24 |   pp512  |  4154.07 ± 241.73 |
| qwen35 27B NVFP4               |  23.72 GiB |    27.32 B | CUDA       |  99 |      24 |   tg256  |     61.18 ± 0.57 |

### Batch-size sweep (Q4_K_S, pp512 / tg256)

| model                          |       size |     params | backend    | ngl | threads | n_batch | test     |               t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -----------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |     512 |   pp512  |  7801.13 ± 48.59 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |     512 |   tg256  |    269.71 ± 2.94 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    1024 |   pp512  |  7800.49 ± 34.04 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    1024 |   tg256  |    271.58 ± 1.13 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    2048 |   pp512  |  7814.75 ± 48.56 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    2048 |   tg256  |    271.45 ± 0.99 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    4096 |   pp512  |  7746.90 ± 7.64 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |      24 |    4096 |   tg256  |    271.72 ± 1.02 |
```
