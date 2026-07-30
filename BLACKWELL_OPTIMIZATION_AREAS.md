# Blackwell Optimization Areas — Project Blackbeard

> Updated: 2026-07-29 | GPU: RTX 5090 (SM120, 32 GiB VRAM)

---

## Session Summary (2026-07-29)

### Commits

| Commit | What | Impact |
|---|---|---|
| `165bdbd5e` | Document nwarps=2 infeasibility | — |
| `a2deb38b4` | Fix NVFP4 iter_k=1024 crash | Restored NVFP4 prefill |
| `ea6b19c51` | VDR=4 Q4_K decode (Blackwell) | Neutral (368 t/s) |
| `fe1aa1a1c` | Blackwell MMVQ param table | Neutral, infra |
| `a3da73bf0` | iter_k=8192 ceiling | pp128 +8% |
| `7f15a43bf` | iter_k=4096 + expand 2048 | pp512 +1.7% |
| `afd2d89c1` | NVFP4 1024 + Q4_K 2048 | pp512 +9.2%, **NVFP4 crash** |
| `5ee1d1308` | Prefill saturation docs | — |
| `2d0b18e71` | VEC int4 loads q5_0/q5_1 | Correctness |
| `72f84c9ec` | MMQ_ITER_K_BB + expand 1024 | 7 types at 1024 |
| `43b7c8e83` | Planning doc v1 | — |
| `e360d7aa4` | iter_k=1024 + 19 types + FA | +88% pp512 |

### Final Benchmarks (RTX 5090, -ngl 99, -t 24)

**Q4_K_XL 30B MoE (16.45 GiB):**

| Test | Ampere Baseline | After All Optimizations | Delta |
|---|---:|---:|---|
| pp128 | 4,429 t/s | 10,808 t/s | **+144%** |
| pp512 | 11,176 t/s | 23,282 t/s | **+108%** |
| pp1024 | — | 22,753 t/s | — |
| tg128 | 365 t/s | 366 t/s | ~0% |
| tg256 | 367 t/s | 367 t/s | ~0% |

**NVFP4 35B MoE (22.88 GiB):**

| Test | Result |
|---|---:|
| pp64 | 3,643 t/s |
| pp128 | 4,973 t/s |
| pp512 | 10,454 t/s |
| pp1024 | 10,319 t/s |
| tg128 | 207 t/s |
| tg256 | 209 t/s |

### iter_k Scaling: Full Journey (Q4_K_XL, pp512)

| iter_k | t/s | Cumulative Delta |
|---|---:|---|
| 256 (Ampere) | 11,176 | — |
| 512 | 17,758 | +59% |
| 1024 | 21,133 | +89% |
| 2048 | 22,896 | +105% |
| 4096 | 23,284 | **+108%** |
| 8192 | 23,107 | +107% (flat) |

**Ceiling at 4096.** 8192 helps pp128 (+8%) but flat at pp512+.

---

## Key Lessons Learned

### 1. NVFP4 iter_k Is Fundamentally Different
The NVFP4 tile-load function (`mmq-load-tiles.cuh:1743`) uses template `iter_k` for
`threads_per_row = iter_k / QK_NVFP4`, unlike all other quant types whose tile-load
uses the fixed `MMQ_ITER_K=256` macro. Increasing NVFP4 iter_k changes thread geometry,
not just the outer loop stride. **NVFP4 must stay at iter_k=512 (MMQ_ITER_K_FP4).**

### 2. CASE Macro First-Match Semantics
The `CASE(type, I_pad, J_min, J_max, J, sram, K_vram, do_quant, fallback)` macro
uses `if (type == T && J_pad >= pad && J_min <= J && J <= J_max && ...) return config;`
— first match wins. **Higher iter_k entries must come BEFORE lower iter_k entries**
for the same (type, J, fallback) tuple. Failing to do this silently produces dead code
that the compiler can't warn about.

### 3. NVFP4 `_nvfp4_nvfp4` vs Standard Tile-Load
There are TWO NVFP4 tile-load functions:
- `ggml_cuda_mmq_load_tiles_nvfp4` (line 1681): uses MMQ_ITER_K=256, for NVFP4×q8_1
- `ggml_cuda_mmq_load_tiles_nvfp4_nvfp4` (line 1737): uses template iter_k, for NVFP4×NVFP4

The crash was in the NVFP4 prefill path (NVFP4×q8_1), which uses the template iter_k
variant. This is why iter_k scaling works for Q4_K/Q5_K but crashes NVFP4.

### 4. Decode Is Memory-Bandwidth-Bound on RTX 5090
Every MMVQ tuning attempt was neutral for decode (tg128: 365→368 t/s):
- nwarps=8: **-26%** (too much shared-mem reduction overhead)
- nwarps=2: **impossible** (blocks_per_iter=0 for VDR=2 types)
- rows_per_block=2: neutral
- VDR=4 for Q4_K: neutral

The RTX 5090's 1.8 TB/s memory bandwidth saturates the MMVQ compute pipeline at
~368 t/s for a 30B MoE model. Further decode gains require reducing memory traffic
(e.g., weight-only quantization, speculative decoding) rather than compute tuning.

### 5. nwarps × VDR × warp_size / qi Must Be ≥ 1
The formula `blocks_per_iter = vdr * nwarps * warp_size / qi` must produce ≥ 1 for
the MMVQ kernel loop to function. With nwarps=2 and VDR=2 (most quant types):
2 × 2 × 32 / 256 = 0 (integer division) → infinite loop. This constraint limits
nwarps tuning options.

---

## Files Modified

| File | What Changed |
|---|---|
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | iter_k scaling for 19 quant types, NVFP4 reverted to 512 |
| `ggml/src/ggml-cuda/mmq.cuh` | MMQ_ITER_K_BB/B2/B3/B4 constants (1024/2048/4096/8192) |
| `ggml/src/ggml-cuda/mmvq.cu` | Blackwell MMVQ parameter table (MMVQ_PARAMETERS_BLACKWELL) |
| `ggml/src/ggml-cuda/vecdotq.cuh` | VDR=4 Q4_K Blackwell decode, VEC int4 loads q5_0/q5_1 |
| `ggml/src/ggml-cuda/fattn.cu` | Context-length FA dispatch (Blackwell) |
| `ggml/src/ggml-cuda/fattn-common.cuh` | VEC int4 loads for q5_0, q5_1 |
| `fp4-benchmark.md` | Prefill saturation matrix |
| `BLACKWELL_OPTIMIZATION_AREAS.md` | This document |

---

## Prioritized Next Steps

### Tier 1: Quick Wins (10-30 min)
- [ ] Try NVFP4 model with `-ngl 90` to see if crash threshold changes
- [ ] Benchmark Q4_K_XL with `-ngl 90` to measure partial offload perf
- [ ] Run `llama-perplexity` on NVFP4 model to verify correctness

### Tier 2: Medium Effort (1-3 hours)
- [ ] **Investigate speculative decoding** — the only proven way to accelerate
      memory-bandwidth-bound decode on Blackwell
- [ ] **Tune FA VEC path** for decode attention (currently uses VEC above 65536 ctx)
- [ ] Try VDR increases for Q5_K, Q6_K on Blackwell (models after Q4_K pattern)
- [ ] Profile NVFP4 model with `nsys` to identify decode bottlenecks

### Tier 3: Major Projects (days)
- [ ] Weight-only quantization for MoE models (reduce memory traffic 2-4x)
- [ ] Blackwell-specific MoE kernel with SMEM prefetch buffers
- [ ] Kernel fusion: attention + MoE gating on Blackwell

### Tier 4: Blue Sky
- [ ] FP4 tensor core mat-vec (bypass dequant entirely)
- [ ] Asynchronous weight prefetch using Blackwell's larger L2 cache
- [ ] Custom Blackwell instruction scheduling for the MMVQ inner loop

---

## Gotchas Reference

| Task | File | Watch Out For |
|---|---|---|
| iter_k changes | `mmq-config-blackwell.cuh` | CASE macro first-match; NVFP4 uses template iter_k |
| MMVQ nwarps | `mmvq.cu` | blocks_per_iter must be ≥ 1; nwarps=2 is unsafe for VDR=2 |
| VDR changes | `vecdotq.cuh` | Q4_K block layout requires u[] duplication; test bit-identical |
| FA dispatch | `fattn.cu` | Context-length thresholds are heuristics; benchmark extremes |
| NVFP4 model | `27B-NVFP4/` | Crashes at pp>1 (pre-existing cublas FP32 issue, not our code) |
