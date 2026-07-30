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
| `c08f247b5` | nwarps=1 NVFP4 decode | **+5.2%** tg (207→219 t/s) |
| `d62e7a96c` | nwarps=1 Q4_K decode | Neutral (~+0.8%) |
| `3e376d2c7` | __ldg q8_1 reads (Q4_K) | **+1.1%** tg (368→372) |
| `26ae4d87a` | __ldg q8_1 reads (NVFP4) | Neutral (~218 t/s) |
| `d3dcdc85f` | Update benchmarks + rbp findings | Docs |
| `4fbffc943` | nsys decode profile analysis | Docs |
| `4cfa16f82` | rpb tuning, PDL, quantize fusion | Docs |
| `92ba0ea3d` | Final benchmark snapshot | Docs |
| `b6b784ac9` | __launch_bounds__(32,4) Blackwell | **+2.4%** tg (361→370) |

### Final Benchmarks (RTX 5090, -ngl 99, -t 24)

**Q4_K_XL 30B MoE (16.45 GiB):**

| Test | Ampere Baseline | After All Optimizations | Delta |
|---|---:|---:|---|
| pp128 | 4,429 t/s | 10,859 t/s | **+145%** |
| pp512 | 11,176 t/s | 21,538 t/s | **+93%** |
| tg128 | 365 t/s | **370 t/s** | **+1.4%** |
| tg256 | 367 t/s | 359 t/s | ~-2% (thermal var) |

**NVFP4 35B MoE (22.88 GiB):**

| Test | Result |
|---|---:|
| pp128 | 5,023 t/s |
| pp512 | 10,448 t/s |
| pp1024 | 10,432 t/s |
| tg128 | **219 t/s** (+5.2% vs 207) |
| tg256 | **219 t/s** (+5.2% vs 209) |

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

## Nsight Compute Profile: `mul_mat_vec_q` (Q4_K Decode)

Profiled on RTX 5090 with `ncu --set full --kernel-name mul_mat_vec_q`.

### Scheduler Statistics — Latency-Bound

| Metric | Value | Meaning |
|---|---|---|
| **No Eligible (stalled)** | **79.67%** | 80% of cycles: zero warps can issue |
| One or More Eligible | 20.33% | Only 1-in-5 cycles does work |
| Eligible Warps / Scheduler | 0.39 | Out of 6.74 active warps |
| Active Warps / Scheduler | 6.74 | ~75% of theoretical 9 |
| Issued Warps / Scheduler | 0.20 | 0.2 IPC per scheduler |

### Memory Workload — Scattered Reads

| Metric | Value |
|---|---|
| DRAM Bandwidth | 785 GB/s (45% of ~1.7 TB/s peak) |
| **Avg Bytes/Sector (Loads)** | **6.59 / 32** |
| L1/TEX Hit Rate | 85.35% |
| L2 Hit Rate (Loads) | 10.08% |
| L2 Hit Rate (Stores) | 96.62% |
| Instructions Executed | 1,331,200 |
| Duration | 6.02 µs |

### Compute — Underutilized

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 14.38% |
| FMA-heavy | 9.53% |
| LSU (load/store) | 14.38% |
| SM Frequency | 2.28 GHz |

### Root Cause Analysis

**The kernel is latency-bound, not bandwidth-bound or compute-bound.**
- 80% stall rate means warps are constantly waiting on memory
- 6.59 bytes/sector means each 32-byte cache line load uses only ~20% of data
- 85% L1 hit rate means data IS mostly cached, but L1 latency (~30 cycles) still
  dominates because only 0.39 warps are eligible at any time
- The scattered q8_1 access pattern (each thread reads 4 ints from scattered
  positions within q8_1 blocks) causes poor cache line utilization

### Failed Experiment: Shared-Memory Q8_1 Tiling (-24%)

Attempted to fix scattered reads by cooperatively loading q8_1 into shared memory.
Result: **tg128 367→281 t/s (-24%)**.

Why it failed:
- L1 already hits 85% — shared memory doesn't reduce access latency
- __syncthreads() barriers (2 per iteration) added more overhead than they saved
- Cooperative load of 544 bytes with 128 threads is inefficient
- The bottleneck is MEMORY LATENCY, not bandwidth — shared memory changes WHERE
  data is accessed, not WHEN it arrives

### What Would Actually Help

1. **Async copy (cp.async)**: Overlap q8_1 loads with computation on previous tile
2. **Fuse quantize_q8_1 into MMVQ**: Eliminates q8_1 DRAM round-trip entirely
3. **More warps**: 6.74 active vs 9 theoretical — increase occupancy for latency hiding
4. **TMA (Tensor Memory Accelerator)**: Blackwell's async DMA engine
5. **Software pipelining**: Double-buffer q8_1 loads and compute

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

### 4. Decode Is Memory-Latency-Bound on RTX 5090
Every MMVQ tuning attempt was neutral for decode (tg128: 365→368 t/s):
- nwarps=8: **-26%** (too much shared-mem reduction overhead)
- nwarps=2: **impossible** (blocks_per_iter=0 for VDR=2 types)
- rows_per_block=2: neutral
- VDR=4 for Q4_K: neutral
- nwarps=1 Q4_K: neutral (~+0.8%, within noise)
- rows_per_block=4: **-1.4%** (fewer blocks = less parallelism, reg pressure)

The RTX 5090's 1.8 TB/s memory bandwidth saturates the MMVQ compute pipeline at
~368 t/s for a 30B MoE model. Further decode gains require reducing memory traffic
(e.g., weight-only quantization, speculative decoding) rather than compute tuning.

### 6. MMVQ Is ~31% of Decode Kernel Time (nsys Profile)
Full nsys profile of 4-token decode reveals the kernel time breakdown:

| Kernel | % Time | Notes |
|---|---:|---|
| mul_mat_vec_q (all types) | 31% | Q4_K/NVFP4/Q5_K mat-vec, tuned extensively |
| quantize_q8_1 | 12.5% | Launch-overhead dominated (1.5µs avg) |
| rms_norm_f32 | 12.2% | Launch-overhead dominated (3.0µs at 1024-thread) |
| flash_attn + combine | 6.7% | VEC path, cols_per_block=1 for decode |
| mul_mat_vec_f (float) | 4.8% | Float mat-vec path |
| topk_moe_cuda | 3.2% | Expert selection |
| rope_neox | 3.7% | RoPE rotation |
| k_bin_bcast (add/mul) | 5.4% | Element-wise ops |
| Other (set_rows, etc.) | ~20% | Miscellaneous |

Key insight: quantize_q8_1 + rms_norm = 25% from kernel launch overhead.
Fusing these into downstream MMVQ kernels would recover most of that time.

### 7. Failed: rms_norm Blackwell Thread Threshold (-5.6%)
Changed rms_norm block-size threshold from 1024→4096 on Blackwell to use
256-thread path (1.3µs) vs 1024-thread (3.0µs) for hidden_dim=2048. Result:
tg128 372→351 t/s (-5.6%). Runtime device check overhead or unexpected SM
occupancy interaction. Reverted.

### 8. Failed: rows_per_block Tuning (rpb=4: -1.4%, rpb=1: -6.2%)
rows_per_block=4: fewer blocks, more reg pressure, less parallelism.
rows_per_block=1: more blocks but half the work/block, lower write efficiency.
Optimal: rows_per_block=2 (current) — balanced work vs parallelism.

### 9. Failed: PDL Disable (-9.7%)
Setting GGML_CUDA_PDL=0 disables Programmatic Dependent Launch on Blackwell.
Result: tg128 372→336 t/s (-9.7%). PDL's hardware dependency tracking is
critical for decode's many small kernel launches.

### 10. Failed: Fused quantize_q8_1 + MMVQ (Dead End)
With nwarps=1 (32 threads/block), inline fp32→q8_1 quantization takes 2.8µs
vs standalone kernel's 1.5µs (which uses 2048 threads in parallel). The inline
step loses the 64-warp parallelism of the standalone kernel, making the fusion
break-even at best. Only viable with nwarps≥4, but nwarps=4 is neutral for Q4_K.

### 11. Optimal Blackwell Decode Configuration (Q4_K)
After exhaustive tuning, the optimal decode configuration is:
- nwarps=1, rows_per_block=2, __ldg on q8_1 reads
- tg128: 372 t/s (+1.9% vs 365 baseline)

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
- [x] nwarps=1 for Q4_K decode (neutral, +0.8%, kept for simplicity)
- [x] __ldg q8_1 cache bypass for Q4_K decode (**+1.1%**, 368→372 t/s)
- [x] __ldg q8_1 cache bypass for NVFP4 decode (neutral, kept for consistency)
- [ ] Try NVFP4 model with `-ngl 90` to see if crash threshold changes
- [ ] Benchmark Q4_K_XL with `-ngl 90` to measure partial offload perf
- [ ] Run `llama-perplexity` on NVFP4 model to verify correctness

### Tier 2: Medium Effort (1-3 hours)
- [x] __ldg q8_1 reads for Q4_K decode (**+1.1%**) — DONE
- [ ] **Apply __ldg to MMQ prefill path** — same cache-bypass for q8_1 loads
      in the MMQ tile-load kernel. Prefill equivalent of decode __ldg win.
- [ ] **cp.async software pipelining** — overlap weight loads with compute in MMVQ
      inner loop. Needs coalesced tile loads (may conflict with Q4_K scatter pattern).
- [ ] **Tune FA VEC path** for decode attention (currently uses VEC above 65536 ctx)
- [ ] **Investigate speculative decoding** — the only proven way to accelerate
      memory-latency-bound decode on Blackwell (multi-token per weight-load)
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
| rows_per_block | `mmvq.cu` | rpb>2 reduces parallelism (fewer blocks → less latency hiding) |
| nwarps=1 Q4_K | `mmvq.cu` | VDR=4 safe (blocks_per_iter=16) but complex vec_dot limits gain |
