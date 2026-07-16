# Blackwell (SM120 / RTX 5090) Optimization Areas

**Target**: RTX 5090 (SM 120, CUDA 13.3, 32 GB VRAM, 99 KB shared memory)
**Repository**: `/mnt/storage/blackbeard` (llama.cpp fork)
**Branch `main`**: Current production branch with NVFP4, FA TILE/VEC/MMA dispatch
**Branch `tq`**: TurboQuant KV cache branch (TBQ3, TURBO3 types)

---

## 1. Flash Attention Kernel Architecture

### Current State
- FA2-level kernels (warp-level tiling, `mma.sync.aligned.m16n8k16`)
- 4 kernel paths: VEC (decode), TILE (no-tensor-core), MMA-F16 (tensor core prefill), WMMA-F16 (AMD)
- SM120 has **no wgmma, no TMA, no tcgen05** — confirmed by FA4's SM120 path inheriting SM80 and multiple public kernel porting reports.
- 99 KB shared memory (168 KB on Ampere, 228 KB on SM100 — smaller means tighter tile sizing).
- Practical consequence: on SM120, tile-size-derived accumulators must live in SMEM or registers, not TMEM. This changes the viable config space versus SM100.

### Optimization Candidates

#### 1a. Stream-K Persistent Tile Scheduling
- **What**: Instead of launching one CTA per output tile, keep CTAs resident and dynamically feed them tiles from a work queue. Better SM utilization on irregular prefill workloads.
- **Status**: **Already implemented in the MMA F16 kernel.** `launch_fattn()` in `fattn-common.cuh` has two dispatch paths (line 1120): Stream-K (`stream_k=true`, used by MMA F16) and parallel-split (`stream_k=false`, used by TILE/VEC). The Stream-K path launches fewer CTAs than tiles and dynamically schedules (KV-tile × output-tile) work via a work-queue pattern. Partial results are merged post-hoc by `flash_attn_stream_k_fixup_uniform` or `flash_attn_stream_k_fixup_general`. The gate condition (`cc >= ADA_LOVELACE || ...`) is satisfied on SM120.
- **Observation**: No implementation gap for MMA F16. The TILE and VEC paths use the parallel-split path instead, which is a design choice (they target decode/small-batch where persistence overhead doesn't pay).
- **Reference**: `launch_fattn()` `stream_k` parameter at `fattn-common.cuh:975`, MMA F16 passes `true` at `fattn-mma-f16.cuh:1963`.
- **Applicable GPUs**: All CUDA GPUs.
#### 1b. Split-KV Decode (Long Context)
- **What**: Distribute K rows across multiple CTAs for long-context decode. Each CTA processes a chunk of K rows; results merged via online softmax.
- **Status**: **Already implemented** in the TILE and VEC kernels. `launch_fattn()` (line 1151) computes `parallel_blocks` — the number of CTAs along the K dimension — and launches `blocks_num.y = parallel_blocks`. Each CTA processes `nbatch_fa` K rows. Partial VKQ accumulators and softmax stats are stored in `dst_tmp`/`dst_tmp_meta` and merged post-hoc by `flash_attn_combine_results<DV>` (line 1270). The VEC kernel explicitly handles this at lines 500-514 where `gridDim.y != 1` triggers the merge-data path.
- **Observation**: Already handles the full long-context decode case. The MMA F16 kernel uses Stream-K instead (see 1a), which is a more sophisticated variation.
- **Reference**: `parallel_blocks` logic at `fattn-common.cuh:1115-1184`, merge kernel at `fattn-common.cuh:914-969`.
- **Applicable GPUs**: All.
#### 1c. Tile Size Auto-Tuning for 99 KB SMEM
- **What**: The TILE and MMA kernels use fixed tile sizes inherited from upstream. Determine whether SM120's 99 KB SMEM (vs. 168 KB on Ampere) requires smaller tiles, or whether different tile sizes could improve performance.
- **Status**: **Analyzed 2026-07-16. SMEM pressure is not the bottleneck.**
  - Full SMEM budget calculation across all Ampere MMA config entries shows the worst case is ~38 KB (D=128, ncols=64, nstages=2) — well within 99 KB. Details:
    - D=128, ncols=8: ~35 KB  — `nstages=2, nbatch_fa=128`
    - D=128, ncols=64: ~28 KB — `nstages=2, nbatch_fa=64`
    - D=512, ncols=8: ~20 KB  — `nstages=1, nbatch_fa=32`
    - D=576, ncols=32: ~37 KB — `nstages=1, nbatch_fa=32`
  - The "D=128 exceeds 99 KB" claim is incorrect. The Ampere D=128 config uses `nbatch_fa=128, nstages=2, nthreads=128` and fits comfortably.
  - The `ggml_cuda_fattn_mma_get_config_sm120()` function described in the previous state **does not exist** in the codebase. No SM120-specific MMA config function was written. The MMA host dispatch (line 231) checks `ampere_mma_available(cc)` first — true for SM120 — and routes to the Ampere config table unchanged.
- **Real gap**: SM120 could likely support **larger** tiles than Ampere for some configs (more nbatch_fa, higher occupancy targets), but this has not been explored. The constraint is register pressure and occupancy, not SMEM. A `ggml_cuda_fattn_mma_get_config_blackwell()` config layer could be added to tune for SM120's tradeoffs.
- **Template hazard confirmed**: `nbatch_fa` appears in `constexpr` array-size contexts in template instantiations. Any new config values must be validated against every compiled (DKQ, DV, ncols1, ncols2) tuple.
- **Difficulty**: Low for an SM120 config table (one function), but Medium for meaningful tuning (requires systematic benchmark sweeps).
- **Applicable GPUs**: SM120-specific.
- **Next action**: If pursuing, write a Blackwell config layer in `fattn-mma-f16.cuh` and tune for register-pressure/occupancy on SM120. This is a clean-slate addition, not a fix.
#### 1d. Warp Specialization (without wgmma)
- **What**: Split warp groups into orchestrator (issue loads/mma) and compute (softmax). Can be done with `mma.sync` + `cp.async`, though less effective than with TMA.
- **Status**: Not implemented. All warps are symmetric.
- **Difficulty**: High. Major kernel restructuring.
- **Reference**: FA3/FA4 warp specialization pattern.
- **Applicable GPUs**: Any with `cp.async` (Turing+). Diminishing returns without TMA on SM120.

#### 1e. VEC Kernel Inner-Loop Optimization (for Quantized KV)
- **What**: The VEC kernel's quantized KQ inner loop (`vec_dot_fattn_vec_KQ_*` functions in `fattn-common.cuh`) uses scalar 4-byte loads (`ggml_cuda_memcpy_1<sizeof(int)>`) per quantized element. Replace with 16-byte vector loads packed into registers before unpacking indices.
- **Status**: **Incorrectly scoped in earlier analysis — no `fattn-turbo.cuh` exists in main branch.** The VEC kernel handles all quantized KV paths. The inner loop access pattern is coalesced at the warp level (each thread reads adjacent 4-byte quantities with a stride of `nthreads`), so it's not "uncoalesced" in the classic sense. However, replacing four 4-byte loads with one 16-byte vector load per 4-element group would reduce load instruction count by 4×, improving instruction throughput and L2 utilization.
- **Difficulty**: Low-Medium. Localized to `vec_dot_fattn_vec_KQ_q4_0`, `_q4_1`, `_q5_0`, `_q5_1` in `fattn-common.cuh`.
- **Applicable GPUs**: All, but most impactful for small-D decode on quantized KV.
- **Note**: The `tq` branch has a separate TURBO3 kernel (`fattn-turbo.cuh`) for the TurboQuant types — its inner loop optimization is tracked as 2a below.
#### 1f. D=512 Tile Instance Validation on SM120
- **What**: Gemma 4 uses D=512 attention. Determine whether the existing TILE/MMA D=512 configs are safe on SM120's 99 KB SMEM.
- **Status**: **Already conservatively configured.** The Ampere MMA table D=512 entries use the minimum viable settings:
  - `nbatch_fa=32, nstages=1, Q_in_reg=false`
  - SMEM budget: ~20 KB (ncols=8) to ~34 KB (ncols=32), well within 99 KB
  - The TILE kernel D=512 entries use `nbatch_fa=64`, also safe
- There is no SMEM or occupancy issue to solve. Performance is gated by architectural limits (no wgmma/TMA), not tunable parameters.
- **Applicable GPUs**: SM120-specific.
#### 1g. Meta-Tuning Inputs for SM120
- **What**: Use public SM120-specific FA implementations as reference configs for tile shape, `BLOCK_Q`, `BLOCK_KV`, and occupancy.
- **Status**: Research only.
- **Reference**:
  - `gau-nernst/fa-5090` — speed-of-light 5090 FA writeup.
  - `florianmattana/fp4-fused-attention-sm120` — SM120 SMEM budget and LMUL/fragment layout experiments.
  - `0xSero/blackwell-gpu-wiki` — SM120 accumulator/smem cliff summary.
- **Applicable GPUs**: SM120.

---

### Summary: Flash Attention Candidate Feasibility

| Candidate | Claimed Status | Actual Status | Work Required | Priority |
|---|---|---|---|---|
| 1a Stream-K | Not implemented | **Already implemented** (MMA F16) | None | N/A |
| 1b Split-KV | Not implemented | **Already implemented** (TILE/VEC) | None | N/A |
| 1c Tile tuning | SMEM pressure needs smaller tiles | **False premise** — Ampere configs fit 99 KB; real gap is register/occupancy tuning | Add Blackwell config layer + benchmarks | Low |
| 1d Warp specialization | Not implemented, high difficulty | Accurate | Major rewrite; low ROI without TMA | Low |
| 1e VEC inner loop | TURBO3 kernel, byte reads | **No TURBO3 kernel on main**; VEC has reasonable coalescing | 128-bit loads in quantized paths | Low |
| 1f D=512 validation | Untested | **Already conservatively configured** | None | N/A |
| 1g Meta-tuning | Research only | Valid; references exist | Read repos; no action without hypothesis | Informational |

**Largest unfilled opportunity not in list**: A `ggml_cuda_fattn_mma_get_config_blackwell()` config tier for the MMA F16 kernel — the actual missing SM120-specific dispatch. The MMA F16 kernel runs entirely through Ampere-mapped `mma.sync.aligned.m16n8k16` with no Blackwell-specific tile config, no Blackwell-specific instructions, and no awareness of SM120's different accumulator/SMEM/register tradeoffs.

---

## 2. KV Cache Compression (tq branch)

### Current State
- 24 TBQ types ported (TBQ3_0 through TBQP4_4, AMX3_1, AMXV3_1, TBQV3_1)
- 3 TURBO types (TURBO2/3/4, PolarQuant-based, no WHT)
- TBQ3 working: 135 t/s decode, 20.4 GB @ 256K ctx on Gemma 4 26B
- TURBO3 working: 73 t/s decode (unoptimized inner loop)
- Upstream TurboQuant interest is active: mainline llama.cpp has public TurboQuant integration discussion, and multiple serving stacks are adding fused TurboQuant/4-bit KV decode kernels.

### Optimization Candidates

#### 2a. TURBO Kernel Decode Speed (Priority)
- **What**: The TURBO kernel's KQ loop reads one byte per element from global memory. Loading 128-bit words and extracting indices in registers would approach TBQ3 speed.
- **Status**: Root cause identified. Tracked from 73 t/s target toward TBQ3-level decode.
- **Difficulty**: Low-Medium. Localized to `fattn-turbo.cuh` inner loop.
- **Dependency**: Branch `tq`.
- **Note**: This is the highest local-KV payoff because the current bottleneck is memory access pattern, not compute.

#### 2b. Hadamard K/V-Cache (from ikawrakow/ik_llama.cpp)
- **What**: Apply Hadamard (WHT) as a separate `ggml_hadamard` op before quantization. Works with ANY existing type (Q4_0, Q8_0, IQ4_NL, etc.). Improves quality by decorrelating elements.
- **Status**: Not ported.
- **Difficulty**: Medium. Self-contained change across ggml/llama/common and CUDA kernels.
- **CLI**: `-khad`/`--k-cache-hadamard`, `-vhad`/`--v-cache-hadamard`.
- **Reference**: ikawrakow/ik_llama.cpp PRs 1033/1034/1527; local repo already auto-builds WHT rotation matrices when quantized caches are enabled in `llama-kv-cache.cpp`.
- **Evidence**: upstream repo README notes Q4_0 achieves low perplexity even without Hadamard; WHT still yields meaningful gains for more aggressive quants.

#### 2c. FP8 KV Cache Calibration
- **What**: Store K/V as FP8 with calibration-driven per-tensor or per-head scale factors instead of naive min/max.
- **Status**: No implementation. Concept only.
- **Difficulty**: High. Requires calibration dataset, scale factor management, and FA changes.
- **Note**: FP8 KV cache is now public in serving stacks. vLLM docs and blog explicitly support `kv_cache_dtype="fp8"` with calibration via llm-compressor. The remaining work is integration, not reverse-engineering.
- **Reference**:
  - vLLM Quantized KV Cache docs.
  - vLLM FP8 KV cache blog, April 2026.
- **Expected benefit**: context-length/concurrency wins at same VRAM, not absolute decode speed.

#### 2d. Low-Perplexity Q4_0 KV Cache (from ikawrakow)
- **What**: Improved Q4_0 quantization specifically for KV cache. Adjusts scale factors to reduce perplexity.
- **Status**: Not ported.
- **Difficulty**: Medium. Modifications to Q4_0 quantize kernel.
- **Reference**: ikawrakow/ik_llama.cpp PRs 1547, 1556.
- **Evidence**: upstream README lists these PRs under "Low perplexity Q4_0 KV cache", suggesting validated quality improvements.

#### 2e. Learned Rotation Matrices (from TheTom / OSCAR lineage)
- **What**: Attention-aware offline covariance rotation for INT2 KV cache quantization, with fixed rotation + clipping baked into GGUF serving path.
- **Status**: Reference available at `/mnt/storage/Projects/turboquant` branch `oscar-clean`. Public paper/code also exists under FutureMLS-Lab/OSCAR and Together AI open-sourced the method.
- **Difficulty**: High. Training/conversion pipeline plus GGUF format work.
- **Reference**:
  - OSCAR paper, arXiv 2605.17757, May 2026.
  - TheTom/llama-cpp-turboquant.
- **Why it matters**: INT2 KV is attractive for 128K+ on 32 GB, but requires offline calibration.

#### 2f. INT2 Min-Max + Oscar2 FA Kernel (from TheTom)
- **What**: 2-bit quantization with min-max centroid selection, combined with a dedicated FA kernel. Different tradeoff vs 3-bit Lloyd-Max.
- **Status**: Reference at `/mnt/storage/Projects/turboquant` branch `oscar`.
- **Difficulty**: Medium. New block type + FA kernel, similar scope to TURBO.
- **Reference**: `/mnt/storage/Projects/turboquant` (TheTom, `oscar` branch).

#### 2g. WHT-Space Fused Attention (from ManthanQuant)
- **What**: Keep Q, K in WHT-rotated space through the entire attention computation. Avoids inverse-WHT on K/V. Dot product preserved. Universal N(0,1) centroids.
- **Status**: Reference implementation at `atcuality2021/manthanquant`.
- **Difficulty**: Medium-High. Kernel restructuring.
- **Note**: Our TBQ3 already has baked-in WHT, so this is partially redundant. Main benefit is code clarity and using true paper-optimal centroids.

#### 2h. Fused TurboQuant Attention Paths in Servicing Stacks
- **What**: SGLang vllm-style fused decode/extend kernels that read packed 4-bit KV directly during attention, with zero intermediate dequant buffer.
- **Status**: Not implemented locally.
- **Difficulty**: High. New backend/cuGraph integration.
- **Reference**:
  - SGLang issue requesting native TurboQuant support.
  - vLLM PR adding TurboQuant v1 attention path.
- **Why it matters**: Even if local llama.cpp does not use this path, it shows the competitive direction of fused KV-facade attention.

---

## 3. Weight Quantization & Compute

### Current State
- `GGML_TYPE_NVFP4 = 40`: FP4 weight type with hardware decode
  - PTX `cvt.rn.f16x2.e2m1x2` for hardware E2M1 decode
  - `prmt.b32` register-only LUT for FP4 lookup (8-entry, two uint32 registers)
  - W4A8 and W4A16 paths
  - **NVFP4 has Blackwell-specific dispatch**: `ggml_cuda_mmq_get_config_blackwell()` uses compact FP4 SRAM layout (`MMQ_SRAM_LAYOUT_FP4`, stride=76 32-bit elements vs Ampere NVFP4 stride=84), `MMQ_ITER_K_FP4=512`, and VDR=8 MMVQ kernel with 128-bit vectorized loads
  - Fallback path uses generic Ampere NVFP4 config when `force_w4a8` is active (W4A16 → W4A8 precision retention)
- `GGML_TYPE_MXFP4`: Also uses Blackwell-specific config with FP4 SRAM layout
- Standard llama.cpp quant types (Q4_0 through IQ4_NL, k-quants): use generic Ampere MMQ config, no SM120-specific tuning

### Optimization Candidates

#### 3a. SM120-Aware NVFP4 Kernel Tuning
- **What**: The current NVFP4 path is usable on SM120. SM120 shares the SM80/WGMLA warping model, not SM100's `tcgen05.mma`/TMA.
- **Status**: **NVFP4 is already SM120-tuned.** Details by code audit:
  - MMQ: `ggml_cuda_mmq_get_config_blackwell()` selects a compact FP4 SRAM layout (stride=76 32-bit elements) vs the Ampere NVFP4 path (stride=84). This saves ~8 KB shared memory, which is essential for J>=112 on SM120's 99 KB ceiling.
    - SMEM budget for Blackwell FP4 layout: J=128 uses ~94.5 KB → fits. Ampere NVFP4 layout J=128 uses ~102.5 KB → **exceeds 99 KB** and is skipped by the J search loop.
  - MMVQ: Blackwell-specific `vec_dot_nvfp4_q8_1_blackwell()` with VDR=8, 128-bit vectorized loads, and `prmt.b32` register-only LUT (no SMEM table lookup). Non-Blackwell NVFP4 uses VDR=4.
  - The W4A8 path (`force_w4a8=true`) intentionally uses the Ampere config to keep activations at higher precision — this path may have SMEM-limited J selection on SM120 (J>=112 skipped).
- **Remaining gap**: Standard quant types (Q4_0 through IQ4_NL, k-quants) have no SM120-specific MMQ config. They use the Ampere MMQ table unchanged. SM120 could benefit from adjusted thread counts or tile sizes for these types.
- **Evidence partially verified**: The public SM120 claim "autotuner skips SM120 GEMM tiles because they overflow 99KB shared memory" is accurate for the Ampere NVFP4 layout (used by W4A8 fallback), but the primary Blackwell FP4 path already handles full J range.
- **Applicable GPUs**: SM120.
- **Caveat**: Pure SM100-specific DSL paths remain off-target for RTX 5090.

#### 3b. IQK Quant Improvements (from ikawrakow)
- **What**: IQ2_KS, IQ4_KS, IQ5_KS, IQ1_S_R4, IQ2_K_R4, etc. — improved importance-matrix-aware weight quants with CUDA quantized GEMM kernels.
- **Status**: Not ported. Upstream llama.cpp doesn't have these.
- **Difficulty**: High. Many new types, each needing CPU + CUDA implementation.
- **Reference**: ikawrakow/ik_llama.cpp (multiple PRs).

#### 3c. Trellis Quants (from ikawrakow)
- **What**: IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT — integer-base trellis quantization for 1-4 bpw. Novel approach with reasonable CPU performance.
- **Status**: Not ported.
- **Difficulty**: Very High. Novel algorithm, significant kernel work.
- **Reference**: ikawrakow PR 113, PR 616, PR 441, etc.

#### 3d. Row-Interleaved Quant Packing (from ikawrakow)
- **What**: `-rtr` flag repacks tensors to row-interleaved format for CPU/GPU hybrid performance. Better cache behavior for CPU-offloaded layers.
- **Status**: Not ported.
- **Difficulty**: Medium. Affects tensor layout and all matmul kernels.
- **Reference**: ikawrakow README (with caveats about k-quants not supporting CUDA row-interleaved).

#### 3e. J Tile Size Validation for Non-FP4 Types on SM120
- **What**: Standard quant types on SM120 use the Ampere MMQ config table unchanged. The J selection loop in `mul_mat_q_switch_J` (line 1469) skips configs where `mmq_get_nbytes_shared()` exceeds `smpbo` (99 KB). On SM120, the Ampere config's larger-tile entries may exceed SMEM for J values that work on Ampere's 168 KB.
- **Status**: **Not benchmarked, but SMEM analysis shows limited risk.**
  - For standard types with `MMQ_ITER_K=256` and NVFP4 SRAM layout (stride=84), J=112/128 would exceed 99 KB.
  - The J search loop already handles this correctly: it iterates J=8..128 in steps of 8 and skips any config exceeding SMEM. On SM120 this naturally limits to J<=96 for the Ampere NVFP4 layout.
  - For Q8_0 SRAM layout (stride=76, same as Blackwell FP4), all J values fit.
- **Previous VDR framing was incorrect**: VDR (vectorization depth per row) does not affect shared memory. It controls how many elements each thread processes in MMVQ, which has no SMEM dependency. The SMEM concern applies to the MMQ tile size J, not VDR. Blackwell NVFP4 already uses VDR=8 (the highest available for any type).
- **Difficulty**: Low. The J selection loop already self-limits correctly. Validation would involve benchmarking whether the auto-selected J values are optimal on SM120.
- **Applicable GPUs**: SM120.

---
## 4. MoE Optimization

### Current State
- Three dispatch paths for MoE matmuls:
  - **Fast path (MMVQ/MMQ/MMF)**: Uses `ggml_cuda_mul_mat_vec_q`/`ggml_cuda_mul_mat_q`/`ggml_cuda_mul_mat_f` with expert routing via `ids` tensor. All tokens are packed into a single batched matmul per type, no per-expert serialization. Includes a dedicated `mul_mat_vec_q_moe` kernel (line 716 in `mmvq.cu`) for multi-token MoE with no shared memory overhead — each warp handles one token independently.
  - **Slow path (serial loop)**: `ggml_cuda_mul_mat_id` (line 1784 in `ggml-cuda.cu`) falls back to host-side expert assignment via `cudaMemcpyAsync` + `cudaStreamSynchronize`, then loops over experts one-by-one. This path is only taken when batch size exceeds MMVQ/MMQ thresholds.
- **Fused FFN (GLU) compute exists**: Subgraph graph fusion (`ggml_cuda_should_fuse_mul_mat`, line 1599) detects patterns `{MUL_MAT_ID, MUL_MAT_ID, GLU}` and `{MUL_MAT_ID, ADD_ID, MUL_MAT_ID, ADD_ID, GLU}` and executes them as a single fused kernel. This applies to the FFN compute step only — not the full MoE routing.
- **Top-k MoE kernel**: Local `topk-moe.cu` provides fused softmax/sigmoid + top-k selection + expert routing, supporting 1–576 experts with bias/scale/clamp. Used for gating.
- GPU-offloaded experts, CPU fallback for overflow.

### Optimization Candidates

#### 4a. Fused Full MoE Dispatch (from ikawrakow)
- **What**: Fuse the entire MoE flow — gating (top-k), expert scatter, FFN computation, and reduction — into one kernel to minimize kernel launch overhead and intermediate data movement.
- **Status**: **Partially implemented.**
  - FFN compute fusion (gate + up + GLU matmuls) is already active via subgraph fusion — `{MUL_MAT_ID, MUL_MAT_ID, GLU}` and `{MUL_MAT_ID, ADD_ID, MUL_MAT_ID, ADD_ID, GLU}` patterns are dispatched as single fused kernels. The ikawrakow PR 229 reference overlaps with what already exists.
  - Full MoE dispatch fusion (top-k selection → scatter → FFN compute → gather → reduction) is NOT implemented. The gating step (`topk-moe.cu`) runs as a separate kernel before the matmul dispatch.
- **Difficulty**: High. A full MoE dispatch fusion would require a new CUDA kernel that combines routing/FFN/reduction, plus new ggml ops or a deeper integration into the graph builder. The scope is larger than just the FFN compute fusion that already exists.

#### 4b. Expert Parallel Dispatch
- **What**: Parallelize expert computation across SMs when a MoE layer has fewer active experts than SMs.
- **Status**: **Partially addressed by the fast path.**
  - The MMVQ/MMQ/MMF paths (used when batch ≤ thresholds) process all tokens for all experts in a single batched launch. The `mul_mat_vec_q_moe` kernel assigns one warp per token, naturally spreading work across SMs.
  - The slow path (large batch) serializes experts in a host loop: `for (int64_t i02 = 0; i02 < ne02; ++i02)`. Each expert launch is sequential, with `cudaStreamSynchronize` between them.
  - Remaining gap: The slow path could launch all expert matmuls concurrently using multiple CUDA streams or a single kernel that iterates active experts internally.
- **Difficulty**: Medium. For the slow path, the main blocker is the `cudaStreamSynchronize` and host-side sorting. A device-side work queue or multi-stream dispatch could remove the serialization.

#### 4c. Dynamic Expert Offloading (Auto-Fit)
- **What**: Automatically determine which experts/layers to offload to GPU based on available VRAM.
- **Status**: Not ported. No auto-fit mechanism exists in this codebase.
- **Difficulty**: Medium. Affects model loading and scheduling.
- **Reference**: ikawrakow PR 1501, 1504, 1872.

#### 4d. SM120 MoE Kernel Shape Validation
- **What**: At small batch, MoE inference on RTX 5090 can be scattering-bound. Validate tile shapes for common expert widths on SM120.
- **Status**: Not benchmarked locally.
- **Difficulty**: Medium.
- **Note**: The existing `mul_mat_vec_q_moe` kernel (VDR-based, no shared memory) already avoids the SMEM constraint that limits other kernels on SM120. Tile shape validation would focus on the MMQ tile-size J selection in the batched path, which uses the generic Ampere config for non-FP4 types.

---

## 5. Memory & Data Movement

### Current State
- `cp.async` used in the MMA F16 FA kernel for KV tile loads (`fattn-mma-f16.cuh` line 400) and mask loads (line 475). Prefetch hint is hardcoded at 64 bytes for KV tiles; mask uses `nbatch_fa * sizeof(half)` or 64 bytes.
- `cp.async` NOT used in MMQ, MMVQ, or TILE/VEC FA kernels — those use plain `ggml_cuda_memcpy_1` loads.
- No TMA, no `cp.async.bulk`, no `wgmma`, no `tcgen05` — none of the SM100+ data movement features exist in this codebase.
- 99 KB shared memory per block on SM120. The `sharedMemPerBlockOptin` is read from device props (`ggml-cuda.cu` line 297), with a bug workaround for Blackwell drivers that return garbage (clamped to 48 KB fallback if outside 1 KB–256 KB range).
- `cudaFuncAttributeMaxDynamicSharedMemorySize` IS already used in the MMA F16 kernel (`fattn-mma-f16.cuh` lines 1945, 1956) to raise the dynamic SMEM limit. The MMQ kernel also uses it via `CUDA_SET_SHARED_MEMORY_LIMIT` (`mmq.cuh` lines 1392-1393).
- KV cache layout: K and V share the same layout (contiguous along head dim × seq len). No V-transpose.

### Optimization Candidates

#### 5a. Shared Memory Budget Optimization
- **What**: Audit every kernel's SMEM usage against 99 KB limit.
- **Status**: **Already partially done.** Key findings from codebase audit:
  - MMA F16 kernel: worst case ~38 KB (D=128, ncols=64, nstages=2). Well within 99 KB. The kernel already uses `cudaFuncSetAttribute` to raise the SMEM limit as needed.
  - MMQ kernel: the J selection loop (`mmq.cuh` line 1469) already skips configs where SMEM exceeds `smpbo`. On SM120, this naturally limits the Ampere NVFP4 layout to J≤96 (J=112/128 exceed 99 KB).
  - The accumulator placement concern from SM100 (TMEM vs SMEM) does not apply to SM120 — SM120 has no TMEM, so accumulators live in registers or SMEM, which is the normal Ampere-level design.
- **Remaining gap**: The TILE and VEC FA kernels use dynamic SMEM for KQ/VKQ buffers but have no SM120-specific sizing. Their Ampere-derived tile sizes already fit within 99 KB.
- **Difficulty**: Low. Audit confirms no SMEM emergency. No code changes needed.
- **Applicable GPUs**: SM120-specific.

#### 5b. V-GMEM Transpose for Better Coalescing
- **What**: Store V cache in transposed layout (d × page_size instead of page_size × d) so V loads during attention are coalesced.
- **Status**: Not implemented. V is stored same layout as K.
- **Difficulty**: Medium-High. Changes KV cache layout, allocation, copy paths, and all V read paths.
- **Reference**: FA4 `paged_kv.py` V-transpose concept for SM100; applicability to SM120 VEC/TILE kernels needs local validation.
- **Note**: V-transpose would benefit the TILE and VEC kernels most (they do per-row V access). The MMA F16 kernel loads V tiles into SMEM and already achieves coalesced access via shared memory.

#### 5c. Async Prefetch Sizing for SM120
- **What**: Use `cp.async` to prefetch quantized KV cache blocks before the warp needs them, or tune existing preload hints for SM120.
- **Status**: **cp.async prefetch already exists in MMA F16 (preload=64) but is absent from the quantized KV path.**
  - MMA F16: uses `preload=64` for KV tile loads. This is a moderate prefetch hint — 128B or 256B might better utilize SM120's memory controller on large tiles.
  - TILE/VEC kernels: do NOT use `cp.async` at all. They load K/V via `ggml_cuda_memcpy_1`, which is a plain load path with no prefetch hint.
- **Difficulty**: Medium for extending cp.async to TILE/VEC; Low for changing preload constant in MMA F16.
- **Note**: The claim that "larger 256B preload may not win on consumer" is plausible but unverified. SM120's HBM controller on RTX 5090 may respond differently to prefetch hints than datacenter SM100. A microbenchmark sweep of 64/128/256 preload values on MMA F16 KV tile loads would be straightforward.

#### 5d. SMEM/L1 Carveout for FA Kernels
- **What**: Use runtime shared-memory carveout controls to move capacity between L1 and SMEM depending on kernel shape.
- **Status**: **Already partially used.** `cudaFuncSetAttribute` with `cudaFuncAttributeMaxDynamicSharedMemorySize` is called by:
  - MMA F16 kernel (`fattn-mma-f16.cuh` lines 1945, 1956)
  - MMQ kernel (`mmq.cuh` lines 1392-1393 via `CUDA_SET_SHARED_MEMORY_LIMIT`)
  The TILE and VEC kernels do NOT use it (they request 0 bytes of dynamic SMEM).
- **Remaining gap**: The Blackwell-specific `preferredSharedMemoryCarveout` control (`cudaFuncAttributePreferredSharedMemoryCarveout`) is NOT used anywhere. This could allow overriding the default L1/SMEM split to favor L1 for kernels with small SMEM footprints (e.g., MMVQ, VEC FA).
- **Difficulty**: Low.
- **Reference**: NVIDIA Blackwell tuning guide on unified shared memory/L1/texture carving.
- **Applicable GPUs**: SM120 (and SM100).

#### 5e. Prefetch Chunk Size Audit
- **What**: Validate `cp.async` preload sizes against SM120 bandwidth/latency.
- **Status**: **Partially overlapping with 5c.** MMA F16 uses preload=64 for KV tiles. No 128/256 preload has been tested. A sweep across 64/128/256 for the MMA F16 KV load path would be a quick experiment.
- **Difficulty**: Low.

---

## 6. Build & Infrastructure

### Current State
- CMake build with CUDA 13.3, targeting `120a-real` (not `120f-virtual` — the "f" suffix conflicts with a CMake arch regex)
- Build flags: `-DGGML_CUDA_MMA=ON -DGGML_CUDA_MMVQ=ON`
- ccache enabled
- Template instances for all kernel variants are **auto-generated** by `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` — a Python script that sweeps over head sizes, quant types, and ncols configurations to produce the individual `.cu` files
- `tq` branch adds additional template instances for TBQ/TBQP/TURBO types (the "169 added files" reference)

### Optimization Candidates

#### 6a. Template Instantiation Optimization
- **What**: Reduce the number of compiled template instances to improve build time and binary size.
- **Status**: **Codegen infrastructure already exists.** The `generate_cu_files.py` script handles instance generation. The complaint is about the number of instances nvcc must compile — each `.cu` is a separate compilation unit. This is an inherent cost of CUDA template instantiation, not a tooling gap.
- **Possible improvements**: 
  - Merge related instances into fewer compilation units (e.g., one file per head size instead of one per ncols1×ncols2) to leverage nvcc's per-TU optimization. Requires CMake list changes only.
  - Use `--use_fast_math` or other per-file compilation flags to reduce register pressure (existing pattern).
- **Difficulty**: Low for script improvements; Medium for compilation-unit merging (risks hitting per-TU timeouts).
- **Applicable GPUs**: All (build time improvement only).

#### 6b. SM120-Specific Dispatch Tuning
- **What**: Add SM120-specific config tables so the dispatch picks Blackwell-tuned parameters instead of falling through to Ampere defaults.
- **Status**: **Partially implemented.**
  - **MMQ**: Already has a Blackwell config tier — `ggml_cuda_mmq_get_config_blackwell()` (`mmq-config-blackwell.cuh`) is dispatched when `blackwell_mma_available(cc)` is true (line 236 of `mmq.cuh`). Uses compact FP4 SRAM layout and MMQ_ITER_K_FP4=512.
  - **MMA F16 FA**: Falls through to Ampere config. The host dispatch (`fattn-mma-f16.cuh` line 231) checks `ampere_mma_available(cc)` first — true for SM120 — and routes to the Ampere config table unconditionally. No Blackwell-specific check exists in the MMA FA path.
  - **TILE/VEC FA**: No SM120-specific handling. Use generic Ampere configs.
- **Remaining gap**: Add Blackwell config tier(s) for MMA F16 (and potentially TILE/VEC) attention kernels. The MMQ path is already covered.
- **Difficulty**: Low for MMA F16 (new config function + dispatch check); Medium for FA kernels (requires benchmark validation of tile size choices).

#### 6c. CUTLASS DSL Integration Exploration
- **What**: Evaluate whether CUTLASS/CUTE-DSL could be used alongside llama.cpp's hand-rolled kernels for Blackwell-specific weight GEMM paths.
- **Status**: **Not explored. Zero CUTLASS dependencies in the codebase.** The only CUTLASS reference is a comment in `common.cuh` linking to NVIDIA's Blackwell SM120 GEMM documentation.
- **Difficulty**: High. Different build system, different kernel architecture, and a major philosophy shift from the independent hand-rolled kernel approach this codebase uses.
- **Note**: This would only be relevant for weight matmul paths (MMQ/MMVQ). The attention kernels (FA) are unlikely candidates since CUTLASS doesn't have an FA epilogue that matches llama.cpp's needs.
- **Evidence**: vLLM uses CUTLASS for datacenter Blackwell FP4/FP8 paths, but their SM120 consumer path remains different from their SM100 path.

---

## 7. Model-Specific Tuning

### Gemma 4 (26B MoE, D=512)
- Primary test model on this hardware.
- D=512 attention needs special handling; local TILE/MMA tables include 576x512/512x512 combos that may pragmatically exceed SM120 budget at some occupancy choices.
- Currently tested with TBQ3 KV cache at 135 t/s decode.
- Unsloth NVFP4 weights fit within 32 GB; use as calibration baseline.

### Qwen3.6
- Widely used on this hardware configuration.
- MoE architecture with MTP support.
- Public guidance and benchmarks show MTP can materially improve decode throughput on this family when implemented correctly.
- Qwen3.6-35B is likely the stronger utilization test on RTX 5090 than dense 27B because active-parameter count stays low despite larger total size.

### General Model Notes
- For long-context decode, prefer TBQ3 over TURBO3 until TURBO inner loop is fixed.
- For perplexity-sensitive use, add Hadamard rotation when cache types are quantized; local infrastructure already supports rotation matrices for quant caches.

---

## Reference Repositories

| Repo | Key Content | Relevance |
|---|---|---|
| `/mnt/storage/blackbeard` (tq branch) | TBQ/TURBO KV cache types | KV cache compression |
| `/mnt/storage/Projects/turboquant` | TheTom's optimizations | oscar2 INT2 FA, learned rotations |
| `ikawrakow/ik_llama.cpp` | Hadamard KV, IQK/Trellis quants, fused MoE | General-purpose quant/performance improvements |
| `Dao-AILab/flash-attention` (cute/) | FA4: SM100 tcgen05, SM120 SM80 fallback | Reference architecture and scheduler |
| `atcuality2021/manthanquant` | WHT-space FA, Lloyd-Max centroids | TBQ verification |
| `DrBearJew/dot4-flash-attention` | INT8 packed16, split-K, RDNA3 | AMD-only but concepts general |
| `gau-nernst/fa-5090` | SM120 FA CUDA writeup | SM120 tile tuning intuition |
| `florianmattana/fp4-fused-attention-sm120` | SM120 FP4 attention + SMEM/budget notes | SM120 accumulator/smem behavior |
| `0xSero/blackwell-gpu-wiki` | SM120 vs SM100 summary, SMEM budget page | Quick architecture reference |
| `lna-lab/blackwell-geforce-nvfp4-gemm` | RTX 5090/5080/5070Ti/RTX PRO 6000 patches for vLLM/FlashInfer/CUTLASS | SM120-specific attention/gemm patches |
| `TheTom/llama-cpp-turboquant` | Extended TurboQuant + MTP/GGUF serving path | Cross-check TurboQuant quality/serving |
| `vLLM docs/blog` | FP8 KV cache, per-head calibration docs | Public KV calibrations ecosystem |
| `tlskinner26/llama-cpp-blackwell-optimization` | CMake `120a-real` autodetect, Gemma 4 26B on 16 GB, FP4 tensor-core discovery | SM120 build/packaging recipe |
| `elsung/blackwell-llm-toolkit` | SM120-verified recipes/configs for llama.cpp/vLLM/TensorRT-LLM/LMCache, RTX PRO 6000 baseline | General SM120 deployment playbook |
| `local-inference-lab/rtx6kpro` | RTX 6000 Pro wiki for SM120: BF16 KV mandatory for GLM-5, SGLang + FlashInfer sparse MLA decode, NVFP4 notes | SM120 backend gotchas |
| `informatico-madrid/blackwell-linux-infra-optimizer` | Linux kernel 6.14 + vLLM SM120 recipe, flash-attn symbol fixes, 58.6 t/s DeepSeek-R1-32B-AWQ | Host/software stack hygiene |
| `flashrt-project/FlashRT` | SM120 quantization-alignment discussion for NVFP4 inference | Kernel porting notes |
| `Andgihat/llama-cpp-mtp-turboquant-sm120-blackwell-windows` | Windows prebuilt combining MTP + TurboQuant + native `sm_120` for RTX 50-series | Build-flag cross-check |
| `Luce-Org/lucebox` | Megakernel + DFlash speculative decoding, Blackwell `sm_120` path, 194 tok/s NVFP4 decode on GB10 | Fused speculative decode concepts |
| `gau-nernst/fa-5090` | SM120 FA writeup | Tile tuning intuition |
| `florianmattana/fp4-fused-attention-sm120` | SM120 SMEM/budget notes | Accumulator/smem behavior |
| `0xSero/blackwell-gpu-wiki` | SM120 vs SM100 summary | Architecture reference |
| `lna-lab/blackwell-geforce-nvfp4-gemm` | SM120 attention/gemm patches | Attention/gemm patches |
| `TheTom/llama-cpp-turboquant` | Extended TurboQuant + MTP/GGUF serving path | TurboQuant quality/serving |

---


## 11. Far-Fringe / Blue-Sky Ideas

These are intentionally outside the immediate roadmap. They are included because one of them could become a unlock after additional research.
The evaluations below assess feasibility and grounding against the codebase — most are system-level or model-level concepts with no CUDA kernel implications.

#### 11a. Persistent KV Replay via Memory-Mapped / io_uring Backing
- **What**: Treat KV blocks as a custom persistent object store backed by `mmap()` or `io_uring` with lifecycle longer than any single request.
- **Evaluation**: System-architecture concept, no CUDA kernel impact. The KV cache in this codebase is entirely in-memory (`ggml_backend_cuda_buffer`), with no persistence layer. Multi-turn agent KV reuse is a server/API concern, not a kernel concern. Plausible for a server wrapper but outside Blackbeard's scope.
- **Risk**: Cache invalidation when quantization, model version, or MTP draft model changes.

#### 11b. Wrapper Fan-Overclock + Power-Curve Management for Sustained SM120 Kernel Runtime
- **What**: Use `nvidia-smi` + custom fan curve + MTBF-tuned power limit to sustain boost.
- **Evaluation**: System administration, not kernel code. The observation about thermal throttling on RTX 5090 under sustained decode is valid. Not a Blackbeard deliverable — belongs in an operational runbook.
- **Caveat**: Affects benchmark reproducibility; document the power/fan config used for any published numbers.

#### 11c. DFlash / Speculative Prefill Ported to GGUF Kernel Space
- **What**: Lucebox-style DFlash/tree-based speculative decoding inside llama.cpp.
- **Evaluation**: No speculative decoding infrastructure exists in this codebase's CUDA backend (zero references to MTP, draft models, or speculative kernels). This would be a major new feature requiring new ggml ops and CUDA kernels. The reference Lucebox implementation is Python/vLLM, not directly portable to GGUF's graph model.
- **Risk**: Significant graph/backend work. Not a kernel-tuning task.

#### 11d. Automatic SM120 Tile Smoke Tests at Build Time
- **What**: Compile-time `static_assert`-style checks for FA tile shapes against 99 KB SMEM.
- **Evaluation**: **Feasible but low value.** The SMEM budget is computed at runtime in `fattn-mma-f16.cuh` (lines 1917-1927) from config-table parameters, not at compile time within the config macro. A `static_assert` could be added to the config macro, but as shown in sections 1/3/5, all Ampere configs fit within 99 KB (worst case ~38 KB). The risk of an upstream change accidentally exceeding SM120's budget is near-zero because the Ampere configs were designed for 168 KB and are far below 99 KB. Not worth the complexity.
- **Reference**: Existing `GGML_CUDA_FATTN_MMA_CONFIG_CASE` macro (line 26) validates parameter ranges but not SMEM.

#### 11e. SM120-Aware MTP Draft-Model Selection Heuristic
- **What**: Auto-choose draft-model size/batch based on SM120's tile/occupancy tradeoffs.
- **Evaluation**: Host/llama.cpp feature, not a CUDA kernel change. No MTP speculative infrastructure exists in the CUDA backend (confirmed by grep). A hypothetical MTP implementation would interact with kernel occupancy (bigger draft model = fewer active CTAs), but this is downstream of any MTP integration, not a prerequisite.

#### 11f. Long-Context Kernel Autotuning by Sequence-Length Bucket
- **What**: Pre-select decode kernel variant based on context-length bucket rather than only head/dtype.
- **Evaluation**: **Partially grounded, premise partly wrong.** The current dispatch (`ggml_cuda_get_best_fattn_kernel`) does not use sequence length — it only considers head size, KV type, and GQA ratio. This is a genuine gap: long contexts change the K/V access pattern (more tiles iterated, different cache locality).
  - However, the evidence claim "SMEM pressure as a function of sequence length" is **incorrect**. SMEM per block is determined by tile size (`nbatch_fa`, `nbatch_K2`, etc.), which is independent of total sequence length. Sequence length affects the number of K tiles iterated, not the tile dimensions. The bottleneck at long context is memory bandwidth (more K/V loaded per Q tile), not SMEM.
  - A useful tuning would be: for short contexts (< 4096), use wider tiles for better arithmetic intensity; for long contexts (128K+), use narrower tiles to reduce per-tile memory traffic and improve occupancy. This is a valid optimization.
- **Difficulty**: Medium. Requires adding context-length to the dispatch criteria and validating tile shape tradeoffs.

#### 11g. Zero-Copy Multi-Process Server Model for Long-Lived Chat Sessions
- **What**: Pinned KV region per user, mapped read-only into decoder workers via POSIX shared memory.
- **Evaluation**: System-architecture concept, no kernel impact. The codebase has no multi-process server model — KV cache is process-local `ggml_backend_cuda_buffer`. This would require a new serving layer outside llama.cpp.

#### 11h. OS-Level Inference Priority Class for SM120
- **What**: Use `chrt`, `nice`, cgroups to prevent desktop preemption.
- **Evaluation**: System administration. Valid concern — desktop GPU workloads compete with compositors. Should be documented in operational notes, not treated as a code deliverable.

---

## 12. External Source Evaluation

Results of mining the repos listed in the previous version of this section. Each was searched 2026-07-16.

| Source | What It Actually Is | Worth Mining? | Extract |
|---|---|---|---|
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | llama.cpp fork: TurboQuant WHT-rotated KV + weight compression + Gemma 4 MTP + Qwen 3.6 NextN speculative decoding. Claims +30-50% throughput. | **Yes — tq branch cross-check.** Confirms TBQ/TURBO integration path. MTP + NextN speculative patterns may be portable. | MTP draft-model batching approach, TurboQuant FA fusion pattern |
| `Indras-Mirror/llama.cpp-turboq-mtp` | llama.cpp fork: "Fused TBQ4 Flash Attention + MTP + Shared Tensors." 84 stars. Claims 82+ tok/s with lossless 4.25 bpv KV at 200K context on RTX 4090. | **Yes — top priority for tq branch.** Directly addresses the fused TBQ4 FA + MTP gap in our `tq` branch. `--shared-tensors` flag hints at memory optimization that avoids KV copies. | Fused TBQ4 FA kernel structure, shared-tensor KV path, MTP integration |
| `wildcardorbit/epic-ark-llm` | Claims "Blackbeard optimization with SM120 runtime optimizations." Could not find a repo matching this exact name. Results returned general LLM+ARK unrelated content. | **Unknown / likely non-existent or renamed.** Worth searching again with different query, but low confidence. | N/A |
| `rogerhamonassistant-ai/vllm-moet` / `Eaven/vllm-moet` | Actual active repo is `kacper-daftcode/vllm-Moet` (6 days old). vLLM v0.24.0 patch + hand-written SM120 SASS kernels for 2-bit MoE quantization + FP4 delta recovery. Runs DeepSeek-V4 on consumer Blackwell. Active Facebook/Reddit discussion. | **Yes — valuable MoE reference.** The SM120 SASS kernel approach is a proof that hand-tuning for SM120's specific instruction set matters. The 2-bit MoE + FP4 delta recovery technique is novel and may apply to our NVFP4 MoE path. | SM120 SASS kernel patterns, 2-bit MoE quantization strategy, FP4 delta recovery approach |
| `qu0b/vllm-dsv4-sm120` | Deploys DeepSeek-V4-Flash on vLLM v0.23.0 with SM120 support. ~180 tok/s warm decode, 1M context. Uses lucifer1004's SM120 kernel patches (DeepGEMM, flashinfer sparse_mla_sm120, NVFP4). | **Yes — SM120 inference stack reference.** Validates that full DeepSeek-V4 can run on SM120 with patch stack. The flashinfer sparse_mla_sm120 kernel is directly relevant to our MLA attention path. | flashinfer sparse_mla_sm120 kernel approach, DeepGEMM SM120 patches, full software stack for SM120 serving |
| `tlskinner26/llama-cpp-blackwell-optimization` | Could not locate a repo by this exact name. Related search results pointed to llama.cpp Blackwell CMake issues (sm_120 compilation, CMAKE_CUDA_ARCHITECTURES fixes) but no dedicated repo. | **Low priority / likely absorbed into upstream.** The CMake `120a-real` fix is already in our codebase. | N/A — already handled |
| `informatico-madrid/blackwell-linux-infra-optimizer` | Linux kernel 6.14 + vLLM SM120 recipe repo. Addresses kernel incompatibilities, P2P deadlocks, memory fragmentation for SM120 LLM inference. Active repo. | **Yes — host-stack reference.** The kernel parameter tuning (P2P fix, memory fragmentation workaround) directly affects stability of long-running inference on our RTX 5090. | Kernel boot parameters for SM120, P2P deadlock workaround, memory fragmentation mitigations |
| `local-inference-lab/rtx6kpro` | RTX PRO 6000 wiki covering all SM120 aspects. Includes: BF16 KV mandatory for GLM-5 on SM120, SGLang + FlashInfer + vLLM configs, FlashInfer FA2 BF16 MLA kernel (SM120-specific), speculative decoding (MTP/EAGLE/DFlash) notes, common SM120 issues. | **Yes — essential reference.** Most comprehensive single source for SM120 inference gotchas. The "BF16 KV mandatory" finding directly affects our KV cache strategy for long-context models. FlashInfer SM120 MLA kernel is a reference for our attention paths. | BF16 KV requirement for SM120, FlashInfer SM120 FA2 kernel patterns, SM120-specific SGLang/vLLM backend configs, DFlash speculative decode notes |
| `Luce-Org/lucebox` + `jiewuxue/luce-megakernel` | Lucebox: Fast LLM speculative inference server with custom kernels. luce-megakernel: hand-tuned CUDA kernels for specific consumer hardware. DFlash speculative decoding achieves ~2× throughput. Has megakernel concept (single CUDA dispatch for all 24 layers of Qwen 3.5). | **Yes — DFlash/megakernel reference.** The DFlash speculative decoding approach is directly applicable to our MTP path. The megakernel concept (single dispatch for full model) is novel but likely too invasive for our fork. | DFlash kernel structure, persistent decode grid (one block per SM), speculative prefill patterns |
| `gau-nernst/fa-5090` | Blog post + code: "Writing Speed-of-Light Flash Attention for 5090 in CUDA C++." Detailed walkthrough of implementing SM120 FA from scratch. Covers warp-level tiling, `mma.sync`, SMEM budgeting, and occupancy tuning. | **Yes — top priority for FA tuning.** This is the single most directly applicable reference for our FA kernel work. It provides concrete tile-size guidance for SM120's 99 KB SMEM and documents the exact instruction choices available on SM120. | SM120 tile-size recommendations, occupancy vs SMEM tradeoff data, mma.sync scheduling patterns |
| `florianmattana/fp4-fused-attention-sm120` | Fused FP4 attention kernel for SM120 written entirely in inline PTX. Uses warp-level `mma.sync` with FP4 E2M1. Includes detailed readme on SM120's lack of tcgen05/TMEM. Has companion blog post. | **Yes — direct reference for our FP4 FA kernel.** Directly comparable to our `fattn-mma-fp4.cuh`. The inline PTX approach for FP4 MMA on SM120 may reveal optimization opportunities our higher-level implementation misses. | FP4 fragment layout for SM120 mma.sync, SMEM budget for FP4 tiles, inline PTX patterns for FP4 MMA |
| `lna-lab/blackwell-geforce-nvfp4-gemm` | SM120 patches for vLLM + FlashInfer + CUTLASS covering NVFP4 inference on RTX 5090/5080/5070 Ti / RTX PRO 6000. Claims 175 tok/s on Qwen3.6-35B MoE. 12 SM120 patches covering FP4 MoE, FlashInfer, and CUTLASS integration. | **Yes — SM120 NVFP4 reference.** The 12-patch set covers the exact SM120 NVFP4 stack we use. The FlashInfer SM120 FA path and CUTLASS NVFP4 GEMM tile selections are directly relevant to our MMQ NVFP4 path. | CUTLASS NVFP4 tile sizes for SM120, FlashInfer SM120 FA integration patterns, SM120-specific NVFP4 MoE GEMM config |
| `0xSero/blackwell-gpu-wiki` | Dedicated SM120 knowledge base covering: SM120 vs SM100 architectural differences, SMEM budget page, FlashInfer on SM120 (Triton + CUTLASS paths), CUDA toolkit compatibility matrix. Also has `deepseek-v4-flash-sm120` runtime patch repo. | **Yes — architecture reference.** The SM120 vs SM100 comparison and SMEM budget data are directly relevant to our dispatch/tuning decisions. Less actionable than the kernel repos above, but necessary context. | SM120 SMEM cliff documentation, SM120 vs SM100 ISA differences, CUDA toolkit version compatibility |

**Top 5 to mine first:** `gau-nernst/fa-5090` (FA tile tuning), `florianmattana/fp4-fused-attention-sm120` (FP4 FA reference), `Indras-Mirror/llama.cpp-turboq-mtp` (fused TBQ4), `lna-lab/blackwell-geforce-nvfp4-gemm` (NVFP4 SM120 patches), `local-inference-lab/rtx6kpro` (SM120 ops wiki).

---

## 13. Repositories and Sources

| Repo | Key Content | Relevance |
|---|---|---|
| `/mnt/storage/blackbeard` (tq branch) | TBQ/TURBO KV cache types | KV cache compression |
| `/mnt/storage/Projects/turboquant` | TheTom's optimizations | oscar2 INT2 FA, learned rotations |
| `ikawrakow/ik_llama.cpp` | Hadamard KV, IQK/Trellis quants, fused MoE | General-purpose quant/performance improvements |
| `Dao-AILab/flash-attention` (cute/) | FA4: SM100 tcgen05, SM120 SM80 fallback | Reference architecture |
| `atcuality2021/manthanquant` | WHT-space FA, Lloyd-Max centroids | TBQ verification |
| `DrBearJew/dot4-flash-attention` | INT8 packed16, split-K, RDNA3 | AMD-only but concepts general |
| `gau-nernst/fa-5090` | SM120 FA CUDA writeup | SM120 tile tuning intuition |
| `florianmattana/fp4-fused-attention-sm120` | SM120 FP4 attention + SMEM/budget notes | Accumulator/smem behavior |
| `0xSero/blackwell-gpu-wiki` | SM120 vs SM100 summary, SMEM budget page | Quick architecture reference |
| `lna-lab/blackwell-geforce-nvfp4-gemm` | RTX 5090/5080/5070Ti/RTX PRO 6000 patches for vLLM/FlashInfer/CUTLASS | SM120-specific attention/gemm patches |
| `TheTom/llama-cpp-turboquant` | Extended TurboQuant + MTP/GGUF serving path | Cross-check TurboQuant quality/serving |
| `vLLM docs/blog` | FP8 KV cache, per-head calibration docs | Public KV calibrations ecosystem |
| `tlskinner26/llama-cpp-blackwell-optimization` | CMake `120a-real` autodetect, Gemma 4 26B on 16 GB, FP4 tensor-core discovery | SM120 build/packaging recipe |
| `elsung/blackwell-llm-toolkit` | SM120-verified recipes/configs for llama.cpp/vLLM/TensorRT-LLM/LMCache, RTX PRO 6000 baseline | General SM120 deployment playbook |
| `local-inference-lab/rtx6kpro` | RTX 6000 Pro wiki for SM120: BF16 KV mandatory for GLM-5, SGLang + FlashInfer sparse MLA decode, NVFP4 notes | SM120 backend gotchas |
| `informatico-madrid/blackwell-linux-infra-optimizer` | Linux kernel 6.14 + vLLM SM120 recipe, flash-attn symbol fixes, 58.6 t/s DeepSeek-R1-32B-AWQ | Host/software stack hygiene |
| `flashrt-project/FlashRT` | SM120 quantization-alignment discussion for NVFP4 inference | Kernel porting notes |
| `Andgihat/llama-cpp-mtp-turboquant-sm120-blackwell-windows` | Windows prebuilt combining MTP + TurboQuant + native `sm_120` for RTX 50-series | Build-flag cross-check |
| `Luce-Org/lucebox` | Megakernel + DFlash speculative decoding, Blackwell `sm_120` path, 194 tok/s NVFP4 decode on GB10 | Fused speculative decode concepts |
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | Fused `TurboFlash` FA + MTP for llama.cpp | TurboQuant attention integration |
| `Indras-Mirror/llama.cpp-turboq-mtp` | Fused TBQ4 Flash Attention + MTP llama.cpp fork | TBQ4 FA integration concepts |
| `wildcardorbit/epic-ark-llm` | Blackbeard runtime optimizations and SM120 performance enhancements | Exact-target comparisons |
| `Eaven/vllm-moet` / `rogerhamonassistant-ai/vllm-moet` | Frontier MoE on RTX PRO 6000 / RTX 5090 with SM120 SASS | MoE shape/scale on consumer Blackwell |
| `qu0b/vllm-dsv4-sm120` | DeepSeek-V4-Flash on SM120 vLLM | KV/MLA/FP4 SM120 patterns |
| `jasl/vllm-ds4-sm120-harness` | SM120 FlashInfer + adapter evidence | SM120 sparse-MLA adapter cues |
| `local-inference-lab/blackwell-llm-docker` | Dockerboards with source-built FlashInfer + PR2913 for SM120 | Deployment/kernel source cues |
| `ggml-org/llama.cpp` discussion #20969 | Canonical TurboQuant discussion thread with CLI/type integration spec | TurboQuant integration spec |
| `ggml-org/llama.cpp` issue #23693 | RTX 5060 Ti `sm_120` `BLACKWELL_NATIVE_FP4=1` kernel/attention failure case | Consumer Blackwell failure modes |
| `Dao-AILab/flash-attention` issue #1665 | SM120 usability discussion | SM120 porting lessons |
| SGLang issue #19637 | SM120 performance optimization plan | SM120 optimization roadmap |
| `sorryhyunblog.vercel.app` | FlashAttention-4 SM120 implementation notes from RTX 5060 Ti (`sm_120`) | Consumer SM120 FA4 lessons |

---

## 14. Benchmarks to Capture Before Claiming Wins

For every optimization candidate in this document, the empirical proof must include:

1. tokens-per-second decode at bs=1, 6 prompt/128 output, or equivalent consistent harness
2. VRAM usage at target context length
3. SM120-specific capture from `ncu`/`nvprof`: SMEM usage per block, register pressure, replay overhead
4. comparative result at D=512, since local baseline already boarded this shape
5. memory-bound percentage estimate from occupancy/warp state profiler

Any optimization that cannot satisfy this packet is not "done", no matter how elegant the patch.

## 15. Research Loop Notes (remove once exhausted)

This section documents active/ongoing search themes so searches are not needlessly repeated:

- FlashAttention-4 SM120 consumer ports: `sorryhyunblog`, GitHub forks, and alternative `sm_120` FA4 attempts.
- llama.cpp forks combining MTP + TurboQuant + Blackwell: multiple repos found in 2026; evaluate kerneling/merge opportunities.
- Frontier MoE on SM120: `vllm-moet`/`vllm-dsv4-sm120` show that DeepSeek-V4/GLM-5-class models can run on consumer Blackwell with significant external patching; useful pattern database, not a local merge target.
- Local inference Google/Jax/OSS: `wildcardorbit/epic-ark-llm` has explicit Blackbeard focus; validate before mining.
- SM120 Docker/infra recipes: `blackwell-llm-docker` and `blackwell-linux-infra-optimizer` both emphasize host stack and kernel/runtime prerequisites for stable SM120 inference.
