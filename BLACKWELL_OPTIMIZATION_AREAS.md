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
- **Status**: Not implemented. TILE/VEC/MMA all use static grid launch.
- **Difficulty**: Medium. New dispatch pattern, doesn't need new instructions.
- **Reference**: CUTLASS Stream-K / `PersistentTileSchedulerSm100StreamK`, FA4 CLC scheduler pattern.
- **Applicable GPUs**: All CUDA GPUs. No SM120 dependency.
- **Note**: Prefill-heavy Gemma 4 / Qwen workloads benefit first.

#### 1b. Split-KV Decode (Long Context)
- **What**: Distribute K rows across multiple CTAs for long-context decode. Each CTA processes a chunk of K rows; results merged via online softmax.
- **Status**: Partial. VEC kernel handles decode but uses single CTA per head. split-K / merge-phase patterns exist in reference implementations.
- **Difficulty**: Medium. Changes the VEC kernel launch config and adds merge phase.
- **Reference**: FA4 split-K decode, DRBearJew dot4 split-K merge.
- **Applicable GPUs**: All. Critical for 128K+ context on any GPU, especially memory-bound decode on 5090's HBM.

#### 1c. Tile Size Auto-Tuning for 99 KB SMEM
- **What**: The TILE and MMA kernels use fixed tile sizes inherited from upstream. SM120's 99 KB SMEM means:
  - Smaller Q-tiles or fewer pipeline stages fit.
  - Accumulator register pressure is the practical limit, not just raw SMEM bytes.
  - Need to re-derive optimal tile sizes for D=64/128/256/512 on SM120.
- **Status**: Untuned. Inherits upstream defaults designed for larger SMEM.
- **Difficulty**: Low. Parameter sweep in `fattn.cuh`. Config tables exist in `fattn-tile.cuh` as compile-time CASE rows.
- **Applicable GPUs**: SM120-specific. Critical tuning knob.
- **Evidence**: `tasks/tile` config table already enumerates D=512/ncols combos, but without SM120 validation. Primary public SM120 kernel efforts also converge on smaller tiles.

#### 1d. Warp Specialization (without wgmma)
- **What**: Split warp groups into orchestrator (issue loads/mma) and compute (softmax). Can be done with `mma.sync` + `cp.async`, though less effective than with TMA.
- **Status**: Not implemented. All warps are symmetric.
- **Difficulty**: High. Major kernel restructuring.
- **Reference**: FA3/FA4 warp specialization pattern.
- **Applicable GPUs**: Any with `cp.async` (Turing+). Diminishing returns without TMA on SM120.

#### 1e. VEC Kernel Inner-Loop Optimization (for Quantized KV)
- **What**: The TURBO3 kernel (`fattn-turbo.cuh`) does per-element byte reads in the KQ loop. Replace with 128-bit vector loads packed into registers before unpacking indices.
- **Status**: Identified, not implemented. The inner loop has uncoalesced global memory access.
- **Difficulty**: Low-Medium. Localized change in the KQ and V dequant inner loops of `fattn-turbo.cuh`.
- **Applicable GPUs**: All, but critical for TURBO KV cache performance.

#### 1f. D=512 Tile Instance Validation on SM120
- **What**: Gemma 4 uses D=512 attention. The existing TILE/MMA config tables include 512x512 rows, but they were not validated against SM120's 99 KB SMEM and occupancy.
- **Status**: Untested for this repo/hardware combo.
- **Difficulty**: Low. Reproducible parameter sweep for D=512 across `fattn-tile.cuh` and `fattn-mma-f16.cuh` config tables.
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
  - `prmt.b32` register-only LUT for FP4 lookup
  - W4A8 and W4A16 paths
  - VDR=8 unrolled kernel
- Standard llama.cpp quant types (Q4_0 through IQ4_NL, k-quants)
- Dequantize to fp16 for compute (W4A16)

### Optimization Candidates

#### 3a. SM120-Aware NVFP4 Kernel Tuning
- **What**: The current NVFP4 path is usable on SM120, but SM120 shares the SM80/WGMLA warping model, not SM100's `tcgen05.mma`/TMA. Public SM120 efforts show that shared-memory overflow is the practical limit, not instruction availability.
- **Status**: Present in main, not specifically tuned for SM120 SMEM ceiling.
- **Difficulty**: Medium. Needs measured tile sweeps, not necessarily new ISA.
- **Evidence**: public SM120 NVFP4 efforts identify the issue as "autotuner skips SM120 GEMM tiles because they overflow 99KB shared memory"; same condition likely constrains our VDR selection.
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

#### 3e. Benchmark-Validated VDR Selection
- **What**: On SM120, larger block/pipeline choices can exceed shared memory. Validate smallest viable VDR first, then build up.
- **Status**: Not benchmarked on this hardware.
- **Difficulty**: Medium.
- **Evidence**: public SM120 efforts recommend starting from smaller shared-memory tiles and growing only when utilization is confirmed.

---

## 4. MoE Optimization

### Current State
- Standard llama.cpp MoE dispatch
- GPU-offloaded experts, CPU fallback
- CUDA top-k MoE softmax/argmax kernel exists locally in `ggml/src/ggml-cuda/topk-moe.cu`, including support for common expert counts.

### Optimization Candidates

#### 4a. Fused MoE Operations (from ikawrakow)
- **What**: Fused gating + expert computation + reduction to reduce kernel launch overhead and improve SM utilization.
- **Status**: Not ported.
- **Difficulty**: High. Requires new ggml ops.
- **Reference**: ikawrakow/ik_llama.cpp PR 229, Feb 2025, fused FFN ops for MoE.

#### 4b. Expert Parallel Dispatch
- **What**: Parallelize expert computation across SMs when a MoE layer has fewer active experts than SMs.
- **Status**: Not implemented.
- **Difficulty**: Medium. Scheduling change within existing matmul kernels.

#### 4c. Dynamic Expert Offloading (Auto-Fit)
- **What**: Automatically determine which experts/layers to offload to GPU based on available VRAM.
- **Status**: Not ported.
- **Difficulty**: Medium. Affects model loading and scheduling.
- **Reference**: ikawrakow PR 1501, 1504, 1872.

#### 4d. SM120 MoE Kernel Shape Validation
- **What**: At small batch, MoE inference on RTX 5090 can be scattering-bound. Validate tile shapes for common expert widths on SM120.
- **Status**: Not benchmarked locally.
- **Difficulty**: Medium.
- **Note**: Qwen3.6-35B MoE decode with MTP already shows MoE-style behavior worth measuring.

---

## 5. Memory & Data Movement

### Current State
- Standard `cp.async` usage where applicable
- No TMA (unsupported on SM120)
- 99 KB shared memory constraint
- Local `cp.async` layer exposes 16-byte copies with optional 64/128/256B preload in `ggml/src/ggml-cuda/cp-async.cuh`.

### Optimization Candidates

#### 5a. Shared Memory Budget Optimization
- **What**: Audit every kernel's SMEM usage against 99 KB limit. On SM120, accumulator placement is the real constraint, not raw SMEM bytes.
- **Status**: Untuned for SM120.
- **Difficulty**: Low. Audit + adjust tile sizes and pipeline stages.
- **Applicable GPUs**: SM120-specific.

#### 5b. V-GMEM Transpose for Better Coalescing
- **What**: Store V cache in transposed layout (d x page_size instead of page_size x d) so V loads during attention are coalesced.
- **Status**: Not implemented in this repo. V is stored same layout as K. FA4 paged attention uses V-transpose on SM100.
- **Difficulty**: Medium-High. Changes KV cache layout, allocation, copy paths, and all V read paths.
- **Reference**: FA4 `paged_kv.py` V-transpose concept for SM100; applicability to SM120 VEC/TILE kernels needs local validation.

#### 5c. Async Prefetch Sizing for SM120
- **What**: Use `cp.async` to prefetch quantized KV cache blocks before the warp needs them. Overlap global load latency with computation.
- **Status**: Basic cp.async exists. Can be extended with smaller/chunked prefetch tuned for SM120 memory controller behavior.
- **Difficulty**: Medium. Pipeline depth tuning.
- **Note**: Larger 256B preload may not win on consumer memory subsystem compared to datacenter SM100 paths. Validate 64B/128B modes.

#### 5d. SMEM/L1 Carveout for FA Kernels
- **What**: Use runtime shared-memory carveout controls to move capacity between L1 and SMEM depending on kernel shape.
- **Status**: Not used in repo FA kernels.
- **Difficulty**: Low.
- **Reference**: NVIDIA Blackwell tuning guide section on unified shared memory/L1/texture, `cudaFuncSetAttribute` with preferred shared-memory carveout on Blackwell.
- **Applicable GPUs**: SM100/SM120. Datacenter B200 uses same 256 KB combined limit; on consumer SM120, the practical total and SMEM limit are both smaller.

#### 5e. Prefetch Chunk Size Audit
- **What**: Validate `cp.async` preload sizes against SM120 bandwidth/latency rather than copying SM100 assumptions.
- **Status**: Not audited locally.
- **Difficulty**: Low-Medium.

---

## 6. Build & Infrastructure

### Current State
- CMake build with CUDA 13.3
- `-DGGML_CUDA_MMA=ON -DGGML_CUDA_MMVQ=ON`
- SM120a target configured
- ccache enabled
- `tq` branch adds 169 FA template instance files on top of existing templates.

### Optimization Candidates

#### 6a. Template Instantiation Optimization
- **What**: The 169 added template instances for TBQ/TBQP/TURBO FA kernels inflate build time and binary size. A codegen script or selective instantiation could reduce this.
- **Status**: Actualized as explicit `.cu` files on `tq`.
- **Difficulty**: Low. Automation/scripting.
- **Applicable GPUs**: All (build time improvement only).

#### 6b. SM120-Specific Autotuner/SELECT-Dispatch Tuning
- **What**: Hook compute-capability dispatch so SM120 picks validated tile tables first, then falls back through proven smaller configs.
- **Status**: Not implemented.
- **Difficulty**: Medium.
- **Evidence**: public SM120 systems report skipped auto-tuned tiles because default candidates exceed 99KB SMEM before a valid SM120 tile is tested.

#### 6c. CUTLASS DSL Integration Exploration
- **What**: Evaluate whether CUTLASS/CUTE-DSL could be used alongside llama.cpp's hand-rolled kernels for Blackwell-specific weight GEMM paths. This is how vLLM does some NVFP4/FP4 stacks.
- **Status**: Not explored.
- **Difficulty**: High. Different build system and kernel architecture.
- **Note**: Major philosophical shift from independent kernel philosophy. Recommend exploration only for weight matmul, not attention.
- **Evidence**: vLLM uses this stack for datacenter Blackwell; SM120 consumer path remains different.

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

---

## Priority Recommendations

**Tier 1 (High Impact, Low-Medium Effort):**
1. **TURBO kernel inner loop** — pack 128-bit K/V loads in `fattn-turbo.cuh` (2a)
2. **Tile size tuning for 99 KB SMEM** — parameter sweep, especially D=512 (1c + 1f)
3. **Split-KV decode** — long-context decode boost (1b)

**Tier 2 (High Impact, Medium Effort):**
4. **Hadamard K/V-cache** — generic quality boost for existing quant types (2b)
5. **CP_ASYNC prefetch sizing for SM120** — better decode bandwidth utilization (5c/5e)
6. **Low-perplexity Q4_0 KV** — better Q4_0 cache quality (2d)

**Tier 3 (Research / Long-Term):**
7. **INT2 / OSCAR-class KV + dedicated FA** — offline rotation + 2-bit cache (2e/2f)
8. **V-GMEM transpose for V cache** — long-context coalescing win if paged cache lands (5b)
9. **WHT-space fused attention** — ManthanQuant cleanup path if WHT-based cache evolves (2g)
10. **IQK/Trellis quants** — new weight quant types for future accuracy/performance frontier (3b/3c)
