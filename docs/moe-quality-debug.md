# MoE Quality Degradation Investigation

**Date:** 2026-07-30
**Build:** `b583-b64266fcd`
**Hardware:** NVIDIA RTX 5090 (SM120, 32 GB)
**Branch:** `main` (private/main)

## Overview

Running MoE (Mixture of Experts) quantized models on NVIDIA Blackwell GPUs (RTX 5090) produces complete gibberish output when all layers are offloaded to GPU (`-ngl 99`). Non-MoE dense models work perfectly. The corruption starts during prefill (the very first generated token is wrong) and compounds progressively with more GPU layers.

## Symptoms

| Model | Type | Quant | ngl | KV Cache | Status |
|---|---|---|---|---|---|
| Qwen3-8B | Dense | Q8_0 | 99 | default | ✅ Coherent |
| Qwen3.6-27B-UD | Dense | Q5_K_XL | 90 | default | ✅ Coherent |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 0 | default | ✅ Coherent (CPU) |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 1 | default | ✅ Coherent |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 10 | default | ✅ Coherent |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 15 | default | ❌ `"Trader Trader..."` |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 20 | default | ❌ `"(wrapper..."` |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 30 | default | ❌ `"escaped..."` |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 50 | default | ❌ `"ASFASF..."` |
| Qwen3-Coder-30B-A3B | **MoE** | Q4_K_XL | 99 | default | ❌ `"kto B BPentity..."` |
| Qwen3.6-27B-A3B | **MoE** | Q5_K_XL | 30 | q4_0 | ❌ Crash: `misaligned address` |
| Qwen3.6-27B-A3B | **MoE** | Q5_K_XL | 30 | q8_0 | ✅ Coherent |
| Qwen3.6-27B-A3B | **MoE** | Q5_K_XL | 99 | — | ❌ OOM (27B > 32 GB) |
| Qwen3.6-35B-A3B | **MoE** | Q6_K_XL | 30 | q4_0 | ❌ Crash: `misaligned address` |

**Key patterns:**
- Quality break threshold: between `-ngl 10` and `-ngl 15` for the 30B MoE model
- Gets progressively worse with more GPU layers (cumulative corruption)
- Non-MoE dense models work perfectly regardless of quant type
- First generated token is already corrupted → **prefill-phase bug**, not decode

## Applied Fix (Working Tree)

### 1. Q4_0 KV Cache `misaligned address` Crash

**File:** `ggml/src/ggml-cuda/quantize.cu`

**Problem:** The `quantize_mmq_q8_1` and `quantize_mmq_q8_1_scatter` kernels cast `const float *` to `const float4 *` without checking 16-byte alignment. On Blackwell SM120+, unaligned `float4` loads cause `misaligned address` CUDA errors when using `--cache-type-k q4_0 --cache-type-v q4_0`.

**Fix:** Added runtime alignment checks — if the source pointer is 16-byte aligned, use `float4` load (fast path); otherwise fall back to 4 scalar `float` loads (safe path).

**Status:** ✅ Applied. Fixes the KV cache crash on Q5_K/Q6_K MoE models at `-ngl 30`. Still in working tree (not committed).

## Ruled Out (7 Independent Diagnostic Tests)

All tests performed on the Q4_K_XL 30B MoE model at `-ngl 99` with `-p 'write a hello world python script'`.

### Test 1: Blackwell vec_dot → Generic vec_dot
**Change:** `vec_dot_q4_K_q8_1_blackwell` → `vec_dot_q4_K_q8_1`
**Result:** ❌ Still gibberish (`"brasbrasbras..."`)
**Conclusion:** The Blackwell-specific Q4_K vec_dot function is NOT the cause.

### Test 2: Q4_K nwarps=1 → nwarps=4
**Change:** `calc_nwarps` for Q4_K at ncols_dst=1: 1 → 4
**Result:** ❌ Still gibberish
**Conclusion:** Single-warp reduction for Q4_K is NOT the cause.

### Test 3: rows_per_block=2 → rows_per_block=1
**Change:** `calc_rows_per_block` for Blackwell ncols_dst=1: 2 → 1
**Result:** ❌ Still gibberish
**Conclusion:** Double-row blocks for Blackwell decode are NOT the cause.

### Test 4: min_blocks_per_sm=4 → min_blocks_per_sm=1
**Change:** `get_min_blocks_per_sm` for Blackwell: 4 → 1
**Result:** ❌ Still gibberish
**Conclusion:** Aggressive register allocation hints are NOT the cause.

### Test 5: VDR=4 → VDR=2
**Change:** `VDR_Q4_K_Q8_1_MMVQ_BLACKWELL` (4) → `VDR_Q4_K_Q8_1_MMVQ` (2)
**Result:** ❌ Still gibberish
**Conclusion:** Higher vector dot reduction factor is NOT the cause.

### Test 6: generic mul_mat_vec_q → MoE kernel
**Change:** Routed single-token MoE decode through `mul_mat_vec_q_moe` kernel instead of generic `mul_mat_vec_q`
**Result:** ❌ Still gibberish
**Conclusion:** The shared-memory reduction in `mul_mat_vec_q` is NOT the cause — even the warp-only MoE kernel produces wrong results.

### Test 7: ALL Blackwell MMVQ/vecdotq changes disabled simultaneously
**Change:** Reverted all five Blackwell-specific Q4_K changes at once (nwarps=4, rows_per_block=1, min_blocks=1, generic vec_dot, VDR=2)
**Result:** ❌ Still gibberish (`"brasbrasbras..."`)
**Conclusion:** None of the Blackwell-specific MMVQ/vecdotq optimizations are the root cause.

### 🔑 Test 8: Force cublas FP32 fallback for ALL MoE matmuls
**Change:** `get_mmvq_mmid_max_batch` returns 0, forcing MUL_MAT_ID to skip MMVQ and MMQ, fall through to cublas FP32 gemm. Also disables CUDA graphs for MUL_MAT_ID.
**Result:** ❌ **Still gibberish** (`"articulated_colourogISC Thai..."`)
**Conclusion:** Even NVIDIA's own cublas FP32 BLAS produces wrong results. This proves the **inputs to MUL_MAT_ID (activation tensor and expert routing IDs) are already corrupted** before any matmul happens. The bug is NOT in any matmul kernel.

## Current Hypothesis: MoE Routing Pipeline

Since the matmul inputs are corrupted before they reach any matmul kernel (MMVQ, MMQ, or cublas), the bug must be in the MoE prefill pipeline that produces those inputs:

```
Gate Projection (dense matmul)
    ↓
Softmax (routing probabilities)
    ↓
Top-K / Argsort (expert indices)
    ↓
[Expert indices fed as `ids` to MUL_MAT_ID]
    ↓
MUL_MAT_ID (matmul) ← INPUTS ALREADY CORRUPTED HERE
```

### Possible Root Causes (ordered by likelihood):

1. **Top-K routing kernel produces wrong expert indices on SM120**
   - The `topk_moe_cuda` kernel in `ggml/src/ggml-cuda/topk-moe.cu` uses `__shfl_down_sync` warp operations
   - SM120 may have different warp shuffle behavior or require different launch bounds
   - No Blackwell-specific code in this kernel, but could be a CUDA compiler bug on SM120

2. **Graph construction produces wrong tensor strides for MoE MUL_MAT_ID**
   - The `ggml_cuda_topk_moe_fusion` function in `ggml/src/ggml-cuda/ggml-cuda.cu` fuses softmax→argsort→get_rows
   - If the fusion misidentifies tensor relationships, wrong `ids` or `weights` tensors could be passed

3. **Q8_1 activation quantization for MoE intermediate activations**
   - The `quantize_row_q8_1_cuda` function in `ggml/src/ggml-cuda/quantize.cu` quantizes FP32 activations to Q8_1
   - MoE activations may have different strides/dimensions that trigger a quantization bug

4. **SM120-specific CUDA compiler bug**
   - The kernel code is correct and works on other architectures
   - NVIDIA's SM120 compiler may generate incorrect code for certain patterns

## Files of Interest

| File | Relevance |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cu` | MMVQ kernel dispatch, Blackwell parameter table (RULED OUT) |
| `ggml/src/ggml-cuda/vecdotq.cuh` | Vec_dot functions, VDR definitions (RULED OUT) |
| `ggml/src/ggml-cuda/quantize.cu` | Q8_1 quantize, Q4_0 alignment fix (PARTIALLY FIXED) |
| `ggml/src/ggml-cuda/mmq.cuh` | MMQ config dispatch, Blackwell MMQ config (POSSIBLY RELEVANT) |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | Blackwell-specific MMQ tile size config |
| `ggml/src/ggml-cuda/topk-moe.cu` | Top-K MoE routing kernel (**PRIMARY SUSPECT**) |
| `ggml/src/ggml-cuda/topk-moe.cuh` | Top-K MoE kernel declaration |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | MoE graph fusion, MUL_MAT_ID dispatch |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | Flash attention Blackwell config (unlikely, dense works) |

## Recommended Next Steps

### Immediate
1. **Git bisect** between known-bad HEAD (`b64266fcd`) and the commit before the first Blackwell change (`775d1e6fd^`). Test with `-ngl 15 -n 1 -p 'write a hello world python script'` — check if first token is coherent.

2. **Compute-sanitizer** on Blackwell hardware:
   ```bash
   compute-sanitizer --tool memcheck ./build-cuda/bin/llama-cli \
     -m /mnt/storage/models/qwen3/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
     -ngl 15 -t 8 -c 2048 -n 4 -p 'write a hello'
   ```
   This catches misaligned accesses, out-of-bounds reads/writes, and race conditions.

### After Root Cause Identified
3. **Bypass topk-moe fusion** — force softmax and argsort to run as separate kernels instead of the fused `topk_moe_cuda` kernel. If quality recovers, the fusion kernel has a bug.

4. **Compare routing IDs** between CPU (`-ngl 0`) and GPU (`-ngl 15`) for the first MoE layer. If the expert indices differ, the top-k routing is the root cause.

5. **Test topk-moe with modified launch bounds** — add `__launch_bounds__` with conservative values to work around potential SM120 compiler bugs.

## Commit Series Reference

```
b64266fcd docs : README upstream comparison
65ad4e744 docs : __launch_bounds__(32,4) win, final session benchmarks
b6b784ac9 cuda : __launch_bounds__(32,4) for Blackwell MMVQ decode
92ba0ea3d docs : final benchmark snapshot
4cfa16f82 docs : complete session findings
d3dcdc85f docs : final benchmarks — +1.9% Q4_K decode
26ae4d87a cuda : __ldg q8_1 reads in NVFP4 Blackwell vec_dot (neutral)
3e376d2c7 cuda : __ldg q8_1 reads in Q4_K Blackwell vec_dot (+1.1% tg)
d62e7a96c cuda : nwarps=1 for Q4_K decode on Blackwell (neutral, infra)
be7d01651 cuda : nwarps=1 for NVFP4 decode on Blackwell (+5.2% tg)
ea6b19c51 cuda : VDR=4 Q4_K decode for Blackwell (infrastructure, neutral)
fe1aa1a1c cuda : add Blackwell MMVQ parameter table (rows_per_block=2, nwarps=8 batch)
a3da73bf0 cuda : iter_k=8192 for Q4_K/Q5_K (found ceiling at 4096)
7f15a43bf cuda : iter_k=4096 for Q4_K/Q5_K, expand 2048
afd2d89c1 cuda : NVFP4 at iter_k=1024, Q4_K/Q5_K at iter_k=2048
775d1e6fd cuda : extend Blackwell MMQ config to Q4_K and Q5_K with iter_k=512 ← FIRST Blackwell commit
```

## Working Tree Status

```
$ git diff --stat
ggml/src/ggml-cuda/quantize.cu | 34 ++++++++++++++++++++++++++++------
1 file changed, 28 insertions(+), 6 deletions(-)
```

Only the Q4_0 KV cache alignment fix is uncommitted. All other diagnostic changes have been reverted.
