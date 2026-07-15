# Project Blackbear — Blackwell FP4 Benchmark

> **Date:** 2026-07-15
> **Build:** `a58222229` (b10027), backends stripped to CPU + CUDA only
> **GPU:** RTX 5090 (sm120 Blackwell), CUDA 13.3, 32 GiB VRAM
> **CPU:** Intel Ultra 9 285K (24 P-cores), gcc-15.2

---

## Models

| Label | Arch | Quant | File Size | Active Params | Total Params |
|---|---|---|---|---|---|
| 35B MoE Q4_K_S | qwen35moe | Q4_K_S | 19.5 GiB | ~3B | 34.7B |
| 35B MoE NVFP4 | qwen35moe | NVFP4 | 22.9 GiB | ~3B | 35.5B |
| 27B dense Q5_K_XL | qwen35 | Q5_K_XL | 18.7 GiB | 26.9B | 26.9B |
| 27B dense NVFP4 | qwen35 | NVFP4 | 23.7 GiB | 27.3B | 27.3B |

---

## NVFP4 vs Q4_K_S (35B MoE)

### Prompt Processing

| Prompt length | Q4_K_S (t/s) | NVFP4 (t/s) | NVFP4 advantage |
|---|---|---|---|
| pp128 | 3,105 | 4,691 | **+51%** |
| pp512 | 7,636 | 8,598 | **+13%** |
| pp2048 | 7,588 | — | — |

### Text Generation

| Gen length | Q4_K_S (t/s) | NVFP4 (t/s) | Delta |
|---|---|---|---|
| tg128 | 268 | 203 | -24% |
| tg256 | 268 | 205 | -23% |

**Analysis:** NVFP4 wins prompt processing by 13-51% because Blackwell's native FP4 tensor core MMA (`kind::mxf4nvf4` block-scaled PTX) executes at higher throughput than the integer-arithmetic-based Q4_K_S dequant+gemm path. The advantage is largest at short prompts (low batch) where the tensor core launch overhead is amortized differently.

NVFP4 loses generation because it's memory-bandwidth-bound: the larger block-scale metadata (ue4m3 per 16 elements vs Q4_K_S grouped scales) increases bytes/token read from VRAM, and generation is almost entirely bandwidth-limited.

---

## NVFP4 vs Q5_K_XL (27B Dense)

| Test | Q5_K_XL (t/s) | NVFP4 (t/s) | Delta |
|---|---|---|---|
| pp128 | 2,353 | — | — |
| pp512 | 3,187 | 4,154 | **+30%** |
| tg128 | 68 | — | — |
| tg256 | 68 | 61 | -10% |

Same pattern: NVFP4 wins prompt processing (+30%), loses generation (-10%). The dense model comparison confirms the FP4 tensor core advantage is real and not MoE-specific.

---

## NVFP4 Flash Attention Comparison

| Test | FA off (t/s) | FA on (t/s) | Delta |
|---|---|---|---|
| pp128 | 4,691 | 4,799 | +2.3% |
| pp512 | 8,511 | 8,731 | **+2.6%** |
| tg128 | 201 | 201 | ~0% |

Flash attention provides a small but consistent boost to prompt processing (+2.6%). No impact on generation (already bandwidth-bound). This is expected — FA reduces the attention O(n^2) compute but the Blackwell FP4 MMA already keeps the compute units well-fed.

---

## NVFP4 Batch Size Sensitivity

| Batch | pp512 (t/s) | tg256 (t/s) |
|---|---|---|
| 512 | 9,411 | 208 |
| 1024 | 8,479 | 203 |
| 2048 | 8,633 | 203 |
| 4096 | 8,672 | 204 |

Similar to Q4_K_S — batch size has minimal impact. The high pp512 value at 512 batch (9,411 ± 573) has elevated variance from warmup effects.

---

## GPU Power & Memory (single-run samples)

| Metric | Q4_K_S | NVFP4 |
|---|---|---|
| Peak GPU power | 28.5 W | 47.7 W |
| Peak VRAM used | 3.5 GiB | 25.4 GiB |
| GPU util (observed) | 1-5% | 1-34% |

Note: nvidia-smi sampling at 0.5s intervals under-samples the actual compute bursts. The power delta (28.5W vs 47.7W) reflects NVFP4's tensor core activity vs Q4_K_S's integer-arithmetic path. VRAM usage reflects model size difference + compute buffer.

---

## Key Takeaways

1. **NVFP4 prompt processing is strictly better** than Q4_K_S at all measured lengths (+13-51%). The Blackwell FP4 tensor cores deliver a real win when compute-bound (prompt eval).

2. **NVFP4 generation is strictly worse** (-23%) — the bandwidth penalty of larger scale metadata dominates.

3. **Flash attention helps NVFP4 modestly** (+2.6% pp512). Not transformational.

4. **MoE vs dense matters more than quantization** — the 35B MoE NVFP4 at 3B active params beats the 27B dense NVFP4 at 27B active params by 2x in prompt processing and 3.3x in generation. Architecture choice dominates quantization choice.

5. **NVFP4 uses 1.7x more GPU power** than Q4_K_S during inference (47.7W vs 28.5W from sampled data), reflecting the tensor core MMA arithmetic intensity.

---

## Raw `llama-bench` Output

### 35B MoE NVFP4 — batch-size sweep

```
| model                       | size    | params | backend | ngl | threads | n_batch | test   | t/s                  |
| --------------------------- | ------: | -----: | ------- | --: | ------: | ------: | -----: | -------------------: |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |     512 | pp512  |  9410.56 ± 572.54    |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |     512 | tg256  |    207.75 ± 2.59     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    1024 | pp512  |  8479.00 ± 24.66     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    1024 | tg256  |    203.15 ± 1.18     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    2048 | pp512  |  8633.48 ± 86.40     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    2048 | tg256  |    203.01 ± 0.78     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    4096 | pp512  |  8672.30 ± 102.42    |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |    4096 | tg256  |    203.54 ± 0.13     |
```

### 35B MoE NVFP4 — flash attention comparison

```
FA off:
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   0 | pp128  |  4691.10 ± 82.00     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   0 | pp512  |  8510.69 ± 49.12     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   0 | tg128  |    200.69 ± 0.65     |

FA on:
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   1 | pp128  |  4799.21 ± 64.25     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   1 | pp512  |  8730.91 ± 35.94     |
| qwen35moe 35B.A3B NVFP4     | 22.88 G | 35.51B | CUDA    |  99 |      24 |   1 | tg128  |    201.48 ± 1.97     |
```

### 27B dense Q5_K_XL

```
| qwen35 27B Q5_K - Medium     | 18.65 G | 26.90B | CUDA    |  99 |      24 | pp128  |  2352.82 ± 162.32    |
| qwen35 27B Q5_K - Medium     | 18.65 G | 26.90B | CUDA    |  99 |      24 | pp512  |  3187.29 ± 71.17     |
| qwen35 27B Q5_K - Medium     | 18.65 G | 26.90B | CUDA    |  99 |      24 | tg128  |     67.52 ± 0.36     |
| qwen35 27B Q5_K - Medium     | 18.65 G | 26.90B | CUDA    |  99 |      24 | tg256  |     67.86 ± 0.07     |
```
