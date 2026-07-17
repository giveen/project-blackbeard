# Blackwell Optimization Ideas

Curated proposals for Blackwell-specific work, ranked by estimated impact-to-effort ratio. Every entry must clearly state what it targets, why it fits SM100, and what benchmark evidence would prove it.

---

## 1. Hardware-Aware Dynamic Speculative Decoding (DSD)

**Status**: Proposed / research complete

**What**: Make the number of draft tokens (K) in speculative decoding adaptive to batch size, using an offline-profiled lookup table mapping `(batch_size) -> optimal K`. The optimal K maximizes `goodput = AL / ITL` (Acceptance Length / Inter-Token Latency).

**Why Blackwell**: RTX 5090's ~2.5× compute vs ~1.5× memory bandwidth shift over prior GPUs moves the compute-bound crossover to lower batch sizes (BS 4-8 vs BS 8-16). DSD's offline profiling automatically captures this hardware-specific curve. Fixed-K SD regresses at high BS; DSD never regresses.

**Expected speedup (RTX 5090, `llama-server`)**:

| Regime | DSD vs vanilla | DSD vs fixed-K SD (K=3) |
|---|---|---|
| BS 1-2 (memory-bound) | 2-3× | ~0% (matches K=3) |
| BS 4-8 (transition) | 1.3-1.8× | ~5-10% |
| BS 16-32 (compute-bound) | 1.1-1.3× | ~15-20% |
| BS 64+ (saturated) | 0-5% | ~20-25% |

**Effort**: ~2-3 days, ~320 lines. No structural refactoring needed — the speculative decoding architecture already supports per-step K control (`dp.n_max`), and the stats counters already track acceptance length and timing.

**Key components**:
1. `--spec-dsd-profile` flag: offline profiling harness that sweeps BS × K and measures AL/ITL
2. Lookup table (flat JSON): `{"bs_1": K_opt, "bs_2": K_opt, ...}`
3. `--spec-dsd-table` flag: runtime loads table, sets `dp.n_max` per draft step
4. Optional live adaptation: update table from runtime statistics

**Risks**: Model-specific and hardware-specific tables (each (target,draft,GPU) triple needs its own profile). No benefit at BS=1 (single-user).

**Verification**: `llama-bench` on `llama-server` at BS ∈ {1,2,4,8,16,32,64,128,256} comparing vanilla vs fixed-K vs DSD, with before/after tokens/sec. The profiling mode itself generates the evidence.

**References**: [Cohere DSD blog](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding), [vLLM PR #32374](https://github.com/vllm-project/vllm/pull/32374)

---

## 2. (Placeholder)

*Space for next proposal.*
