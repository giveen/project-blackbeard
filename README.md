# Project Blackbeard

<picture>
  <img alt="Project Blackbeard" src="media/chat-splash.png" width="700">
</picture>



> A Blackwell-focused performance fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), targeting NVIDIA RTX 50-series / SM100 hardware.

---

## Performance vs Upstream llama.cpp

Benchmarked 2026-07-30 on identical hardware, identical model, identical `llama-bench` parameters.

**Hardware:** RTX 5090 (SM120, 32 GiB VRAM), CUDA 13.x  
**Model:** Qwen3-Coder 30B-A3B MoE, Q4_K_XL (16.45 GiB), `-ngl 99 -t 24`

| Test | llama.cpp `e9fa078` | Blackbeard `8ba2152` | Speedup |
|---|---:|---:|---|
| **pp128** (prompt processing, 128 tokens) | 3,955 t/s | **10,887 t/s** | **+175%** |
| **pp512** (prompt processing, 512 tokens) | 9,312 t/s | **21,879 t/s** | **+135%** |
| **pp1024** (prompt processing, 1024 tokens) | 9,360 t/s | **21,084 t/s** | **+125%** |
| **tg128** (text generation, 128 tokens) | 352 t/s | **362 t/s** | **+3%** |
| **tg256** (text generation, 256 tokens) | 350 t/s | **360 t/s** | **+3%** |

**Key wins:**
- **Prefill is 2.25-2.75x faster** via `iter_k` scaling (256 -> 4096) across 19 quant types, exploiting Blackwell's larger shared memory (128 KB) for wider VRAM tile sizes.
- **Decode +3%** via `__ldg` cache-bypass on single-use q8_1 reads, `__launch_bounds__(32,4)` for improved SM occupancy, and nwarps=1 on Q4_K/NVFP4 to eliminate shared-memory reduction overhead.
- 22 commits across decode/prefill tuning, all backed by nsys/ncu profiling on real hardware.

This project is built on the excellent foundation of [llama.cpp](https://github.com/ggml-org/llama.cpp), which was created and is maintained by [Georgi Gerganov](https://github.com/ggerganov) and the [ggml-org](https://github.com/ggml-org) community. We respectfully acknowledge that work; Project Blackbeard is a narrow downstream optimization effort focused on one hardware family, not an upstream replacement.

Project Blackbeard focuses on:
- NVIDIA Blackwell GPU kernels and memory-path optimizations.
- SM100-specific layouts and scheduling improvements.
- Reproducible Blackwell benchmarking and verification procedures.
- Keeping changes scoped so they remain reviewable and maintainable against future upstream GGML changes.

Non-Blackwell architectures are out of scope for this fork.

---

## What this repo does

At its core, this remains an LLM inference library using GGUF models. The practical surface area is still the same primary tools:

- `llama-cli` - local chat / completion
- `llama-server` - lightweight OpenAI-compatible HTTP server
- `llama-bench` - benchmark token generation and prompt processing
- `llama-perplexity` - evaluate model quality metrics

It still supports Hugging Face GGUF models via `-hf`, GGUF quantization workflows, and CPU fallback when needed.

---

## Current backend scope

This fork currently targets:

- CUDA only, with Blackwell/SM100-specific kernel paths
- CPU fallback for debugging and correctness checks

We do not maintain the broad cross-platform backend matrix that upstream `llama.cpp` supports.

Upstream `llama.cpp` also supports Metal, BLAS/BLIS, Vulkan, SYCL, HIP, MUSA, OpenCL, CANN, WebGPU, IBM zDNN, VirtGPU, and other backends. Those paths are not the focus here.

---

## Quick start

```sh
# Run local completion
llama-cli -m model.gguf

# Run local chat
llama-cli -m model.gguf -cnv

# Start OpenAI-compatible server
llama-server -m model.gguf --port 8080

# Run a benchmark
llama-bench -m model.gguf
```

To use models directly from Hugging Face:

```sh
llama-cli -hf org/model-id-GGUF
```

---

## Requirements

- NVIDIA Blackwell GPU
- CUDA toolkit compatible with the tested release
- cmake + ninja or your preferred build workflow

---

## Contribution expectations

Submissions for this fork must include:
- a clear Blackwell hardware target
- a reproducible benchmark delta
- an ability to explain the scheduling or memory-latency impact of the change

AI-assisted implementations are allowed, but must be accompanied by real profiling on Blackwell hardware and disclosed in the submission.

Please do not submit changes targeting non-Blackwell architectures.

---

## License

MIT, same as upstream llama.cpp.
