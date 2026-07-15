# Project Blackbeard

<picture>
  <img alt="Project Blackbeard" src="media/chat-splash.png" width="700">
</picture>



> A Blackwell-focused performance fork of `llama.cpp`, targeting NVIDIA RTX 50-series / SM100 hardware.

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
