# Instructions for Project Blackbeard (llama.cpp Blackwell Fork)

> [!IMPORTANT]
> Project Blackbeard is a specialized fork of `llama.cpp` dedicated **exclusively** to performance optimizations, architectural enhancements, and kernel tuning for **NVIDIA Blackwell** GPUs. Contributions targeting other architectures are out of scope.
> AI-assisted and fully autonomous AI agents are welcome to contribute, provided all submissions meet our empirical performance standards and hardware verification requirements.

---

## Guidelines for Contributors & AI Agents

We view AI agents as force multipliers, not hazards. To maintain a high-velocity, high-performance repository, we grant AI agents the freedom to draft code, write documentation, and manage Git workflows, provided they adhere to our validation standards.

### The Blackwell Golden Rule

**No profile, no merge.**
Every performance-impacting modification--whether authored by a human, an AI assistant, or an autonomous agent--must be accompanied by concrete, reproducible benchmark data (e.g., `llama-bench` runs, custom profiling outputs) demonstrating a verifiable improvement on actual Blackwell hardware (Compute Capability 10.0+ / SM100).

---

## Permitted AI Agent Actions

To streamline development, AI agents are permitted to perform the following actions:

* **Draft PR Descriptions & Comments:** Agents may draft PR descriptions, commit messages, and replies to reviewer comments. However, any drafted text must be factual, concise, and free of generic "AI fluff" or empty technical jargon.
* **Generate CUDA & C++ Optimizations:** Agents are encouraged to write Blackwell-specific CUDA kernels, leverage SM100-specific instructions, and optimize FP4/FP6/FP8 tensor core paths.
* **Manage Git Workflows:** Agents may stage, commit, and push changes, as well as draft Pull Requests using tools like the GitHub CLI (`gh`), *if* authorized by their local workspace configuration.

---

## Agent Verification & Execution Protocol

When executing tasks within this repository, AI agents must follow this operational sequence:

1. **Verify Hardware Target:** Target: SM100.
Confirm that the proposed optimization specifically targets Blackwell architecture features (e.g., larger L1 shared memory capacity, asynchronous copy behaviors, or tensor core structures).


2. **Implement and Profile:** Run llama-bench.
Implement the changes and execute local benchmarking. The agent or the hosting user must run profiling tools on actual Blackwell hardware to capture the performance delta.


3. **Generate Clean Documentation:** Drafting the PR.
Draft the PR description or commit message. It must include:

* The exact Blackwell hardware used (e.g., RTX 5090).
* Before/After benchmark metrics (tokens/sec, latency, or bandwidth).
* A concise explanation of the architectural change.


4. **Commit and Push:** Git Integration.
Commit the changes. Agents should append `Assisted-by: <agent-name>` or `Co-authored-by: <agent-name>` to the commit message metadata to maintain an audit trail.


---

## Code and Formatting Standards

To keep the codebase clean and maintainable, agents must adhere to the following formatting rules:

* **No Unicode in Code/Commits:** Avoid emdashes (`--`), unicode arrows (`->`), or characters like `x` and `...`. Use ASCII equivalents instead: `-`, `->`, `x`, `...`.
* **Concise Code Comments:** Do not let the AI generate paragraphs of textbook definitions inside CUDA kernels. Keep comments focused strictly on the hardware-level reasoning.
* **Architectural Guards:** Ensure all Blackwell-specific CUDA intrinsics are properly guarded behind preprocessor directives (e.g., checking for CUDA compilation flags targeting compute capability 100).

---

## Examples

### Good Submission (Empirical & Autonomous)

An example of an agent-generated PR description that is concise, factual, and includes the required performance profile:

```markdown
Subject: cuda : optimize flash-attention kernel for Blackwell SM100 layout

This PR optimizes shared memory tiling for the attention kernels specifically
targeting Blackwell's larger L1 shared memory capacity.

Co-authored-by: Claude-3.5-Sonnet

Benchmarks (RTX 5090, CUDA 13.2, batch=512):
- Before: 142.4 t/s
- After:  158.1 t/s (+11% speedup)

```

### Bad Submission (Theoretical & Verbose)

Avoid PRs that rely on theoretical gains without empirical proof:

```markdown
Subject: cuda : refactor loops for theoretical speedup

I have refactored the loops in the CUDA kernel to utilize parallel processing paradigms
which theoretically increases instruction-level parallelism and optimizes warp scheduling.

(Error: No benchmarks provided, no hardware tested)

```

### Code Comments comparison

```cpp
// GOOD (Concise, hardware-specific explanation)
// SM100 allows larger shared memory allocation per block; increase tile size
#define BB_ATTN_TILE_SIZE 256

// BAD (Verbose, generic explanation of CUDA concepts)
// We increase the tile size to 256 because Blackwell architectures have expanded 
// shared memory capacities per Streaming Multiprocessor (SM), allowing us to fit 
// more elements in flight simultaneously and reduce global memory roundtrips.
#define BB_ATTN_TILE_SIZE 256

```
