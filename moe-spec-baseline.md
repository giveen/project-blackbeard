# moe-spec Baseline Benchmarks
Date: 2026-07-16
Branch: moe-spec
GPU: RTX 5090 (32 GB VRAM)

## Qwen3.6-35B NVFP4 (22.88 GB, 35.51B params, A3B MoE)

| Config | pp128 | tg256 |
|---|---|---|
| All GPU (-ngl 99) | 1,774 t/s | 206 t/s |
| Partial offload (-ngl 20) | 236 t/s | — |
| Minimal GPU (-ngl 10) | 180 t/s | — |

## Gemma4 26B Q5_K_S (17.54 GB, 25.23B params, A4B MoE)

| Config | pp128 | tg256 |
|---|---|---|
| All GPU (-ngl 99) | 3,573 t/s | 229 t/s |
