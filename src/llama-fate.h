// llama-moe-cache — FATE: predictive expert caching for MoE inference
// Copyright (C) 2026 Ongun Manav
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License.
//
// For commercial licensing: ongunmnv@gmail.com

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;

// CUDA prefetch functions (implemented in ggml-cuda.cu)
extern "C" {
    void * fate_prefetch_stream_create(void);
    void   fate_prefetch_h2d(void * stream, void * dst, const void * src, size_t n);
    void   fate_prefetch_sync(void * stream);
    void   fate_prefetch_stream_destroy(void * stream);
    void   fate_prefetch_insert_barrier(void * backend_ptr, void * prefetch_stream);
    bool   fate_prefetch_pin_memory(const void * ptr, size_t size);
    void * fate_prefetch_alloc_pinned(size_t size);
    void   fate_prefetch_free_pinned(void * p);
    void   fate_debug_d2h(void * dst, const void * src, size_t n);
    int    fate_debug_ptr_type(const void * ptr);
}

// ---------------------------------------------------------------------------
// GPU VRAM pool — dedicated persistent buffer for caching expert weights
// ---------------------------------------------------------------------------
struct fate_gpu_pool {
    ggml_backend_buffer_t buffer      = nullptr;
    ggml_context *        ctx         = nullptr;
    ggml_tensor *         pool_tensor = nullptr;
    size_t                slot_bytes  = 0;
    uint32_t              n_slots     = 0;

    struct slot_info {
        uint64_t key       = UINT64_MAX;
        uint64_t last_used = 0;
    };
    std::vector<slot_info> slots;
    std::unordered_map<uint64_t, uint32_t> key_to_slot;
    uint64_t tick = 0;

    bool     init(ggml_backend_t backend, size_t slot_bytes, size_t target_mb);
    void     free_pool();
    int32_t  find_or_alloc(uint64_t key);
    void *   slot_device_ptr(uint32_t idx);

    static uint64_t make_key(uint32_t layer, uint32_t kind, uint32_t expert) {
        return ((uint64_t)layer << 16) | ((uint64_t)kind << 8) | expert;
    }
};

// ---------------------------------------------------------------------------
// Background prefetch engine
//
// Uses a separate CUDA stream + worker thread to overlap H2D expert copies
// with GPU compute.  Two prediction strategies:
//   1. Temporal:     previous token's experts for each layer
//   2. Cross-layer:  current layer's experts predict next layer's experts
// ---------------------------------------------------------------------------
struct fate_prefetcher {
    static const uint32_t N_KINDS = 4;

    void * stream = nullptr;
    void * staging = nullptr;     // pinned staging buffer for async H2D
    size_t staging_size = 0;

    std::thread           worker;
    std::mutex            mtx;
    std::condition_variable cv;
    std::condition_variable done_cv;
    bool                  quit       = false;
    bool                  has_jobs   = false;
    bool                  processing = false;

    struct job { void * dst; const void * src; size_t n; };
    std::vector<job> jobs;

    int32_t  last_layer = -1;
    uint32_t n_layer = 0, n_expert = 0, n_expert_used = 0;

    // expert selections: [layer] → set of expert ids
    std::vector<std::vector<int32_t>> cur;
    std::vector<std::vector<int32_t>> prev;

    // CPU base pointers for merged expert tensors: [layer * N_KINDS + kind]
    struct tensor_src {
        const void * base = nullptr;
        size_t expert_bytes = 0;
        size_t padded_bytes = 0; // expert_bytes + 512 padding for non-last experts
    };
    std::vector<tensor_src> sources;

    std::atomic<uint64_t> prefetched{0};

    void init(uint32_t nl, uint32_t ne, uint32_t neu, size_t max_expert_bytes);
    void register_src(uint32_t layer, uint32_t kind, const void * base, size_t eb);
    void on_token_start(fate_gpu_pool & pool);
    void on_expert(uint32_t layer, int32_t expert_id);
    void on_layer_done(uint32_t layer, fate_gpu_pool & pool);
    void sync();
    void shutdown();

private:
    void submit(std::vector<job> && work);
    void worker_fn();
};

// ---------------------------------------------------------------------------
// Main FATE system
// ---------------------------------------------------------------------------
struct fate_system {
    uint32_t n_layer       = 0;
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    size_t   expert_bytes_max = 0;

    fate_gpu_pool    pool;
    fate_prefetcher  prefetch;
    ggml_backend_t   gpu_backend = nullptr;

    struct {
        std::atomic<uint64_t> accesses{0};
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> misses{0};
    } stats;

    bool init(const llama_model & model, ggml_backend_t backend, int32_t cache_mb = 0);
    void shutdown();

    bool on_expert_copy(ggml_backend_t backend,
                        struct ggml_tensor * dst,
                        const void * src_data, size_t offset, size_t size,
                        int32_t expert_id, int64_t n_expert_total,
                        const char * tensor_name);

    void print_stats() const;

    static int parse_layer(const char * name);
    static int parse_tensor_kind(const char * name);
};

extern fate_system * g_fate;
