// llama-moe-cache — FATE: predictive expert caching for MoE inference
// Copyright (C) 2026 Ongun Manav
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License.
//
// For commercial licensing: ongunmnv@gmail.com

#include "llama-fate.h"
#include "llama-model.h"
#include "llama-context.h"  // for get_sched()

#include <cstdio>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// Runtime dispatch for CUDA prefetch functions.
//
// These functions are defined in ggml-cuda.cu and exported from libggml-cuda.so,
// which is loaded at runtime via dlopen(RTLD_LOCAL).  Because RTLD_LOCAL hides
// the symbols from other libraries, we cannot rely on the linker to resolve them.
// Instead we look them up via dlsym(RTLD_DEFAULT) at each call — after
// ggml_backend_load_all() runs, the CUDA backend's symbols become visible
// through the main binary's symbol table (since libggml-cuda.so links against
// libcudart which is already in the global namespace).
//
// If a function is not yet available (CUDA backend not loaded), the dispatch
// returns a no-op / failure value.
// ---------------------------------------------------------------------------

#include <dlfcn.h>

// Helper: resolve a CUDA function symbol at runtime
template <typename T>
static T resolve_cuda_func(const char * name) {
    static void * handle = nullptr;
    if (!handle) {
        handle = dlopen("libggml-cuda.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) {
            handle = dlopen("libggml-cuda.so", RTLD_LAZY | RTLD_GLOBAL);
        }
    }
    if (handle) {
        void * sym = dlsym(handle, name);
        if (sym) return reinterpret_cast<T>(sym);
    }
    // Fallback: try RTLD_DEFAULT (works if already in global namespace)
    void * sym = dlsym(RTLD_DEFAULT, name);
    if (sym) return reinterpret_cast<T>(sym);
    return nullptr;
}

extern "C" {

void * fate_prefetch_stream_create(void) {
    typedef void * (*fn_t)(void);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_stream_create");
    return fn ? fn() : nullptr;
}

void fate_prefetch_h2d(void * s, void * d, const void * src, size_t n) {
    typedef void (*fn_t)(void *, void *, const void *, size_t);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_h2d");
    if (fn) fn(s, d, src, n);
}

void fate_prefetch_sync(void * s) {
    typedef void (*fn_t)(void *);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_sync");
    if (fn) fn(s);
}

void fate_prefetch_stream_destroy(void * s) {
    typedef void (*fn_t)(void *);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_stream_destroy");
    if (fn) fn(s);
}

void fate_prefetch_insert_barrier(void * bp, void * ps) {
    typedef void (*fn_t)(void *, void *);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_insert_barrier");
    if (fn) fn(bp, ps);
}

bool fate_prefetch_pin_memory(const void * p, size_t sz) {
    typedef bool (*fn_t)(const void *, size_t);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_pin_memory");
    return fn ? fn(p, sz) : false;
}

void * fate_prefetch_alloc_pinned(size_t sz) {
    typedef void * (*fn_t)(size_t);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_alloc_pinned");
    return fn ? fn(sz) : nullptr;
}

void fate_prefetch_free_pinned(void * p) {
    typedef void (*fn_t)(void *);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_free_pinned");
    if (fn) fn(p);
}

void fate_debug_d2h(void * d, const void * s, size_t n) {
    typedef void (*fn_t)(void *, const void *, size_t);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_debug_d2h");
    if (fn) fn(d, s, n);
}

int fate_debug_ptr_type(const void * p) {
    typedef int (*fn_t)(const void *);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_debug_ptr_type");
    return fn ? fn(p) : -1;
}

void fate_prefetch_d2d(void * bp, void * d, const void * s, size_t n) {
    typedef void (*fn_t)(void *, void *, const void *, size_t);
    static fn_t fn = resolve_cuda_func<fn_t>("fate_prefetch_d2d");
    if (fn) fn(bp, d, s, n);
}

} // extern "C"

fate_system * g_fate = nullptr;

// ===========================================================================
// GPU VRAM pool
// ===========================================================================

bool fate_gpu_pool::init(ggml_backend_t backend, size_t slot_sz, size_t target_mb) {
    slot_bytes = slot_sz;
    size_t target_bytes = target_mb * (size_t)(1024*1024);
    n_slots = std::max<uint32_t>(1, (uint32_t)(target_bytes / slot_bytes));
    size_t pool_bytes = (size_t)n_slots * slot_bytes;

    ggml_init_params params = {
        /*.mem_size   =*/ pool_bytes + ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) return false;

    pool_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, pool_bytes);
    if (!pool_tensor) { free_pool(); return false; }

    // Get the backend's default buffer type (CUDA buffer type for CUDA backend)
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!buft) {
        fprintf(stderr, "FATE: failed to get default buffer type\n");
        free_pool();
        return false;
    }

    buffer = ggml_backend_buft_alloc_buffer(buft, pool_bytes);
    if (!buffer) { fprintf(stderr, "FATE: buft_alloc failed\n"); return false; }

    void * base = ggml_backend_buffer_get_base(buffer);
    pool_tensor->data = (char *)base;
    pool_tensor->buffer = buffer;

    slots.resize(n_slots, slot_info{UINT64_MAX, 0});
    fprintf(stderr, "FATE: pool = %u slots x %zu bytes = %.1f MB\n",
            n_slots, slot_bytes, (float)pool_bytes / (1024*1024));
    return true;
}

void fate_gpu_pool::free_pool() {
    if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
    if (ctx)    { ggml_free(ctx);                     ctx    = nullptr; }
    pool_tensor = nullptr;
    slots.clear();
    key_to_slot.clear();
}

int32_t fate_gpu_pool::find_or_alloc(uint64_t key) {
    auto it = key_to_slot.find(key);
    if (it != key_to_slot.end()) {
        return (int32_t)it->second;
    }
    // LRU eviction: find oldest slot
    uint64_t oldest_tick = UINT64_MAX;
    uint32_t oldest_idx = 0;
    for (uint32_t i = 0; i < n_slots; i++) {
        if (slots[i].key == UINT64_MAX) {
            oldest_idx = i;
            break;
        }
        if (slots[i].last_used < oldest_tick) {
            oldest_tick = slots[i].last_used;
            oldest_idx = i;
        }
    }
    // Evict old key
    if (slots[oldest_idx].key != UINT64_MAX) {
        key_to_slot.erase(slots[oldest_idx].key);
    }
    slots[oldest_idx] = slot_info{key, ++tick};
    key_to_slot[key] = oldest_idx;
    return (int32_t)oldest_idx;
}

void * fate_gpu_pool::slot_device_ptr(uint32_t idx) {
    if (!pool_tensor || idx >= n_slots) return nullptr;
    return (char *)pool_tensor->data + (size_t)idx * slot_bytes;
}

// ===========================================================================
// Observer — MTP draft expert selection recording
// ===========================================================================

void fate_observer::init(uint32_t nl) {
    n_layer = nl;
    selections.clear();
    selections.resize(nl);
}

void fate_observer::observe(uint32_t layer, int32_t expert_id) {
    if (layer >= n_layer) return;
    std::lock_guard<std::mutex> lock(mtx);
    selections[layer].push_back(expert_id);
}

void fate_observer::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto &v : selections) v.clear();
}

bool fate_observer::has_data() const {
    // can be called without lock during read — caller must synchronize
    for (const auto &v : selections) {
        if (!v.empty()) return true;
    }
    return false;
}

void fate_observer::extract(std::vector<std::vector<int32_t>> &out) const {
    std::lock_guard<std::mutex> lock(const_cast<fate_observer *>(this)->mtx);
    out.resize(selections.size());
    for (size_t i = 0; i < selections.size(); i++) {
        out[i] = selections[i]; // copy
    }
}

// ===========================================================================
// Prefetcher
// ===========================================================================

void fate_prefetcher::init(uint32_t nl, uint32_t ne, uint32_t neu, size_t max_expert_bytes) {
    n_layer       = nl;
    n_expert      = ne;
    n_expert_used = neu;

    cur.resize(n_layer);
    prev.resize(n_layer);
    mtp_pred.resize(n_layer);

    sources.resize(n_layer * N_KINDS);

    stream = fate_prefetch_stream_create();
    if (!stream) {
        fprintf(stderr, "FATE: prefetch stream creation failed, disabling prefetch\n");
        return;
    }

    staging_size = max_expert_bytes + 512;
    staging = fate_prefetch_alloc_pinned(staging_size);
    if (!staging) {
        fprintf(stderr, "FATE: pinned staging allocation failed (size=%zu), disabling prefetch\n", staging_size);
        fate_prefetch_stream_destroy(stream);
        stream = nullptr;
        return;
    }

    worker = std::thread(&fate_prefetcher::worker_fn, this);

    fprintf(stderr, "FATE: prefetcher initialized (stream=%p staging=%zu)\n", stream, staging_size);
}

void fate_prefetcher::worker_fn() {
    std::unique_lock<std::mutex> lock(mtx);
    while (!quit) {
        cv.wait(lock, [this]{ return has_jobs || quit; });
        if (quit) break;
        processing = true;
        has_jobs = false;
        auto work = std::move(jobs);
        lock.unlock();

        for (auto & j : work) {
            fate_prefetch_h2d(stream, j.dst, j.src, j.n);
        }
        fate_prefetch_sync(stream);

        lock.lock();
        processing = false;
        done_cv.notify_one();
    }
}

void fate_prefetcher::submit(std::vector<job> && work) {
    if (!stream) return;
    std::lock_guard<std::mutex> lock(mtx);
    for (auto & j : work) jobs.push_back(j);
    has_jobs = true;
    cv.notify_one();
}

void fate_prefetcher::register_src(uint32_t layer, uint32_t kind, const void * base, size_t eb) {
    if (layer >= n_layer || kind >= N_KINDS) return;
    uint32_t idx = layer * N_KINDS + kind;
    sources[idx] = {base, eb, eb + 512};
}

void fate_prefetcher::on_token_start(fate_gpu_pool & /*pool*/) {
    std::swap(cur, prev);
    for (auto & v : cur) v.clear();
    last_layer = -1;
    // MTP predictions persist across tokens — they're refreshed by the next
    // MTP draft decode. Clear only if not using MTP guidance.
    if (!use_mtp_pred) {
        clear_mtp_prediction();
    }
}

void fate_prefetcher::on_expert(uint32_t layer, int32_t expert_id) {
    if (layer < n_layer && layer < (uint32_t)cur.size()) {
        cur[layer].push_back(expert_id);
    }
}

// Shared helper: issue prefetch work for a set of experts at a given layer
void fate_prefetcher::prefetch_experts_for_layer(uint32_t layer,
                                                  const std::unordered_set<int32_t> &expert_set,
                                                  fate_gpu_pool & pool) {
    if (!stream) return;
    if (layer >= n_layer) return;
    if (expert_set.empty()) return;

    std::vector<job> work;
    for (int32_t eid : expert_set) {
        for (uint32_t k = 0; k < N_KINDS; k++) {
            uint32_t idx = layer * N_KINDS + k;
            if (idx >= sources.size() || !sources[idx].base) continue;
            uint64_t key = fate_gpu_pool::make_key(layer, k, (uint32_t)eid);
            if (pool.key_to_slot.count(key)) continue; // already cached
            int32_t slot = pool.find_or_alloc(key);
            if (slot < 0) continue;
            void * dst_ptr = pool.slot_device_ptr((uint32_t)slot);
            const void * src = (const char *)sources[idx].base
                               + (size_t)eid * sources[idx].expert_bytes;
            size_t copy_n = ((uint32_t)eid < n_expert - 1)
                          ? sources[idx].padded_bytes
                          : sources[idx].expert_bytes;
            work.push_back({dst_ptr, src, copy_n});
        }
    }
    if (!work.empty()) {
        submit(std::move(work));
        prefetched += work.size();
    }
}

void fate_prefetcher::on_layer_done(uint32_t layer, fate_gpu_pool & pool) {
    if (layer + 1 >= n_layer) return;
    if (!stream) return;

    if (use_mtp_pred && !mtp_pred.empty() && !mtp_pred[layer + 1].empty()) {
        // --- MTP-guided prediction ---
        // Use the MTP draft head's predicted experts for the next layer
        std::unordered_set<int32_t> predicted(mtp_pred[layer + 1].begin(),
                                               mtp_pred[layer + 1].end());
        prefetch_experts_for_layer(layer + 1, predicted, pool);
    } else {
        // --- Temporal + cross-layer heuristic (fallback) ---
        std::unordered_set<int32_t> predicted;
        for (int32_t e : cur[layer]) predicted.insert(e);
        if (layer + 1 < (uint32_t)prev.size())
            for (int32_t e : prev[layer + 1]) predicted.insert(e);
        if (!predicted.empty()) {
            prefetch_experts_for_layer(layer + 1, predicted, pool);
        }
    }
}

void fate_prefetcher::sync() {
    if (!stream) return;
    fate_prefetch_sync(stream);
}

void fate_prefetcher::shutdown() {
    if (worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            quit = true;
            cv.notify_one();
        }
        worker.join();
    }
    if (staging) { fate_prefetch_free_pinned(staging); staging = nullptr; }
    if (stream)  { fate_prefetch_stream_destroy(stream); stream = nullptr; }
}

void fate_prefetcher::set_mtp_prediction(const std::vector<std::vector<int32_t>> &per_layer_pred) {
    if (per_layer_pred.size() != n_layer) return;
    for (uint32_t i = 0; i < n_layer; i++) {
        mtp_pred[i] = per_layer_pred[i];
    }
    use_mtp_pred = true;
}

void fate_prefetcher::clear_mtp_prediction() {
    for (auto &v : mtp_pred) v.clear();
    use_mtp_pred = false;
}

// ===========================================================================
// Helpers
// ===========================================================================

int fate_system::parse_layer(const char * name) {
    if (!name) return -1;
    // format: "blk.<layer>.<tensor_name>.<suffix>"
    if (strncmp(name, "blk.", 4) != 0) return -1;
    name += 4;
    char * end = nullptr;
    long l = strtol(name, &end, 10);
    if (end == name || l < 0 || l > 65535) return -1;
    return (int)l;
}

int fate_system::parse_tensor_kind(const char * name) {
    if (!name) return -1;
    if (strstr(name, "ffn_gate_up_exps")) return 0; // fused gate+up
    if (strstr(name, "ffn_gate_exps"))    return 0;
    if (strstr(name, "ffn_gate"))         return 0;
    if (strstr(name, "ffn_up_exps"))      return 1;
    if (strstr(name, "ffn_up"))           return 1;
    if (strstr(name, "ffn_down_exps"))    return 2;
    if (strstr(name, "ffn_down"))         return 2;
    return -1;
}

// ===========================================================================
// Lifecycle
// ===========================================================================

bool fate_system::init(const llama_model & model, ggml_backend_t backend, int32_t cache_mb) {
    gpu_backend = backend;

    const auto & hp = model.hparams;
    n_layer       = hp.n_layer();
    n_expert      = hp.n_expert;
    n_expert_used = hp.n_expert_used;

    if (n_expert == 0) {
        fprintf(stderr, "FATE: not a MoE model\n");
        return false;
    }

    // Initialize observer for MTP draft context
    observer.init(n_layer + hp.n_layer_nextn);

    // Detect tensor format: fused (ffn_gate_up_exps) or separate (ffn_gate_exps + ffn_up_exps)
    bool has_fused_gate_up = false;
    if (!model.layers.empty()) {
        const auto & lay0 = model.layers[0];
        has_fused_gate_up = (lay0.ffn_gate_up_exps != nullptr) &&
                            (lay0.ffn_gate_exps == nullptr);
    }

    // Use nb[2] (the actual per-expert stride the scheduler uses) instead of
    // ggml_nbytes/n_expert, because Q4_K_M uses different quant types per layer.
    size_t gate_up_bytes_max = 0, down_bytes_max = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (has_fused_gate_up) {
            if (lay.ffn_gate_up_exps && lay.ffn_gate_up_exps->nb[2] > 0)
                gate_up_bytes_max = std::max(gate_up_bytes_max, (size_t)lay.ffn_gate_up_exps->nb[2]);
        } else {
            if (lay.ffn_gate_exps && lay.ffn_gate_exps->nb[2] > 0)
                gate_up_bytes_max = std::max(gate_up_bytes_max, (size_t)lay.ffn_gate_exps->nb[2]);
            if (lay.ffn_up_exps && lay.ffn_up_exps->nb[2] > 0)
                gate_up_bytes_max = std::max(gate_up_bytes_max, (size_t)lay.ffn_up_exps->nb[2]);
        }
        if (lay.ffn_down_exps && lay.ffn_down_exps->nb[2] > 0)
            down_bytes_max = std::max(down_bytes_max, (size_t)lay.ffn_down_exps->nb[2]);
    }

    expert_bytes_max = std::max(gate_up_bytes_max, down_bytes_max);
    if (expert_bytes_max == 0) return false;

    fprintf(stderr, "FATE: n_layer=%u n_expert=%u n_expert_used=%u%s\n",
            n_layer, n_expert, n_expert_used,
            has_fused_gate_up ? " (fused gate+up)" : "");
    fprintf(stderr, "FATE: max expert strides: gate_up=%zuB down=%zuB max=%.1fMB\n",
            gate_up_bytes_max, down_bytes_max, (float)expert_bytes_max / (1024*1024));

    size_t padded_slot = expert_bytes_max + 512;
    uint32_t min_slots = n_layer * n_expert_used * 3;
    size_t min_mb = (size_t)min_slots * padded_slot / (1024*1024) + 64;
    size_t target_mb = (cache_mb > 0) ? (size_t)cache_mb : std::max(min_mb, (size_t)4096);

    if (!pool.init(backend, padded_slot, target_mb)) {
        fprintf(stderr, "FATE: pool allocation failed\n");
        return false;
    }

    // Pin expert weight memory for truly async H2D prefetch
    uint32_t pinned = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (has_fused_gate_up) {
            if (lay.ffn_gate_up_exps && lay.ffn_gate_up_exps->data)
                pinned += fate_prefetch_pin_memory(lay.ffn_gate_up_exps->data, ggml_nbytes(lay.ffn_gate_up_exps));
        } else {
            if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
                pinned += fate_prefetch_pin_memory(lay.ffn_gate_exps->data, ggml_nbytes(lay.ffn_gate_exps));
            if (lay.ffn_up_exps && lay.ffn_up_exps->data)
                pinned += fate_prefetch_pin_memory(lay.ffn_up_exps->data, ggml_nbytes(lay.ffn_up_exps));
        }
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_down_exps->data, ggml_nbytes(lay.ffn_down_exps));
    }
    fprintf(stderr, "FATE: pinned %u/%u expert tensors for async prefetch\n",
            pinned, n_layer * (has_fused_gate_up ? 2u : 3u));

    // Init prefetcher with CPU source pointers for every expert tensor
    prefetch.init(n_layer, n_expert, n_expert_used, expert_bytes_max);
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (has_fused_gate_up) {
            if (lay.ffn_gate_up_exps && lay.ffn_gate_up_exps->data)
                prefetch.register_src(il, 0, lay.ffn_gate_up_exps->data, (size_t)lay.ffn_gate_up_exps->nb[2]);
        } else {
            if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
                prefetch.register_src(il, 0, lay.ffn_gate_exps->data, (size_t)lay.ffn_gate_exps->nb[2]);
            if (lay.ffn_up_exps && lay.ffn_up_exps->data)
                prefetch.register_src(il, 1, lay.ffn_up_exps->data, (size_t)lay.ffn_up_exps->nb[2]);
        }
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            prefetch.register_src(il, 2, lay.ffn_down_exps->data, (size_t)lay.ffn_down_exps->nb[2]);
    }

    fprintf(stderr, "FATE: system initialized (%u cache slots + prefetch + MTP observer)\n", pool.n_slots);
    return true;
}

void fate_system::shutdown() {
    prefetch.shutdown();
    pool.free_pool();
}

// ===========================================================================
// Expert copy hook — the hot path
//
// Called per-expert during MUL_MAT_ID weight copies.
// Detects layer transitions to drive prefetch.
// ===========================================================================

bool fate_system::on_expert_copy(ggml_backend_t backend,
                                  struct ggml_tensor * dst,
                                  const void * src_data, size_t offset, size_t size,
                                  int32_t expert_id, int64_t n_expert_total,
                                  const char * tensor_name) {
    int layer = parse_layer(tensor_name);
    int kind  = parse_tensor_kind(tensor_name);
    if (layer < 0 || kind < 0 || (uint32_t)layer >= n_layer) return false;
    if (!pool.pool_tensor) return false;

    // Track the expert access for temporal prediction fallback
    prefetch.on_expert((uint32_t)layer, expert_id);

    // --- layer transition: prefetch predicted experts for the next layer ---
    if (layer != prefetch.last_layer) {
        if (prefetch.last_layer >= 0 && prefetch.stream) {
            // Prefetch predicted experts for the next layer.
            // Uses either MTP-guided predictions or temporal/cross-layer fallback.
            prefetch.on_layer_done((uint32_t)prefetch.last_layer, pool);
            // Use CUDA event barrier instead of full sync when MTP guidance is active
            // This lets the main stream proceed without waiting for prefetch to complete,
            // as MTP predictions are more reliable and the prefetch should finish
            // before the main stream reaches the predicted layer.
            if (prefetch.use_mtp_pred) {
                fate_prefetch_insert_barrier((void *)backend, prefetch.stream);
            } else {
                // Fallback: sync for temporal prediction (less reliable, higher risk of
                // prefetch consuming bandwidth needed by main stream)
                fate_prefetch_sync(prefetch.stream);
            }
        }
        if (layer < prefetch.last_layer || prefetch.last_layer < 0) {
            prefetch.on_token_start(pool);
        }
        prefetch.last_layer = layer;
    }

    // --- pool lookup ---
    uint64_t key = fate_gpu_pool::make_key((uint32_t)layer, (uint32_t)kind, (uint32_t)expert_id);
    stats.accesses++;

    auto it = pool.key_to_slot.find(key);

    if (it != pool.key_to_slot.end()) {
        // Cache hit — D2D copy from pool slot to dst (~500 GB/s)
        stats.hits++;
        pool.slots[it->second].last_used = ++pool.tick;
        void * slot_ptr = pool.slot_device_ptr(it->second);
        fate_prefetch_d2d((void *)backend, (char *)dst->data + offset, slot_ptr, size);
        return true;
    }

    // Cache miss — copy from CPU source (PCIe ~25 GB/s) and write to pool
    stats.misses++;
    int32_t slot = pool.find_or_alloc(key);
    if (slot >= 0) {
        ggml_backend_tensor_set_async(backend, dst, src_data, offset, size);
        ggml_backend_tensor_set_async(backend, pool.pool_tensor,
                                       src_data,
                                       (size_t)slot * pool.slot_bytes, size);
    } else {
        ggml_backend_tensor_set_async(backend, dst, src_data, offset, size);
    }

    return true;
}

// ===========================================================================
// Observer-only hook — records expert selections without caching
//
// Installed on the MTP draft context's scheduler.
// ===========================================================================

bool fate_system::on_expert_observe(void * user_data,
                                     ggml_backend_t backend,
                                     struct ggml_tensor * dst,
                                     const void * src_data, size_t offset, size_t size,
                                     int32_t expert_id, int64_t n_expert_total,
                                     const char * tensor_name) {
    (void)backend;
    (void)dst;
    (void)src_data;
    (void)offset;
    (void)size;
    (void)n_expert_total;

    auto * sys = (fate_system *)user_data;
    if (!sys) return false;

    int layer = parse_layer(tensor_name);
    if (layer < 0) return false;

    sys->observer.observe((uint32_t)layer, expert_id);
    return false; // don't handle — let normal copy proceed
}

// ===========================================================================
// Transfer observer data → MTP prediction buffer
// ===========================================================================

void fate_system::transfer_observer_to_prediction(uint32_t mtp_layer_offset) {
    if (!observer.has_data()) return;

    // Extract observed expert IDs from the MTP draft decode
    std::vector<std::vector<int32_t>> observed;
    observer.extract(observed);

    // Build per-trunk-layer predictions from MTP head observations.
    // MTP head layer at index `mtp_layer_offset` in the model has its own MoE FFN.
    // We use its expert selections as predictions for ALL trunk layers.
    // This is the paper's key finding: MTP head routing decisions strongly
    // correlate with trunk routing decisions for the next token.
    std::vector<std::vector<int32_t>> per_layer_pred(n_layer);

    if (mtp_layer_offset < observed.size() && !observed[mtp_layer_offset].empty()) {
        // Build a deduplicated prediction set for the observed MTP layer
        std::unordered_set<int32_t> pred_set(observed[mtp_layer_offset].begin(),
                                              observed[mtp_layer_offset].end());
        std::vector<int32_t> pred(pred_set.begin(), pred_set.end());
        // Use the same prediction for all trunk layers
        for (uint32_t il = 0; il < n_layer; il++) {
            per_layer_pred[il] = pred;
        }
    } else if (observed.size() > mtp_layer_offset) {
        // Fallback: search all observed layers for data
        for (uint32_t i = 0; i < observed.size(); i++) {
            if (!observed[i].empty()) {
                std::unordered_set<int32_t> pred_set(observed[i].begin(), observed[i].end());
                std::vector<int32_t> pred(pred_set.begin(), pred_set.end());
                for (uint32_t il = 0; il < n_layer; il++) {
                    per_layer_pred[il] = pred;
                }
                break;
            }
        }
    }

    bool any_nonempty = false;
    for (const auto &v : per_layer_pred) {
        if (!v.empty()) { any_nonempty = true; break; }
    }

    if (any_nonempty) {
        prefetch.set_mtp_prediction(per_layer_pred);
        fprintf(stderr, "FATE: MTP prediction active (%zu experts/layer from %u MTP heads)\n",
                per_layer_pred[0].size(), (unsigned)observed.size());
    }

    observer.reset();
}

// ===========================================================================
// Stats
// ===========================================================================

void fate_system::print_stats() const {
    uint64_t a = stats.accesses.load();
    uint64_t h = stats.hits.load();
    uint64_t m = stats.misses.load();
    double pct = a > 0 ? 100.0 * (double)h / (double)a : 0.0;
    fprintf(stderr, "\nFATE: %lu accesses, %lu hits (%.1f%%), %lu misses, %llu prefetched%s\n",
            (unsigned long)a, (unsigned long)h, pct, (unsigned long)m,
            (unsigned long long)prefetch.prefetched.load(),
            prefetch.use_mtp_pred ? " [MTP-guided]" : " [temporal]");
}

// ===========================================================================
// Public API
// ===========================================================================

void llama_fate_install_observer(llama_context * ctx) {
    if (!g_fate || !ctx) return;

    // Get the context's scheduler and install observer-only hook
    auto * sched = ctx->get_sched();
    if (!sched) return;

    ggml_backend_sched_set_expert_hook(sched, fate_system::on_expert_observe, (void *)g_fate);
    fprintf(stderr, "FATE: observer installed on context\n");
}

void llama_fate_transfer_prediction() {
    if (!g_fate) return;
    // MTB head layer offset: for models with n_layer trunk + n_layer_nextn MTP blocks,
    // the first MTP head's MoE layer is at index n_layer.
    g_fate->transfer_observer_to_prediction(g_fate->n_layer);
}
