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

#include <cstdio>
#include <cstring>
#include <algorithm>

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

    buffer = ggml_backend_alloc_buffer(backend, pool_bytes);
    if (!buffer) { free_pool(); return false; }

    ggml_backend_tensor_alloc(buffer, pool_tensor, (void *)pool_tensor->data);
    ggml_backend_tensor_set(pool_tensor, nullptr, 0, pool_bytes); // zero-init

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
// Prefetcher
// ===========================================================================

void fate_prefetcher::init(uint32_t nl, uint32_t ne, uint32_t neu, size_t max_expert_bytes) {
    n_layer       = nl;
    n_expert      = ne;
    n_expert_used = neu;

    cur.resize(n_layer);
    prev.resize(n_layer);

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
}

void fate_prefetcher::on_expert(uint32_t layer, int32_t expert_id) {
    if (layer < n_layer && layer < (uint32_t)cur.size()) {
        cur[layer].push_back(expert_id);
    }
}

void fate_prefetcher::on_layer_done(uint32_t layer, fate_gpu_pool & pool) {
    if (layer + 1 >= n_layer) return;
    if (!stream) return;

    std::unordered_set<int32_t> predicted;
    for (int32_t e : cur[layer]) predicted.insert(e);
    if (layer + 1 < (uint32_t)prev.size())
        for (int32_t e : prev[layer + 1]) predicted.insert(e);

    std::vector<job> work;
    for (int32_t eid : predicted) {
        uint32_t next_l = layer + 1;
        for (uint32_t k = 0; k < N_KINDS; k++) {
            uint32_t idx = next_l * N_KINDS + k;
            if (idx >= sources.size() || !sources[idx].base) continue;
            uint64_t key = fate_gpu_pool::make_key(next_l, k, (uint32_t)eid);
            if (pool.key_to_slot.count(key)) continue;
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
    if (strstr(name, "ffn_gate_exps"))  return 0;
    if (strstr(name, "ffn_gate"))       return 0;
    if (strstr(name, "ffn_up_exps"))    return 1;
    if (strstr(name, "ffn_up"))         return 1;
    if (strstr(name, "ffn_down_exps"))  return 2;
    if (strstr(name, "ffn_down"))       return 2;
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

    // Use nb[2] (the actual per-expert stride the scheduler uses) instead of
    // ggml_nbytes/n_expert, because Q4_K_M uses different quant types per layer.
    size_t gate_bytes_max = 0, up_bytes_max = 0, down_bytes_max = 0;
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->nb[2] > 0)
            gate_bytes_max = std::max(gate_bytes_max, (size_t)lay.ffn_gate_exps->nb[2]);
        if (lay.ffn_up_exps && lay.ffn_up_exps->nb[2] > 0)
            up_bytes_max = std::max(up_bytes_max, (size_t)lay.ffn_up_exps->nb[2]);
        if (lay.ffn_down_exps && lay.ffn_down_exps->nb[2] > 0)
            down_bytes_max = std::max(down_bytes_max, (size_t)lay.ffn_down_exps->nb[2]);
    }

    expert_bytes_max = std::max({gate_bytes_max, up_bytes_max, down_bytes_max});
    if (expert_bytes_max == 0) return false;

    fprintf(stderr, "FATE: n_layer=%u n_expert=%u n_expert_used=%u\n", n_layer, n_expert, n_expert_used);
    fprintf(stderr, "FATE: max expert strides: gate=%zuB up=%zuB down=%zuB max=%.1fMB\n",
            gate_bytes_max, up_bytes_max, down_bytes_max, (float)expert_bytes_max / (1024*1024));

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
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_gate_exps->data, ggml_nbytes(lay.ffn_gate_exps));
        if (lay.ffn_up_exps && lay.ffn_up_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_up_exps->data, ggml_nbytes(lay.ffn_up_exps));
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            pinned += fate_prefetch_pin_memory(lay.ffn_down_exps->data, ggml_nbytes(lay.ffn_down_exps));
    }
    fprintf(stderr, "FATE: pinned %u/%u expert tensors for async prefetch\n",
            pinned, n_layer * 3);

    // Init prefetcher with CPU source pointers for every expert tensor.
    prefetch.init(n_layer, n_expert, n_expert_used, expert_bytes_max);
    for (uint32_t il = 0; il < n_layer && il < (uint32_t)model.layers.size(); il++) {
        const auto & lay = model.layers[il];
        if (lay.ffn_gate_exps && lay.ffn_gate_exps->data)
            prefetch.register_src(il, 0, lay.ffn_gate_exps->data, (size_t)lay.ffn_gate_exps->nb[2]);
        if (lay.ffn_up_exps && lay.ffn_up_exps->data)
            prefetch.register_src(il, 1, lay.ffn_up_exps->data, (size_t)lay.ffn_up_exps->nb[2]);
        if (lay.ffn_down_exps && lay.ffn_down_exps->data)
            prefetch.register_src(il, 2, lay.ffn_down_exps->data, (size_t)lay.ffn_down_exps->nb[2]);
    }

    fprintf(stderr, "FATE: system initialized (%u cache slots + prefetch)\n", pool.n_slots);
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
                                  int32_t expert_id, int64_t /*n_expert_total*/,
                                  const char * tensor_name) {
    int layer = parse_layer(tensor_name);
    int kind  = parse_tensor_kind(tensor_name);
    if (layer < 0 || kind < 0 || (uint32_t)layer >= n_layer) return false;
    if (!pool.pool_tensor) return false;

    // --- layer transition: prefetch predicted experts for the next layer ---
    if (layer != prefetch.last_layer) {
        if (prefetch.last_layer >= 0 && prefetch.stream) {
            uint32_t prev_l = (uint32_t)prefetch.last_layer;
            uint32_t next_l = (uint32_t)layer;
            if (next_l < n_layer) {
                std::unordered_set<int32_t> predicted;
                for (int32_t e : prefetch.cur[prev_l]) predicted.insert(e);
                if (next_l < (uint32_t)prefetch.prev.size())
                    for (int32_t e : prefetch.prev[next_l]) predicted.insert(e);

                for (int32_t eid : predicted) {
                    for (uint32_t k = 0; k < fate_prefetcher::N_KINDS; k++) {
                        uint32_t idx = next_l * fate_prefetcher::N_KINDS + k;
                        if (idx >= prefetch.sources.size() || !prefetch.sources[idx].base) continue;
                        uint64_t key = fate_gpu_pool::make_key(next_l, k, (uint32_t)eid);
                        if (pool.key_to_slot.count(key)) continue;
                        int32_t slot = pool.find_or_alloc(key);
                        if (slot < 0) continue;
                        void * dst_ptr = pool.slot_device_ptr((uint32_t)slot);
                        const void * src = (const char *)prefetch.sources[idx].base
                                           + (size_t)eid * prefetch.sources[idx].expert_bytes;
                        size_t copy_n = ((uint32_t)eid < n_expert - 1)
                                      ? prefetch.sources[idx].padded_bytes
                                      : prefetch.sources[idx].expert_bytes;
                        if (prefetch.staging && copy_n <= prefetch.staging_size) {
                            memcpy(prefetch.staging, src, copy_n);
                            fate_prefetch_h2d(prefetch.stream, dst_ptr, prefetch.staging, copy_n);
                        } else {
                            fate_prefetch_h2d(prefetch.stream, dst_ptr, src, copy_n);
                        }
                        prefetch.prefetched++;
                    }
                }
            }
            fate_prefetch_insert_barrier((void *)backend, prefetch.stream);
        }

        if (layer < prefetch.last_layer || prefetch.last_layer < 0) {
            prefetch.on_token_start(pool);
        }
        prefetch.last_layer = layer;
    }

    prefetch.on_expert((uint32_t)layer, expert_id);

    // --- pool lookup ---
    uint64_t key = fate_gpu_pool::make_key((uint32_t)layer, (uint32_t)kind, (uint32_t)expert_id);
    stats.accesses++;

    auto it = pool.key_to_slot.find(key);
    bool is_hit = (it != pool.key_to_slot.end());

    if (is_hit) {
        stats.hits++;
        pool.slots[it->second].last_used = ++pool.tick;
        void * slot_ptr = pool.slot_device_ptr(it->second);
        ggml_backend_tensor_set_async(backend, dst, slot_ptr, offset, size);
    } else {
        stats.misses++;
        int32_t slot = pool.find_or_alloc(key);
        if (slot >= 0) {
            ggml_backend_tensor_set_async(backend, dst, src_data, offset, size);
            ggml_backend_tensor_set_async(backend, pool.pool_tensor,
                                           (const char *)dst->data + offset,
                                           (size_t)slot * pool.slot_bytes, size);
        } else {
            ggml_backend_tensor_set_async(backend, dst, src_data, offset, size);
        }
    }

    return true;
}

// ===========================================================================
// Stats
// ===========================================================================

void fate_system::print_stats() const {
    uint64_t a = stats.accesses.load();
    uint64_t h = stats.hits.load();
    uint64_t m = stats.misses.load();
    double pct = a > 0 ? 100.0 * (double)h / (double)a : 0.0;
    fprintf(stderr, "\nFATE: %lu accesses, %lu hits (%.1f%%), %lu misses, %llu prefetched\n",
            (unsigned long)a, (unsigned long)h, pct, (unsigned long)m,
            (unsigned long long)prefetch.prefetched.load());
}
