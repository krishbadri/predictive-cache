## Predictive Cache (C++17) — High‑Throughput, Predictive, Sharded Caching Library

High-performance, header-only caching primitives with lock‑sharding, TinyLFU admission control, and an optional per‑shard first‑order Markov predictor for prefetch/protect. Includes runnable demos and a Google Benchmark suite for reproducible evaluation.

### Executive Summary
- Engineered for high write/read concurrency via lock‑striped sharding.
- Constant‑time critical operations (get/put/erase) in the steady state.
- TinyLFU admission using a Count‑Min Sketch to resist cache pollution under skew.
- Predictive prefetch/protect using per‑shard Markov chains to exploit sequential/Markovian locality.
- Minimal dependencies, simple integration (headers only), portable (GCC/Clang/MSVC).

### System Design Overview

```
Client threads
    │
    ▼
Key hashing → shard index (std::hash % N)
    │
    ├─► Shard 0: { mutex, TinyLFU‑LRU core, MarkovPredictor }
    ├─► Shard 1: { mutex, TinyLFU‑LRU core, MarkovPredictor }
    ├─► ...
    └─► Shard N‑1: { mutex, TinyLFU‑LRU core, MarkovPredictor }

Core (per shard):
  - LRU data structure (DLL + hash table) for O(1) recency updates and eviction
  - Count‑Min Sketch for TinyLFU admission (new ≥ victim ⇒ admit)
  - Optional Markov predictor: learn prev→cur transitions; prefetch top‑K next keys
```

### Algorithms, Complexity, and Guarantees
- LRU get/put/erase: amortized O(1). Eviction touches only tail (LRU key).
- LFU variant (standalone): O(1) average via frequency buckets and min‑freq tracking.
- TinyLFU admission: O(d) updates/queries in a d‑row Count‑Min Sketch (default d=4; w=4096).
  - Standard CMS guarantees: with width `w` and depth `d`, over `N` operations the estimate overcounts by ≤ εN with probability ≥ 1−δ, where ε≈e/w and δ≈exp(−d). This implementation uses fast, fixed seeds and power‑of‑two width for masking.
- Sharding: O(1) shard index; operations are serialized per shard only.
- Memory: O(capacity) across shards with light constant factors; CMS adds `w*d*sizeof(uint32_t)` per shard.

### Concurrency and Consistency Model
- Per‑shard coarse‑grained `std::mutex`. No cross‑shard coordination on single key ops.
- Single‑key operations are linearizable within their shard; keys hashed to different shards have no transactional semantics across shards.
- Memory visibility adheres to C++11+ rules; locks provide happens‑before relations around reads/writes.
- Practical effect: strong ordering for a given key; high throughput under mixed workloads with low contention when keyspace is well‑distributed.

### Predictive Prefetching (Markov)
- Each shard maintains a first‑order Markov model over keys it serves: counts of `prev → curr` transitions.
- On `get(k)`: learn the transition, then rank candidates via `P(next | k)`; prefetch top‑K keys meeting configurable count/probability thresholds.
- Prefetch is realized as inserting default‑constructed placeholder values if absent; this “protects” likely next keys via admission and recency even before the actual request.
- Aging: `decay_models()` halves counts to forget stale patterns and cap state.

### TinyLFU Admission — Details
- Count‑Min Sketch tracks approximate frequency per key.
- On admission, the new key is admitted if `estimate(new) ≥ estimate(victim_LRU)`; otherwise the new key is dropped, improving hit ratio under scanning/low‑reuse workloads.
- Aging: `decay()` (on the TinyLFU‑LRU wrapper) halves CMS counters to adapt to drift.

### Highlights
- **LRUCache**: O(1) get/put via linked-list + hash map.
- **LFUCache**: O(1) average get/put with frequency lists and min-frequency tracking.
- **TinyLFUAdmittingLRU**: LRU with Count-Min Sketch–based admission to raise hit rate under skew.
- **ShardedLRU / ShardedWTinyLFU**: Per-shard locks for concurrency and scale-out.
- **PredictiveShardedCache**: Adds a lightweight Markov predictor to prefetch/protect likely next keys.
- **Benchmarks**: Zipf, uniform, and sequential-burst workloads via Google Benchmark.

---

## API Surface (selected)

- `LRUCache<Key,Value>`
  - `std::optional<Value> get(const Key&)`
  - `void put(const Key&, const Value&)`
  - `bool erase(const Key&)`
  - `bool contains(const Key&) const`
  - `size_t size() const`, `size_t capacity() const`

- `TinyLFUAdmittingLRU<Key,Value>`
  - Same API as `LRUCache` plus `void decay()` for periodic aging.

- `ShardedWTinyLFU<Key,Value>` and `ShardedLRU<Key,Value>`
  - `get/put/erase` as above; internally route to a shard by key hash.
  - `size_t num_shards() const`

- `PredictiveShardedCache<Key,Value>`
  - `get/put/erase` as above
  - `size_t num_shards() const`
  - `void decay_models()` for predictor aging
  - `struct Options { size_t shards; size_t prefetch_topk; uint32_t min_trans_count; double min_trans_prob; bool enable_prefetch; }`

---

## Getting Started

### Requirements
- C++17-compatible compiler (GCC/Clang/MSVC)
- CMake 3.10+

The caches live in `include/` and are header-only. The `src/` and `benchmarks/` targets are for examples and performance testing.

### Build (Release)

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

This produces (paths may vary by platform):
- `build/Release/main` – simple sanity demo
- `build/Release/bench` – ad‑hoc benchmark runner
- `build/Release/gbench` – Google Benchmark suite

On Windows with MSVC, executables are under `build/Release/` and use the generated `.sln` as usual.

---

## Quick Usage

Add the headers to your CMake target:

```cmake
target_include_directories(your_target PUBLIC ${CMAKE_SOURCE_DIR}/include)
```

### LRU
```cpp
#include "LRUCache.hpp"

LRUCache<int, std::string> cache(3);
cache.put(1, "A");
cache.put(2, "B");
cache.get(1);                // makes 1 most-recent
cache.put(3, "C");
cache.put(4, "D");           // evicts least-recent (key 2)
```

### Sharded W‑TinyLFU (LRU + TinyLFU admission)
```cpp
#include "ShardedWTinyLFU.hpp"

ShardedWTinyLFU<int, std::string> cache(/*capacity=*/1000, /*shards=*/8);
if (!cache.get(42)) cache.put(42, "value");
```

### Predictive (Markov) Sharded Cache
```cpp
#include "PredictiveShardedCache.hpp"

using Key = int;
PredictiveShardedCache<Key, std::string>::Options opts;
opts.shards = 8;
opts.prefetch_topk = 2;       // predict & prefetch top-2 likely next keys
opts.min_trans_count = 2;
opts.min_trans_prob  = 0.10;
opts.enable_prefetch = true;

PredictiveShardedCache<Key, std::string> pcache(/*capacity=*/1000, opts);
if (!pcache.get(7)) pcache.put(7, "x");
```

Notes:
- Prefetching works by inserting default-constructed placeholder values for predicted keys when absent.
- You can periodically call `decay_models()` to make the predictor forget stale patterns.

---

## Tuning & Sizing Guide
- **Shards**: start with number of physical cores for mixed read/write workloads; increase if hotspots persist.
- **Capacity split**: evenly divided across shards; choose a global capacity first, then shard count.
- **TinyLFU (CMS) width/depth**: defaults (`w=4096, d=4`) are a good balance for most; increase `w` to reduce overestimation under very large keyspaces.
- **Predictive thresholds**:
  - `prefetch_topk`: 1–3 for most; higher increases memory pressure with diminishing returns.
  - `min_trans_count` / `min_trans_prob`: raise to suppress noise; lower to react faster to new patterns.
- **Aging cadence**: call `decay()` / `decay_models()` periodically (e.g., timer/ops based) to adapt to drift.

---

## Architecture

### Core Data Structures
- `LRUCache<Key,Value>`
  - Doubly-linked list of `(key,value)` for recency, plus `unordered_map` for O(1) lookup.
  - `get()` moves the key to MRU; `put()` updates or inserts and evicts LRU when full.

- `LFUCache<Key,Value>`
  - Tracks frequencies per key and maintains per-frequency key lists.
  - Evicts from the current `min_freq_` list on capacity pressure.

- `CountMinSketch`
  - Fixed-width, fixed-depth sketch with saturating counters and optional `decay_half()`.
  - Used by TinyLFU to estimate popularity. For best performance, use a power-of-two width (the implementation masks with `width_-1`).

- `TinyLFUAdmittingLRU<Key,Value>`
  - Wraps `LRUCache` and consults the sketch on admission: a new item is admitted if its estimated frequency ≥ that of the LRU victim.

### Concurrency via Sharding
- `ShardedLRU` and `ShardedWTinyLFU`
  - Split total capacity across N shards.
  - Each shard is protected by its own mutex; index is `hash(key) % num_shards`.
  - `size()` aggregates per-shard sizes while briefly locking each shard.

### Predictive Layer
- `PredictiveShardedCache<Key,Value>`
  - Base cache: `ShardedWTinyLFU`.
  - Per-shard `MarkovPredictor<Key>` learns transitions `prev → current` on every `get()` and `put()`.
  - On `get(k)`, it predicts top‑K next keys (with configurable min count/probability) and prefetches by inserting placeholders if missing.
  - Optional model aging via `decay_models()`.

---

## Benchmarks

Two styles are provided:

- Ad‑hoc runner: `src/bench.cpp` – prints hit rate and throughput for a few workloads (uniform, Zipf, sequential burst).
- Google Benchmark suite: `benchmarks/bm_cache.cpp`
  - Measures operations and reports `hit_rate` in counters:
  - Zipf workloads for `ShardedLRU` and `ShardedWTinyLFU`.
  - Predictive vs non‑predictive on sequential bursts.

Run Google Benchmarks (recommended):
```bash
cd build
./Release/gbench --benchmark_counters_tabular=true
# or on Unix-like:
# ./gbench --benchmark_counters_tabular=true
```

Benchmarking methodology:
- Warmups ensure predictors and admission structures stabilize before timing.
- Reported `hit_rate` is computed inside the benchmarks; throughput derives from total ops / wall time.
- For fair comparisons, capacity, shard count, and keyspace are held constant across policies.

Representative expectations (will vary by machine/workload):
- TinyLFU generally improves hit rate over plain LRU under Zipf/heavy skew.
- Predictive mode improves sequential‑pattern hit rates after brief warmup; gains depend on transition strength.

You can adjust capacity, shards, and key-space in the benchmark arguments or source.

---

## Productionization Notes
- Determinism: benchmarks use fixed RNG seeds; library behavior is deterministic given key sequences.
- Observability: hook your metrics around call sites; counters such as hits/misses, evictions, admissions, and prefetches are straightforward to expose.
- Build/tooling: works with MSVC v143 and CMake FetchContent for Google Benchmark; the library itself has no runtime deps.
- Safety: no exceptions thrown on hot paths except invalid constructor args (e.g., zero shards/capacity); prefer guarding at integration points.
- Portability: standard C++17 only; no platform intrinsics.

---

## Project Structure
- `include/`
  - `LRUCache.hpp`, `LFUCache.hpp`, `CountMinSketch.hpp`
  - `TinyLFUAdmittingLRU.hpp` – LRU with TinyLFU admission
  - `ShardedLRU.hpp`, `ShardedWTinyLFU.hpp` – concurrent sharded caches
  - `MarkovPredictor.hpp`, `PredictiveShardedCache.hpp` – predictive layer
- `src/`
  - `main.cpp` – minimal sanity demo
  - `bench.cpp` – simple benchmark runner
- `benchmarks/`
  - `bm_cache.cpp` – Google Benchmark suite
- `CMakeLists.txt` – builds examples and integrates Google Benchmark via FetchContent
- `tests/` – placeholder for future tests

---

## Tips & Caveats
- Choose a `cms_width` that’s a power of two for the Count‑Min Sketch.
- TinyLFU’s `decay()` and predictor’s `decay_models()` help adapt to drift.
- Sharding improves concurrency; tune `shards` for your workload and core count.
- For read‑mostly workloads with sequential correlations, enable predictive prefetching.
- For memory-sensitive deployments, consider disabling prefetch and rely on admission/protection only.

---

## Contributing
Issues and pull requests are welcome—especially around additional eviction policies, workload generators, and correctness/perf tests.

---

## License
No license has been specified in this repository. If you plan to use this in a project, please contact the repository owner to clarify licensing.


