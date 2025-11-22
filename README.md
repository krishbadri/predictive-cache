## Predictive Cache (C++17)

High-performance, header-only caching primitives with sharding, TinyLFU admission, and an optional Markov-based predictive layer for prefetch/protect. Includes runnable demos and Google Benchmark microbenchmarks.

### Highlights
- **LRUCache**: O(1) get/put via linked-list + hash map.
- **LFUCache**: O(1) average get/put with frequency lists and min-frequency tracking.
- **TinyLFUAdmittingLRU**: LRU with Count-Min Sketch–based admission to raise hit rate under skew.
- **ShardedLRU / ShardedWTinyLFU**: Per-shard locks for concurrency and scale-out.
- **PredictiveShardedCache**: Adds a lightweight Markov predictor to prefetch/protect likely next keys.
- **Benchmarks**: Zipf, uniform, and sequential-burst workloads via Google Benchmark.

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

Example snippet (Zipf, higher is better):
- `hit_rate` increases with TinyLFU admission under heavy skew.
- Predictive mode improves sequential‑pattern hits after a short warmup.

You can adjust capacity, shards, and key-space in the benchmark arguments or source.

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


