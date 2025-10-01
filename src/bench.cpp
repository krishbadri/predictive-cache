#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <string>
#include <cmath> // std::pow

#include "ShardedLRU.hpp"
#include "LFUCache.hpp"
#include "TinyLFUAdmittingLRU.hpp"
#include "ShardedWTinyLFU.hpp"
#include "PredictiveShardedCache.hpp"

// Generic benchmark for any cache with: get(key)->optional<V>, put(key,val), num_shards()
template <typename Cache, typename NextKeyFn>
double run_benchmark(Cache& cache, size_t ops, NextKeyFn&& next_key, bool warmup = true) {
    using Clock = std::chrono::high_resolution_clock;

    size_t hits = 0, misses = 0;

    if (warmup) {
        // quick warmup so caches/predictors learn a bit
        const size_t warm = cache.num_shards() * 10 + 100;
        for (size_t i = 0; i < warm; ++i) {
            auto k = next_key();
            if (cache.get(k)) ++hits; else { ++misses; cache.put(k, "x"); }
        }
        hits = misses = 0;
    }

    auto t0 = Clock::now();
    for (size_t i = 0; i < ops; ++i) {
        auto k = next_key();
        if (cache.get(k)) ++hits; else { ++misses; cache.put(k, "x"); }
    }
    auto t1 = Clock::now();
    std::chrono::duration<double> dt = t1 - t0;

    double hit_rate = (hits + misses) ? double(hits) / double(hits + misses) : 0.0;
    std::cout << "ops=" << ops
              << " hits=" << hits
              << " misses=" << misses
              << " hit_rate=" << hit_rate
              << " time=" << dt.count() << "s"
              << " throughput=" << (ops / std::max(1e-9, dt.count())) << " ops/s\n";
    return hit_rate;
}

int main() {
    using Key = int;

    // --- knobs ---
    const size_t key_space = 10'000;
    const size_t capacity  = 1'000;   // 10% of key_space
    const size_t shards    = 8;
    const size_t ops       = 1'000'000;

    std::mt19937 rng(123);

    // ===== Uniform workload =====
    {
        ShardedLRU<Key, std::string> cache(capacity, shards);

        std::uniform_int_distribution<Key> uni(0, (Key)key_space - 1);
        auto uniform = [&]() { return uni(rng); };

        std::cout << "=== Uniform workload ===\n";
        run_benchmark(cache, ops, uniform);
    }

    // ===== Zipf workload =====
    std::vector<double> weights(key_space);
    const double s = 1.2;
    for (size_t i = 0; i < key_space; ++i) weights[i] = 1.0 / std::pow(double(i + 1), s);
    std::discrete_distribution<Key> zipf(weights.begin(), weights.end());
    auto zipf_gen = [&]() { return zipf(rng); };

    {
        ShardedLRU<Key, std::string> cache(capacity, shards);
        std::cout << "=== Zipf(s=1.2) workload ===\n";
        run_benchmark(cache, ops, zipf_gen);
    }

    // ===== Sequential burst workload (A->B->C repeating blocks) =====
    std::vector<Key> seq;
    seq.reserve(key_space);
    for (Key i = 0; i < (Key)key_space; i += 3) {
        seq.push_back(i);
        if (i + 1 < (Key)key_space) seq.push_back(i + 1);
        if (i + 2 < (Key)key_space) seq.push_back(i + 2);
    }
    size_t idx = 0;
    auto seq_gen = [&]() {
        Key k = seq[idx];
        idx = (idx + 1) % seq.size();
        return k;
    };

    {
        ShardedLRU<Key, std::string> cache(capacity, shards);
        std::cout << "=== Sequential burst workload ===\n";
        run_benchmark(cache, ops, seq_gen);
    }

    // ===== LFU baseline on Zipf =====
    std::cout << "\n=== LFU vs Zipf workload ===\n";
    {
        LFUCache<Key, std::string> lfu(capacity);
        size_t hits = 0, misses = 0;
        for (size_t i = 0; i < ops; ++i) {
            Key k = zipf_gen();
            if (lfu.get(k)) ++hits;
            else { ++misses; lfu.put(k, "x"); }
        }
        double hit_rate = double(hits) / (hits + misses);
        std::cout << "LFU hit_rate=" << hit_rate << "\n";
    }

    // ===== W-TinyLFU on Zipf =====
    std::cout << "\n=== W-TinyLFU (LRU+admission) on Zipf ===\n";
    {
        ShardedWTinyLFU<Key, std::string> cache(capacity, shards);
        size_t hits = 0, misses = 0;
        for (size_t i = 0; i < ops; ++i) {
            Key k = zipf_gen();
            if (cache.get(k)) ++hits;
            else { ++misses; cache.put(k, "x"); }
        }
        double hit_rate = double(hits) / (hits + misses);
        std::cout << "W-TinyLFU hit_rate=" << hit_rate << "\n";
    }

    // ===== Predictive (Markov) on Sequential =====
    std::cout << "\n=== Predictive (Markov) on Sequential workload ===\n";
    {
        typename PredictiveShardedCache<Key, std::string>::Options opts;
        opts.shards = shards;
        opts.prefetch_topk = 2;      // try 1–3
        opts.min_trans_count = 2;
        opts.min_trans_prob  = 0.10;
        opts.enable_prefetch = true;

        PredictiveShardedCache<Key, std::string> pcache(capacity, opts);

        // small warmup to let model learn transitions
        run_benchmark(pcache, 10'000, seq_gen, /*warmup=*/true);

        // now measure (no extra warmup)
        run_benchmark(pcache, ops, seq_gen, /*warmup=*/false);
    }

    // ===== Predictive (Markov) on Zipf =====
    std::cout << "\n=== Predictive (Markov) on Zipf ===\n";
    {
        typename PredictiveShardedCache<Key, std::string>::Options opts;
        opts.shards = shards;
        opts.prefetch_topk = 2;
        opts.min_trans_count = 2;
        opts.min_trans_prob  = 0.10;
        opts.enable_prefetch = true;

        PredictiveShardedCache<Key, std::string> pcache(capacity, opts);

        // small warmup so the predictor sees some transitions
        run_benchmark(pcache, 10'000, zipf_gen, /*warmup=*/true);

        // measure
        run_benchmark(pcache, ops, zipf_gen, /*warmup=*/false);
    }


    return 0;
}
