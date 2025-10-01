#include <benchmark/benchmark.h>
#include <random>
#include <vector>
#include <cmath>
#include "ShardedLRU.hpp"
#include "ShardedWTinyLFU.hpp"
#include "PredictiveShardedCache.hpp"

using Key = int;

static std::discrete_distribution<Key> make_zipf(size_t key_space, double s) {
    std::vector<double> w(key_space);
    for (size_t i=0;i<key_space;++i) w[i] = 1.0/std::pow(double(i+1), s);
    return std::discrete_distribution<Key>(w.begin(), w.end());
}

static void BM_LRU_Zipf(benchmark::State& st) {
    size_t capacity = st.range(0), key_space = st.range(1), shards = 8;
    ShardedLRU<Key, std::string> cache(capacity, shards);
    std::mt19937 rng(123);
    auto zipf = make_zipf(key_space, 1.2);

    size_t hits=0, misses=0;
    for (auto _ : st) {
        Key k = zipf(rng);
        if (cache.get(k)) ++hits;
        else { ++misses; cache.put(k, "x"); }
    }
    st.counters["hit_rate"] = double(hits)/(hits+misses);
    st.counters["ops"] = hits+misses;
}
BENCHMARK(BM_LRU_Zipf)->Args({1000, 10000})->Unit(benchmark::kNanosecond);

static void BM_TinyLFU_Zipf(benchmark::State& st) {
    size_t capacity = st.range(0), key_space = st.range(1), shards = 8;
    ShardedWTinyLFU<Key, std::string> cache(capacity, shards);
    std::mt19937 rng(123);
    auto zipf = make_zipf(key_space, 1.2);

    size_t hits=0, misses=0;
    for (auto _ : st) {
        Key k = zipf(rng);
        if (cache.get(k)) ++hits;
        else { ++misses; cache.put(k, "x"); }
    }
    st.counters["hit_rate"] = double(hits)/(hits+misses);
    st.counters["ops"] = hits+misses;
}
BENCHMARK(BM_TinyLFU_Zipf)->Args({1000, 10000})->Unit(benchmark::kNanosecond);

// Predictive on sequential burst
static void BM_Predictive_Seq(benchmark::State& st) {
    size_t capacity = st.range(0), key_space = st.range(1), shards = 8;

    PredictiveShardedCache<Key, std::string>::Options opt;
    opt.shards = shards; opt.prefetch_topk = 1; opt.min_trans_count = 4; opt.min_trans_prob = 0.2;
    PredictiveShardedCache<Key, std::string> cache(capacity, opt);

    std::vector<Key> seq; seq.reserve(key_space);
    for (Key i=0;i<(Key)key_space;i+=3){ seq.push_back(i); if(i+1<key_space)seq.push_back(i+1); if(i+2<key_space)seq.push_back(i+2); }
    size_t idx=0; auto next=[&]{ Key k=seq[idx]; idx=(idx+1)%seq.size(); return k; };

    // short warmup to train the Markov model
    for (int i=0;i<10000;++i) { Key k=next(); if(!cache.get(k)) cache.put(k,"x"); }

    size_t hits=0, misses=0;
    for (auto _ : st) {
        Key k = next();
        if (cache.get(k)) ++hits; else { ++misses; cache.put(k, "x"); }
    }
    st.counters["hit_rate"] = double(hits)/(hits+misses);
    st.counters["ops"] = hits+misses;
}
BENCHMARK(BM_Predictive_Seq)->Args({1000, 10000})->Unit(benchmark::kNanosecond);

BENCHMARK_MAIN();
