#pragma once
#include <vector>
#include <mutex>
#include <optional>
#include <memory>
#include "ShardedWTinyLFU.hpp"
#include "MarkovPredictor.hpp"

// Adds Markov prefetch/protect to ShardedWTinyLFU.
// Prefetch policy: on get(k), prefetch top-P predicted next keys for the same shard.
template <typename Key, typename Value>
class PredictiveShardedCache {
public:
    struct Options {
        size_t shards = 8;
        size_t prefetch_topk = 1;
        uint32_t min_trans_count = 4;
        double min_trans_prob = 0.2;
        bool enable_prefetch = true;      // if false, only "protects" via admission/recency
    };

    PredictiveShardedCache(size_t capacity, const Options& opt = Options{})
        : base_(capacity, opt.shards), opts_(opt),
          preds_(opt.shards), locks_(opt.shards), prev_(opt.shards) {}

    std::optional<Value> get(const Key& key) {
        const size_t i = shidx(key);
        std::scoped_lock lk(locks_[i]);

        // learn transition: prev_i -> key
        if (prev_[i].has_value()) {
            preds_[i].observe(*prev_[i], key);
        }
        prev_[i] = key;

        auto result = base_.get(key);

        if (opts_.enable_prefetch) {
            auto cand = preds_[i].topk_next(key, opts_.prefetch_topk,
                                            opts_.min_trans_count, opts_.min_trans_prob);
            for (const auto& nxt : cand) {
                // simple prefetch: insert placeholder if not present
                if (!base_.get(nxt)) {
                    base_.put(nxt, Value{}); // default-constructed value as a stand-in
                }
            }
        }
        return result;
    }

    void put(const Key& key, const Value& value) {
        const size_t i = shidx(key);
        std::scoped_lock lk(locks_[i]);
        base_.put(key, value);
        prev_[i] = key; // treat put as an access for sequence learning
    }

    bool erase(const Key& key) {
        const size_t i = shidx(key);
        std::scoped_lock lk(locks_[i]);
        return base_.erase(key);
    }

    size_t num_shards() const { return opts_.shards; }

    // Optional: call occasionally
    void decay_models() {
        for (size_t i = 0; i < preds_.size(); ++i) {
            std::scoped_lock lk(locks_[i]);
            preds_[i].decay_half();
        }
    }

private:
    size_t shidx(const Key& k) const { return hasher_(k) % opts_.shards; }

    ShardedWTinyLFU<Key, Value> base_;
    Options opts_;

    // per-shard predictor + last key seen
    std::vector<MarkovPredictor<Key>> preds_;
    std::vector<std::optional<Key>> prev_;
    std::vector<std::mutex> locks_;
    std::hash<Key> hasher_;
};
