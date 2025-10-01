#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <functional>
#include "TinyLFUAdmittingLRU.hpp"

template <typename Key, typename Value>
class ShardedWTinyLFU {
public:
    ShardedWTinyLFU(size_t capacity, size_t shards,
                    size_t cms_width = 4096, size_t cms_depth = 4)
        : locks_(shards), shards_(shards)
    {
        if (shards == 0) throw std::invalid_argument("shards must be > 0");
        const size_t base  = capacity / shards;
        const size_t extra = capacity % shards;

        for (size_t i = 0; i < shards; ++i) {
            const size_t cap = base + (i == shards - 1 ? extra : 0);
            shards_[i] = std::make_unique<TinyLFUAdmittingLRU<Key, Value>>(cap, cms_width, cms_depth);
        }
    }

    std::optional<Value> get(const Key& key) {
        const size_t i = idx(key);
        std::scoped_lock l(locks_[i]);
        return shards_[i]->get(key);
    }

    void put(const Key& key, const Value& value) {
        const size_t i = idx(key);
        std::scoped_lock l(locks_[i]);
        shards_[i]->put(key, value);
    }

    bool erase(const Key& key) {
        const size_t i = idx(key);
        std::scoped_lock l(locks_[i]);
        return shards_[i]->erase(key);
    }

    size_t num_shards() const { return shards_.size(); }

private:
    size_t idx(const Key& key) const {
        return hasher_(key) % shards_.size();
    }

    std::vector<std::mutex> locks_;
    std::vector<std::unique_ptr<TinyLFUAdmittingLRU<Key, Value>>> shards_;
    std::hash<Key> hasher_;
};
