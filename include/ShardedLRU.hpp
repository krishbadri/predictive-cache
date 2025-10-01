#pragma once
#include <vector>
#include <mutex>
#include <memory>
#include <optional>
#include <functional>
#include <stdexcept>
#include "LRUCache.hpp"

template <typename Key, typename Value>
class ShardedLRU {
    public:
        ShardedLRU(size_t capacity, size_t num_shards) : locks_(num_shards), shards_(num_shards) {
            if (num_shards == 0) {
                throw std::invalid_argument("num_shards must be > 0");
            }
            const size_t base = capacity / num_shards;
            const size_t extra = capacity % num_shards;
            for (size_t i = 0; i < num_shards; i++){
                const size_t cap = base + (i == num_shards - 1 ? extra : 0);
                shards_[i] = std::make_unique<LRUCache<Key, Value>>(cap);
            }
        }

        std::optional<Value> get(const Key& key) {
            const size_t i = shard_idx(key); // hash function
            std::scoped_lock lock(locks_[i]);
            return shards_[i]->get(key);
        }

        void put(const Key& key, const Value& value){
            const size_t i = shard_idx(key);
            std::scoped_lock lock(locks_[i]);
            shards_[i]->put(key, value);
        }

        bool erase(const Key& key) {
            const size_t i = shard_idx(key);
            std::scoped_lock lock(locks_[i]);
            return shards_[i]->erase(key);
        }

        bool contains(const Key& key){
            const size_t i = shard_idx(key);
            std::scoped_lock lock(locks_[i]);
            return shards_[i]->contains(key);
        }

        size_t size() {
            size_t s = 0;
            for (size_t i = 0; i < shards_.size(); i++){
                std::scoped_lock lock(locks_[i]);
                s += shards_[i]->size();
            }
            return s;
        }

        size_t num_shards() const {
            return shards_.size();
        }

    
    private:
        size_t shard_idx(const Key& key) const {
            return hasher_(key) % shards_.size();
        }

        std::vector<std::mutex> locks_;
        std::vector<std::unique_ptr<LRUCache<Key, Value>>> shards_;
        std::hash<Key> hasher_;

};