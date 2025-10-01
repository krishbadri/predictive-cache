#pragma once
#include <optional>
#include "LRUCache.hpp"
#include "CountMinSketch.hpp"

template <typename Key, typename Value>
class TinyLFUAdmittingLRU {
    public:
        TinyLFUAdmittingLRU(size_t capacity, size_t cms_width = 4096, size_t cms_depth = 4) : lru_(capacity), cms_(cms_width, cms_depth) {}

        std::optional<Value> get(const Key& key) {
            // lru cache has a map of key to it
            // it points to DLL node, MRU ordered
            cms_.add(key);
            return lru_.get(key);
        }

        void put(const Key& key, const Value& value) {
            cms_.add(key);
            
            if (lru_.contains(key)){
                lru_.put(key, value);
                return;
            }

            if (lru_.size() < lru_.capacity()) {
                lru_.put(key, value);
                return;
            }

            auto victim = lru_.peek_lru_key();
            if (!victim) {
                lru_.put(key, value);
                return;
            }
            uint32_t newEst = cms_.estimate(key);
            uint32_t victimEst = cms_.estimate(*victim);
            if (newEst >= victimEst) {
                lru_.erase(*victim);
                lru_.put(key, value);
            }

        }

        bool erase(const Key& key) {
            return lru_.erase(key);
        }

        bool contains(const Key& key) const {
            return lru_.contains(key);
        }

        size_t size() const {
            return lru_.size();
        }

        size_t capacity() const {
            return lru_.capacity();
        }

        void decay() {
            cms_.decay_half();
        }

        
    private:
        LRUCache<Key, Value> lru_;
        CountMinSketch cms_;

};