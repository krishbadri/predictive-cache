#pragma once
#include <unordered_map>
#include <list>
#include <optional>
#include <stdexcept>

template <typename Key, typename Value>
class LFUCache {
    public:
        explicit LFUCache(size_t capacity) : capacity_(capacity) {
            if (capacity_ == 0) throw std::invalid_argument("capacity must be > 0");
        }
        std::optional<Value> get(const Key& key) {
            auto it = kv_.find(key);
            if (it == kv_.end()){
                return std::nullopt;
            }
            touch(key);
            return it->second.value;
        }

        void put(const Key& key, const Value& value) {
            if (capacity_ == 0) {
                return;
            }
            auto it = kv_.find(key);
            if (it != kv_.end()) {
                it->second.value = value;
                touch(key);
                return;
            }
            
            if (kv_.size() >= capacity_) {
                evict();
            }

            freq_list_[1].push_front(key);
            kv_[key] = {value, 1, freq_list_[1].begin()};
            min_freq_ = 1;
        }

        bool erase(const Key& key) {
            auto it = kv_.find(key);
            if (it == kv_.end()){
                return false;
            }
            size_t f = it->second.freq;
            freq_list_[f].erase(it->second.it);
            if (freq_list_[f].empty() && min_freq_ == f){
                min_freq_ += 1;
            }
            kv_.erase(it);
            return true;
        }

        bool contains(const Key& key) const {
            return kv_.count(key) != 0;
        }

        size_t size() const {
            return kv_.size();
        }

        size_t capacity() const {
            return capacity_;
        }


    private:
        struct Node {
            Value value;
            size_t freq;
            typename std::list<Key>::iterator it;
        };

        size_t capacity_;
        size_t min_freq_ = 0;
        std::unordered_map<Key, Node> kv_;
        std::unordered_map<size_t, std::list<Key>> freq_list_;

        void touch(const Key& key) {
            auto& node = kv_[key];
            size_t f = node.freq;
            // remove from current freq list
            freq_list_[f].erase(node.it);
            if (freq_list_[f].empty() && f == min_freq_) {
                ++min_freq_;
            }
            // add to higher freq list
            freq_list_[f + 1].push_front(key);
            node.freq++;
            node.it = freq_list_[f + 1].begin();
        }

        void evict() {
            auto & minFreqList = freq_list_[min_freq_];
            Key evict_key = minFreqList.back();
            minFreqList.pop_back();
            kv_.erase(evict_key);
            if (minFreqList.empty()) {
                freq_list_.erase(min_freq_);
            }
        }
        
};


