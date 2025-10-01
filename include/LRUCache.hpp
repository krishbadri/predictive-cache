#pragma once
#include <list>
#include <unordered_map>
#include <optional>
#include <utility>

template <typename Key, typename Value>
class LRUCache {
    public:
        explicit LRUCache(size_t capacity) : capacity_(capacity) {}
        // Returns value if present; moves key to MRU position
        std::optional<Value> get(const Key& key){
            auto it = map_.find(key);
            if (it == map_.end()){
                return std::nullopt;
            }
            items_.splice(items_.begin(), items_, it->second);
            return it->second->second;
        }
    
        void put(const Key& key, const Value& value){
            auto it = map_.find(key);
            if (it != map_.end()){
                it->second->second = value;
                items_.splice(items_.begin(), items_, it->second);
                return;
            }
            items_.emplace_front(key, value);
            map_[key] = items_.begin();
            if (map_.size() > capacity_){
                auto& [oldKey, _] = items_.back();
                map_.erase(oldKey);
                items_.pop_back();
            }
        }

        bool erase(const Key& key){
            auto it = map_.find(key);
            if (it == map_.end()){
                return false;
            }
            items_.erase(it->second);
            map_.erase(it);
            return true;
        }

        bool contains(const Key& key) const {
            return map_.count(key) != 0;
        }

        size_t size() const {
            return map_.size();
        }

        size_t capacity() const {
            return capacity_;
        }

        std::optional<Key> peek_lru_key() const {
            if (items_.empty()) {
                return std::nullopt;
            }
            return items_.back().first;
        }

    private:
        std::list<std::pair<Key, Value>> items_;
        std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> map_;
        size_t capacity_;
    
};
