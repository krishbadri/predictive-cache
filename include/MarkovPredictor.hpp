#pragma once
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <optional>
#include <cstdint>

template <typename Key>
class MarkovPredictor {
    public:
        void observe(const Key& prev, const Key& cur) {
            auto & m = trans_[prev];
            auto & cnt = m[cur];
            ++cnt;
            ++totals_[prev];
        }

        std::vector<Key> topk_next(const Key& cur, size_t top_k = 2, uint32_t min_count = 2, double min_prob = 0.05) const {
            std::vector<Key> out;
            auto it = trans_.find(cur);
            if (it == trans_.end()) return out;
            const auto total_it = totals_.find(cur);
            if ((total_it == totals_.end()) || (total_it->second == 0)) return out;
            double total = static_cast<double>(total_it->second);
            std::vector<std::pair<Key, double>> cand;
            cand.reserve(it->second.size());
            for (const auto& [nxt, c] : it->second) {
                if (c < min_count) continue;
                double p = c / total;
                if (p >= min_prob) cand.emplace_back(nxt, p);
            }
            std::sort(cand.begin(), cand.end(), [](auto& a, auto& b){return a.second > b.second;});
            for (size_t i = 0; i < std::min(top_k, cand.size()); i++) {
                out.push_back(cand[i].first);
            }
            return out;
        }


        void decay_half() {
            for (auto& [k,mp] : trans_) {
                for (auto it = mp.begin(); it != mp.end();) {
                    it->second >>= 1;
                    if (it->second == 0) it = mp.erase(it);
                    else ++it;
                } 
            }
            for (auto it = totals_.begin(); it != totals_.end();) {
                it->second >>= 1;
                if (it->second == 0) it = totals_.erase(it);
                else ++it;
            }

        }






    private:
        std::unordered_map<Key, std::unordered_map<Key, uint32_t>> trans_;
        std::unordered_map<Key, uint32_t> totals_;


};