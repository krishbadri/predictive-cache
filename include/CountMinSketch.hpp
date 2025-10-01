#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>

class CountMinSketch {
public:
    CountMinSketch(size_t width = 4096, size_t depth = 4)
        : width_(width), depth_(depth), rows_(depth, std::vector<uint32_t>(width, 0)) {}

    // MUTATING: must NOT be const
    template <typename Key>
    void add(const Key& k) {
        for (size_t i = 0; i < depth_; ++i) {
            size_t idx = index(k, i);              // ok to call from non-const
            uint32_t& c = rows_[i][idx];           // mutate counter
            if (c != UINT32_MAX) ++c;              // saturating
        }
    }

    // READ-ONLY: keep const
    template <typename Key>
    uint32_t estimate(const Key& k) const {
        uint32_t m = UINT32_MAX;
        for (size_t i = 0; i < depth_; ++i) {
            size_t idx = index(k, i);              // needs index() to be const
            m = std::min(m, rows_[i][idx]);
        }
        return m == UINT32_MAX ? 0u : m;
    }

    void decay_half() {
        for (auto& row : rows_)
            for (auto& c : row) c >>= 1;
    }

private:
    // PURE: must be const so it can be called from estimate()
    template <typename Key>
    size_t index(const Key& k, size_t i) const {
        uint64_t h = std::hash<Key>{}(k);
        static constexpr uint64_t seeds[8] = {
            0x9e3779b185ebca87ULL, 0xc2b2ae3d27d4eb4fULL,
            0x165667b19e3779f9ULL, 0xd6e8feb86659fd93ULL,
            0x94d049bb133111ebULL, 0x2545f4914f6cdd1dULL,
            0x60642e2a34326f15ULL, 0x9e3779b97f4a7c15ULL
        };
        h ^= seeds[i & 7] + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        return static_cast<size_t>(h & (width_ - 1)); // width should be power-of-two
    }

    size_t width_, depth_;
    std::vector<std::vector<uint32_t>> rows_;
};
