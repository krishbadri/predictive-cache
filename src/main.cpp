#include <iostream>
#include <thread>
#include <vector>
#include "ShardedLRU.hpp"

int main() {
    ShardedLRU<int, std::string> cache(8 /*total capacity*/, 4 /*shards*/);

    // Single-thread sanity
    cache.put(1, "A");
    cache.put(2, "B");
    cache.put(3, "C");
    std::cout << "get(2): " << cache.get(2).value_or("MISS") << "\n";
    std::cout << "contains(1): " << cache.contains(1) << "\n";

    // Multithreaded writes/reads
    const int N = 10000;
    auto writer = [&] (int base) {
        for (int i = 0; i < N; ++i) cache.put(base + i, "X");
    };
    auto reader = [&] (int base) {
        size_t hits = 0;
        for (int i = 0; i < N; ++i) if (cache.get(base + i)) ++hits;
        std::cout << "Reader hits from base " << base << ": " << hits << "\n";
    };

    std::thread t1(writer, 0);
    std::thread t2(writer, 5000);
    std::thread t3(reader, 0);
    std::thread t4(reader, 5000);
    t1.join(); t2.join(); t3.join(); t4.join();

    std::cout << "Total shards: " << cache.num_shards() << "\n";
    std::cout << "Approx size now: " << cache.size() << "\n"; // <= total capacity
    return 0;
}
