cmake_minimum_required(VERSION 3.10)
project(PredictiveCache)
set(CMAKE_CXX_STANDARD 17)

add_executable(main src/main.cpp)
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(bench src/bench.cpp)
target_include_directories(bench PUBLIC ${CMAKE_SOURCE_DIR}/include)

set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)

include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3)
FetchContent_MakeAvailable(benchmark)

add_executable(gbench benchmarks/bm_cache.cpp)
target_link_libraries(gbench PRIVATE benchmark::benchmark)
target_include_directories(gbench PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_target_properties(gbench PROPERTIES CXX_STANDARD 17)
