#include <catch2/catch_all.hpp>

#include "hdc.hpp"

TEST_CASE("Vector binary bundle") {
    const int dim = 10000;
    const int no_vectors = 1000;
    auto im = hdc::ItemMemory<hdc::bin_t>(dim, no_vectors);
    std::vector<hdc::bin_t> test_data;
    for (int i = 0; i < no_vectors; i++) { test_data.emplace_back(im.at(i)); }

    BENCHMARK("Binary bundle D=10000") {
        return hdc::add(test_data);
    };
}
