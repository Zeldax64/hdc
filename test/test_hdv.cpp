#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

#include "HDV.hpp"

static const hdv::dim_t _DIM = 1000;

static bool _is_orthogonal(const hdv::HDV &a, const hdv::HDV &b) {
    hdv::dim_t dim = a.dim;
    std::size_t dist = hdv::dist(a, b);
    float difference = (float)dist/(float)dim;
    // We consider that two vectors are orthogonal if at least 40% of its
    // entries are different.
    return difference > 0.40;
}

//TODO: Remove later since it is useless
//static bool _is_similar(const hdv::HDV &a, const hdv::HDV &b) {
//    hdv::dim_t dim = a.dim;
//    std::size_t dist = hdv::dist(a, b);
//    float difference = (float)dist/(float)dim;
//    //printf("similar(): Dist: %ld, dim: %ld diff: %f%%\n", dist, dim, difference);
//    // We consider that two vectors are similar if their difference is lower
//    // than 10 %.
//    return difference < 0.10;
//}

/*
 * Random HVs should be orthogonal to each other.
 */
TEST_CASE("Orthogonality of random vectors") {
    const int VECTORS = 100;
    std::vector<hdv::HDV>vectors;
    for (int i = 0; i < VECTORS; i++) {
        vectors.emplace_back(hdv::HDV(_DIM));
    }

    for (int i = 0; i < VECTORS; i++) {
        for (int j = 0; j < VECTORS; j++) {
            if (i != j) {
                const auto &a = vectors[i];
                const auto &b = vectors[j];
                REQUIRE(_is_orthogonal(a, b));
            }
        }
    }
}

/*
 * Given a set of HVs, the bundle operation on them must result in a HV that is
 * similar to all original HVs.
 */
TEST_CASE("Bundle") {
    std::vector<hdv::HDV> vectors;
    for (int i = 0; i < 11; i++) {
        vectors.emplace_back(hdv::HDV(_DIM));
    }
    hdv::HDV res = hdv::maj(vectors);
    hdv::dim_t dim = res.dim;
    for (int i = 0; i < vectors.size(); i++) {
        std::size_t dist = hdv::dist(res, vectors[i]);
        float difference = (float)dist/(float)dim;
        REQUIRE(!_is_orthogonal(res, vectors[i]));
    }
}

/*
 * Given a set of HVs, the binding operation on them must result in a HV that
 * is dissimilar to all original HVs.
 */
TEST_CASE("Binding") {
    hdv::HDV a(_DIM), b(_DIM), c(_DIM);

    auto check = [](const hdv::HDV &a, const hdv::HDV &b, const hdv::HDV &c) {
        for (std::size_t i = 0; i < c._data.size(); i++) {
            if (!(c._data[i] == (a._data[i] ^ b._data[i]))) {
                REQUIRE(false);
            }
        }
        REQUIRE(true);
    };

    SECTION("Mask test") {
        int bits_in_vec_t = sizeof(hdv::vec_t) * 8;
        // Bit masks
        hdv::vec_t p = 0, p_ = 0;

        // Create pattern "1010...10"
        for (int i = 0; i < bits_in_vec_t; i++) {
            if (i % 2) {
                p |= 1 << i;
            }
        }
        p_ = ~p;
        for (std::size_t i = 0; i < a._data.size(); i++) {
            a._data[i] = p;
            b._data[i] = p_;
        }

        c = a*b; // C should be a vector filled with 1s
        check(a, b, c);
    }
    SECTION("Random test") {
        c = a * b;
        check(a, b, c);
        REQUIRE(_is_orthogonal(a, c));
        REQUIRE(_is_orthogonal(b, c));
    }
}

/*
 * The permute operation must result in a vector that is orthogonal to the
 * original
 */
TEST_CASE("Permute") {

}
