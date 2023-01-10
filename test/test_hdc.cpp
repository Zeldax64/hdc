#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

#include "hdc.hpp"

static const hdc::dim_t _DIM = 1000;

static bool _is_orthogonal(const hdc::HDV &a, const hdc::HDV &b) {
    hdc::dim_t dim = a.dim;
    std::size_t dist = hdc::dist(a, b);
    float difference = (float)dist/(float)dim;
    // We consider that two vectors are orthogonal if at least 40% of its
    // entries are different.
    return difference > 0.40;
}

//TODO: Remove later since it is useless
//static bool _is_similar(const hdc::HDV &a, const hdc::HDV &b) {
//    hdc::dim_t dim = a.dim;
//    std::size_t dist = hdc::dist(a, b);
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
    std::vector<hdc::HDV>vectors;
    for (int i = 0; i < VECTORS; i++) {
        vectors.emplace_back(hdc::HDV(_DIM));
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
    std::vector<hdc::HDV> vectors;
    for (int i = 0; i < 11; i++) {
        vectors.emplace_back(hdc::HDV(_DIM));
    }
    hdc::HDV res = hdc::maj(vectors);
    hdc::dim_t dim = res.dim;
    for (int i = 0; i < vectors.size(); i++) {
        std::size_t dist = hdc::dist(res, vectors[i]);
        float difference = (float)dist/(float)dim;
        REQUIRE(!_is_orthogonal(res, vectors[i]));
    }
}

/*
 * Given a set of HVs, the binding operation on them must result in a HV that
 * is dissimilar to all original HVs.
 */
TEST_CASE("Binding") {
    hdc::HDV a(_DIM), b(_DIM), c(_DIM);

    auto check = [](const hdc::HDV &a, const hdc::HDV &b, const hdc::HDV &c) {
        for (std::size_t i = 0; i < c._data.size(); i++) {
            if (!(c._data[i] == (a._data[i] ^ b._data[i]))) {
                REQUIRE(false);
            }
        }
        REQUIRE(true);
    };

    SECTION("Mask test") {
        int bits_in_vec_t = sizeof(hdc::vec_t) * 8;
        // Bit masks
        hdc::vec_t p = 0, p_ = 0;

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

/*
 * Initialization of an Item Memory (IM) must contain orthogonal vectors
 */
TEST_CASE("Item Memory") {
    int entries = 100;
    auto im = hdc::init_im(entries, _DIM);

    for (int i = 0; i < im.size(); i++) {
        for (int j = 0; j < im.size(); j++) {
            if (i != j) {
                REQUIRE(_is_orthogonal(im[i], im[j]));
            }
        }
    }
}

/*
 * Initialization of a Continuous Item Memory (CIM) must contain subsquent
 * vectors that are similar to each other.
 */
TEST_CASE("Continuous Item Memory") {
    int entries = 100;
    auto cim = hdc::init_cim(entries, _DIM);

    // Let's consider that a vector is similar to the five subsequent vectors
    for (int i = 0; i < cim.size(); i++) {
        for (int j = i; j < cim.size() && j < i+5; j++) {
            REQUIRE(!_is_orthogonal(cim[i], cim[j]));
        }
    }

    // Test disimilarity between distant items in CIM
    for (int i = 0; i < 10; i++) {
        REQUIRE(_is_orthogonal(cim[i], cim[cim.size()-i-1]));
    }

}
