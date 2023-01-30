#include <cstddef>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

#include "ContinuousItemMemory.hpp"
#include "ItemMemory.hpp"
#include "hdc.hpp"
#include "types.hpp"

static const hdc::dim_t _DIM = 1000;

template<typename VectorType>
static bool _is_orthogonal(const VectorType &a, const VectorType &b) {
    float difference= hdc::dist(a, b);
    // We consider that two vectors are orthogonal if at least 40% of its
    // entries are different.
    return difference > 0.40;
}

/*
 * Random HVs should be orthogonal to each other.
 */
template<typename T>
static void _test_orthogonality(int entries, hdc::dim_t dim) {
    std::vector<T>vectors;
    for (int i = 0; i < entries; i++) {
        vectors.emplace_back(T(dim));
    }

    for (int i = 0; i < entries; i++) {
        for (int j = 0; j < entries; j++) {
            if (i != j) {
                const auto &a = vectors[i];
                const auto &b = vectors[j];
                REQUIRE(_is_orthogonal(a, b));
            }
        }
    }
}

TEST_CASE("Orthogonality of random vectors") {
    _test_orthogonality<hdc::bin_t>(100, _DIM);
    _test_orthogonality<hdc::int32_t>(50, _DIM);
    _test_orthogonality<hdc::float_t>(50, _DIM);
    _test_orthogonality<hdc::double_t>(50, _DIM);
}

/*
 * Given a set of HVs, the bundle operation on them must result in a HV that is
 * similar to all original HVs.
 */
template<typename T>
static void _test_bundle(std::size_t entries, hdc::dim_t dim) {
    std::vector<T> vectors;
    for (auto i = 0; i < entries; i++) {
        vectors.emplace_back(T(dim));
    }
    T res = hdc::add(vectors);
    for (auto i = 0; i < vectors.size(); i++) {
        REQUIRE(!_is_orthogonal(res, vectors[i]));
    }
}

TEST_CASE("Bundle") {
    _test_bundle<hdc::bin_t>(5, _DIM);
    _test_bundle<hdc::int32_t>(5, _DIM);
    _test_bundle<hdc::float_t>(5, _DIM);
    _test_bundle<hdc::double_t>(5, _DIM);
}

/*
 * Given a set of HVs, the binding operation on them must result in a HV that
 * is dissimilar to all original HVs.
 */
template<typename T>
static void _test_bind(std::size_t entries, hdc::dim_t dim) {
    std::vector<T> vectors;
    for (auto i = 0; i < entries; i++) {
        vectors.emplace_back(T(dim));
    }
    T res = hdc::mul(vectors);
    for (auto i = 0; i < vectors.size(); i++) {
        REQUIRE(_is_orthogonal(res, vectors[i]));
    }

}

TEST_CASE("Binding") {
    _test_bind<hdc::bin_t>(5, _DIM);
    _test_bind<hdc::int32_t>(5, _DIM);
    _test_bind<hdc::float_t>(5, _DIM);
    _test_bind<hdc::double_t>(5, _DIM);
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
template<typename T>
static void _test_im(unsigned int entries, hdc::dim_t dim) {
    auto im = hdc::ItemMemory<T>(entries, dim);

    for (int i = 0; i < im.size(); i++) {
        for (int j = 0; j < im.size(); j++) {
            if (i != j) {
                REQUIRE(_is_orthogonal(im.at(i), im.at(j)));
            }
        }
    }
}

TEST_CASE("Item Memory") {
    _test_im<hdc::bin_t>(100, _DIM);
    _test_im<hdc::int32_t>(100, _DIM);
    _test_im<hdc::float_t>(100, _DIM);
    _test_im<hdc::double_t>(100, _DIM);
}

/*
 * Initialization of a Continuous Item Memory (CIM) must contain subsquent
 * vectors that are similar to each other.
 */
template<typename T>
static void _test_cim(unsigned int entries, hdc::dim_t dim) {
    auto cim = hdc::ContinuousItemMemory<T>(entries, _DIM);

    // Let's consider that a vector is similar to its five subsequent neighbours
    for (int i = 0; i < cim.size(); i++) {
        for (int j = i; j < cim.size() && j < i+5; j++) {
            REQUIRE(!_is_orthogonal(cim.at(i), cim.at(j)));
        }
    }

    // Test disimilarity between distant items in CIM
    for (int i = 0; i < 10; i++) {
        REQUIRE(_is_orthogonal(cim.at(i), cim.at(cim.size()-i-1)));
    }
}

TEST_CASE("Continuous Item Memory") {
    _test_cim<hdc::bin_t>(100, _DIM);
    _test_cim<hdc::int32_t>(100, _DIM);
    _test_cim<hdc::float_t>(100, _DIM);
    _test_cim<hdc::double_t>(100, _DIM);
}

