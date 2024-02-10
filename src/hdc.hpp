#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "types.hpp"
#include "AssociativeMemory.hpp"
#include "ContinuousItemMemory.hpp"
#include "ItemMemory.hpp"
#include "Vector.hpp"

namespace hdc {
    // Vector data types
    using int32_t = Vector<int32_t>;
    using float_t = Vector<float>;
    using double_t = Vector<double>;
    using bin_t = Vector<bin_vec_t>;

    template<typename T>
    float dist(const T& v1, const T& v2) {
        return v1.dist(v2);
    }

    template<typename T>
    T add(const T& v1, const T& v2, const T& v3) {
        T res(v1);
        res.add(v2, v3);
        return res;
    }

    template<typename T>
    T add(const std::vector<T>& vectors) {
        return T::add(vectors);
    }

    template<typename T>
    T mul(const T& v1, const T& v2) {
        T res(v1);
        res.mul(v2);
        return res;
    }

    template<typename T>
    T mul(const std::vector<T>& vectors) {
        T res(vectors[0]);
        for (auto i = 1; i < vectors.size(); i++) {
            res.mul(vectors[i]);
        }
        return res;
    }

    template<typename T>
    T p(const T& v1, std::uint32_t times=1) {
        T res(v1);
        res.p(times);
        return res;
    }
}

