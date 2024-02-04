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
    using vec_t = std::uint32_t;

    class HDV
    {
    public:
        dim_t dim;
        HDV(dim_t dim, bool random=true);
        virtual ~HDV();

        // Operations
        void p(std::uint32_t times=1); // Permute
        HDV operator*(const HDV &op) const;
        //bool operator[](dim_t index);
        int ones();
        void flip(dim_t index, dim_t flips);
        void invert();

        std::vector<vec_t> _data;
    };

    std::vector<HDV> init_im(std::size_t entries, dim_t dim);
    std::vector<HDV> init_cim(std::size_t entries, dim_t dim);
    int am_search(const hdc::HDV &query, const std::vector<HDV> &am);

    dim_t dist(const HDV &op1, const HDV &op2);
    HDV flip(const HDV &v, dim_t index, dim_t flips);
    HDV invert(const HDV &v);
    HDV p(const HDV &op, std::uint32_t times=1);
    HDV maj(const HDV &op1, const HDV &op2);
    HDV maj(const HDV &op1, const HDV &op2, const HDV &op3);
    HDV maj(const std::vector<hdc::HDV> &v);
    HDV mul(const std::vector<hdc::HDV> &v);

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

std::ostream& operator<<(std::ostream &os, const hdc::HDV &v);
