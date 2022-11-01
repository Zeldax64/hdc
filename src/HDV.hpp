#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

namespace hdv {
    using dim_t = std::uint64_t;
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
    int am_search(const hdv::HDV &query, const std::vector<HDV> &am);

    dim_t dist(const HDV &op1, const HDV &op2);
    HDV flip(const HDV &v, dim_t index, dim_t flips);
    HDV invert(const HDV &v);
    HDV p(const HDV &op, std::uint32_t times=1);
    HDV maj(const HDV &op1, const HDV &op2);
    HDV maj(const HDV &op1, const HDV &op2, const HDV &op3);
    HDV maj(const std::vector<hdv::HDV> &v);
}

std::ostream& operator<<(std::ostream &os, const hdv::HDV &v);
