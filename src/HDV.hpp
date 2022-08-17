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
        void p(); // Permute
        HDV operator*(const HDV &op);
        //bool operator[](dim_t index);
        int ones();

        std::vector<vec_t> _data;
    };

    dim_t dist(const HDV &op1, const HDV &op2);
    HDV p(const HDV &op);
    HDV maj(const HDV &op1, const HDV &op2);
    HDV maj(const HDV &op1, const HDV &op2, const HDV &op3);
    HDV maj(const std::vector<hdv::HDV> &v);
}

std::ostream& operator<<(std::ostream &os, const hdv::HDV &v);
