#pragma once

#include <vector>

#include "types.hpp"

namespace hdc {
    template<typename T>
    class ContinuousItemMemory
    {
    public:
        ContinuousItemMemory(std::size_t size, dim_t dim) {
            this->_cim.emplace_back(T(dim));
            hdc::dim_t index = 0;
            hdc::dim_t flips = dim / (size-1);

            for (int i = 1; i < size; i++) {
                // Copy the previous iteration vector and invert it
                T v = this->_cim[i-1];
                v.invert(index, flips);
                this->_cim.emplace_back(v);
                index += flips;
            }
        }
        virtual ~ContinuousItemMemory() {};

        std::size_t size() const { return this->_cim.size(); }
        const T at(std::size_t pos) const {
            return this->_cim.at(pos);
        };

    private:
        std::vector<T> _cim;
    };
}
