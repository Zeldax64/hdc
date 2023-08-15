#pragma once

#include <vector>

#include "BaseMemory.hpp"
#include "types.hpp"

namespace hdc {
    template<typename T>
    class ContinuousItemMemory : public BaseMemory<T>
    {
    public:
        ContinuousItemMemory(std::size_t size, dim_t dim) {
            this->_data.emplace_back(T(dim));
            hdc::dim_t index = 0;
            hdc::dim_t flips = dim / (size-1);

            for (int i = 1; i < size; i++) {
                // Copy the previous iteration vector and invert it
                T v = this->_data[i-1];
                v.invert(index, flips);
                this->_data.emplace_back(v);
                index += flips;
            }
        }

        ContinuousItemMemory(const std::string& path) : BaseMemory<T>(path) {};
        ContinuousItemMemory(const char* path) : BaseMemory<T>(path) {};

        virtual ~ContinuousItemMemory()=default;
    };
}
