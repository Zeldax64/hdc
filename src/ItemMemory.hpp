#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "BaseMemory.hpp"
#include "types.hpp"

namespace hdc {
    template<typename T>
    class ItemMemory : public BaseMemory<T>
    {
    public:
        ItemMemory(std::size_t size, dim_t dim) {
            for (auto i = 0; i < size; i++) {
                this->_data.emplace_back(T(dim));
            }
        }
        virtual ~ItemMemory()=default;
    };
}
