#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "types.hpp"

namespace hdc {
    template<typename T>
    class ItemMemory
    {
    public:
        ItemMemory(std::size_t size, dim_t dim) {
            for (auto i = 0; i < size; i++) {
                this->_im.emplace_back(T(dim));
            }
        }
        virtual ~ItemMemory() {};

        std::size_t size() const { return this->_im.size(); }
        const T at(std::size_t pos) const {
            return this->_im.at(pos);
        };

    private:
        std::vector<T> _im;
    };
}
