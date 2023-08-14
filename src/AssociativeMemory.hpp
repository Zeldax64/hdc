#pragma once

#include <limits>
#include <vector>

#include "BaseMemory.hpp"
#include "types.hpp"
#include "Vector.hpp"

namespace hdc {
    template<typename VectorType>
    class AssociativeMemory : public BaseMemory<VectorType>
    {
    public:
        AssociativeMemory()=default;
        AssociativeMemory(const std::vector<VectorType>& am)
            : BaseMemory<VectorType>::BaseMemory(am) {}
        virtual ~AssociativeMemory()=default;

        void clear() { this->_data.clear(); }
        void emplace_back(VectorType v) { this->_data.emplace_back(v); }

        std::size_t search(const VectorType& query) const {
            std::size_t am_index = 0;
            float min_dist = std::numeric_limits<float>::infinity();

            for (int i = 0; i < this->_data.size(); i++) {
                auto new_dist = query.dist(this->_data[i]);
                if (new_dist < min_dist) {
                    min_dist = new_dist;
                    am_index = i;
                }
            }

            return am_index;
        }
    };
}
