#pragma once

#include <limits>
#include <fstream>

#include <vector>

#include "types.hpp"
#include "Vector.hpp"

namespace hdc {
    template<typename VectorType>
    class AssociativeMemory
    {
    public:
        AssociativeMemory()=default;
        AssociativeMemory(std::vector<VectorType> am) : _am(am) {}
        virtual ~AssociativeMemory()=default;

        void clear() { this->_am.clear(); }
        void emplace_back(VectorType v) { this->_am.emplace_back(v); }

        std::size_t size() const {
            return this->_am.size();
        }
        std::size_t search(const VectorType& query) const {
            std::size_t am_index = 0;
            float min_dist = std::numeric_limits<float>::infinity();

            for (int i = 0; i < this->_am.size(); i++) {
                auto new_dist = query.dist(this->_am[i]);
                if (new_dist < min_dist) {
                    min_dist = new_dist;
                    am_index = i;
                }
            }

            return am_index;
        }

        void save(const std::string& path) const { this->save(path.c_str()); }

        void save(const char* path) const {
            std::ofstream output_file(path);

            for (const auto& v : this->_am) {
                output_file << v << "\n";
            }
        }

        void load(const char* path) const {
            std::ifstream input_file(path);

            input_file << this->size() << "\n";
            // TODO: Create an approach to load models.
        }

    private:
        std::vector<VectorType> _am;
    };
}
