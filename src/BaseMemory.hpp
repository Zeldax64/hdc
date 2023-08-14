#pragma once


#include <fstream>
#include <vector>

#include "types.hpp"

namespace hdc {
    template<typename T>
    class BaseMemory
    {
    // Abstract base class
    protected:
        BaseMemory()=default;
        BaseMemory(const std::vector<T>& data) : _data(data) {};
        virtual ~BaseMemory()=default;

    public:
        std::size_t size() const { return this->_data.size(); }

        const T at(std::size_t pos) const {
            return this->_data.at(pos);
        };

        void save(const std::string& path) const { this->save(path.c_str()); }

        void save(const char* path) const {
            std::ofstream output_file(path);

            for (const auto& v : this->_data) {
                output_file << v << "\n";
            }
        }

        void load(const char* path) const {
            std::ifstream input_file(path);

            input_file << this->size() << "\n";
            // TODO: Create an approach to load models.
        }


    protected:
        std::vector<T> _data; /*! Inner data representation. */
    };

}
