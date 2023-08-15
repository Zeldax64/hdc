#pragma once

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.hpp"
#include "Vector.hpp"

namespace hdc {
    template<typename T>
    class BaseMemory
    {
    // Abstract base class
    protected:
        BaseMemory()=default;
        BaseMemory(const std::vector<T>& data) : _data(data) {};
        BaseMemory(const std::string& path) { this->load(path); };
        BaseMemory(const char* path) { this->load(path); };
        virtual ~BaseMemory()=default;

    public:
        std::size_t size() const { return this->_data.size(); }

        const T at(std::size_t pos) const {
            return this->_data.at(pos);
        };

        void save(const std::string& path) const { this->save(path.c_str()); }

        void save(const char* path) const {
            std::ofstream output_file(path);

            if (!output_file.is_open()) {
                std::string msg("Error when opening file: ");
                msg += path;
                throw std::runtime_error(msg);
            }

            for (const auto& v : this->_data) {
                output_file << v << "\n";
            }
        }

        void load(const std::string& path) { this->load(path.c_str()); }
        void load(const char* path) {
            // Check if path is valid and open file
            if (!std::filesystem::is_regular_file(path)) {
                std::string msg = ("Given model path is not a file!");
                msg += " Path: "+ std::string(path);
                throw std::runtime_error(msg);
            }

            std::ifstream input_file(path);

            if (!input_file.is_open()) {
                std::string msg("Error when opening file: ");
                msg += path;
                throw std::runtime_error(msg);
            }

            // Parse
            std::string line;
            while (!input_file.eof()) {
                std::getline(input_file, line);
                // The files should not contain any empty lines
                if (line.empty()) {
                    break;
                }
                // Create a new hdc::Vector inside _data from the string in line
                this->_data.emplace_back(line);
            }
        }


    protected:
        std::vector<T> _data; /*! Inner data representation. */
    };

}
