#include <cassert>
#include <fstream>
#include <iostream>

#include "common.hpp"

std::vector<char> read_bin_file(const char *path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        assert(f.is_open());
    }

    std::size_t f_size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buffer;
    buffer.resize(f_size);
    f.read(buffer.data(), f_size);

    if (!f) {
        std::cerr << "Failed to read file" << std::endl;
        assert(false);
    }

    return buffer;
}
