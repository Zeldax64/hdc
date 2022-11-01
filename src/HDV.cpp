#include "HDV.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <ostream>
#include <vector>

static std::ostream& print_hdv(
        std::ostream &os,
        const std::vector<hdv::vec_t> &vec) {
    for (auto i : vec) {
        os << std::bitset<32>(i) << " ";
    }
    return os << "\n\n";
}

// Returns the size of vec_t in bits
static int _sizeof_vec_t() {
    return sizeof(hdv::vec_t) * 8;
}

static bool _get_bit(hdv::vec_t val, int bit) {
    return (val & (1 << bit));
}

static int _popcount(hdv::vec_t val) {
    int pop = 0;
    for (int i = 0; i < _sizeof_vec_t(); i++) {
        pop += _get_bit(val, i) ? 1 : 0;
    }
    return pop;
}

hdv::HDV::HDV(dim_t dim, bool random) {
    int bits = _sizeof_vec_t();
    // Find how many vec_t elements we need in the vector to support
    // the given dimension. The number of elements is ceiled to the
    // first multiple of vec_t.
    int elements = dim / bits + (dim % bits != 0);
    this->_data.resize(elements);
    this->dim = this->_data.size()*bits;

    if (random) {
        // Populate the vector with random elements
        //std::srand(time(0));
        std::generate(this->_data.begin(), this->_data.end(), rand);
    }
}

hdv::HDV::~HDV() {}

void hdv::HDV::p(std::uint32_t times) {
    // Get HV's most significant bit (MSB)
    bool hv_msb = _get_bit(this->_data[0], _sizeof_vec_t()-1);
    bool next_msb;

    for (std::uint32_t p = 0; p < times; p++) {
        for (int i = 0; i < this->_data.size(); i++) {
            next_msb = _get_bit(this->_data[i+1], _sizeof_vec_t()-1);
            if (i == this->_data.size()-1) {
                next_msb = hv_msb;
            }
            this->_data[i] <<= 1 | (next_msb ? 1 : 0);
        }
    }
}

hdv::HDV hdv::HDV::operator*(const HDV &op) const {
    assert(this->dim == op.dim);

    HDV res(this->dim, false);
    for (int i = 0; i < this->_data.size(); i++) {
        res._data[i] = this->_data[i] ^ op._data[i];
    }

    return res;
}

std::ostream& operator<<(std::ostream &os, const hdv::HDV &v) {
    return print_hdv(os, v._data);
}

//bool hdv::HDV::operator[](hdv::dim_t index) {
//    // The HV data layout is a vector of vec_t type. The first
//    // dimension of the HV is the MSB of the first entry of the
//    // vector.
//    //      1     |     2     |...|     N     |
//    // Data:
//    // MSB-----LSB|MSB-----LSB|...|MSB-----LSB|
//    //  0 ..... 1 | 1 ..... 1 |...| 1 ..... 1 |
//    // Dimension:
//    //  0º..................................nº|
//    int bits_per_entry = _sizeof_vec_t();
//    hdv::vec_t entry = this->_data[index/bits_per_entry];
//    int bit_index = index % bits_per_entry;
//    bool bit = _get_bit(entry, bits_per_entry-bit_index);
//    return bit;
//}

int hdv::HDV::ones() {
    int ones = 0;
    for (int i = 0; i < this->_data.size(); i++) {
        ones += _popcount(this->_data[i]);
    }

    return ones;
}

std::vector<hdv::HDV> hdv::init_im(std::size_t entries, hdv::dim_t dim) {
    std::vector<hdv::HDV> im;

    for (std::size_t i = 0; i < entries; i++) {
        im.emplace_back(hdv::HDV(dim));
    }

    return im;
}


std::vector<hdv::HDV> hdv::init_cim(std::size_t entries, hdv::dim_t dim) {
    assert(entries > 1);

    std::vector<hdv::HDV> cim;
    cim.emplace_back(hdv::HDV(dim));
    hdv::dim_t index = 0;
    hdv::dim_t flips = dim / entries;

    for (int i = 1; i < entries; i++) {
        hdv::HDV prev = cim[i-1];
        cim.emplace_back(hdv::flip(prev, index, flips));
        index += flips;
    }

    return cim;
}

int hdv::am_search(const hdv::HDV &query, const std::vector<HDV> &am) {
    int am_index = 0;
    hdv::dim_t min_dist = am[0].dim;

    for (int i = 0; i < am.size(); i++) {
        hdv::dim_t new_dist = hdv::dist(query, am[i]);
        if (new_dist < min_dist) {
            min_dist = new_dist;
            am_index = i;
        }
    }

    return am_index;
}

void hdv::HDV::flip(dim_t index, dim_t flips) {
    for (dim_t i = 0; i < flips; i++) {
        // Get the _data entry that contains the current dimension index
        std::size_t data_index = (index+i) / _sizeof_vec_t();
        vec_t data_word = this->_data.at(data_index);
        // Flip the correspondent dim bit
        //data_word ^= 0xA >> ((index+i) % _sizeof_vec_t());
        std::size_t bit_index = (_sizeof_vec_t() - ((index+i) % _sizeof_vec_t()));
        data_word ^= 1 << bit_index;
        this->_data[data_index] = data_word;
    }
}

void hdv::HDV::invert() {
    this->flip(0, this->dim);
}

hdv::HDV hdv::flip(const hdv::HDV &v, hdv::dim_t index, hdv::dim_t flips) {
    hdv::HDV flipped = v;
    flipped.flip(index, flips);
    return flipped;
}

hdv::HDV hdv::invert(const HDV &v) {
    hdv::HDV inverse = v;
    inverse.invert();
    return inverse;
}

//static hdv::dim_t _dot(const hdv::HDV &v1, const hdv::HDV &v2) {
//    hdv::dim_t res = 0;
//    for (std::size_t i = 0; i < v1._data.size(); i++) {
//        res += _popcount(v1._data[i] & v2._data[i]);
//    }
//    return res;
//}
//
//static float _norm(const hdv::HDV &v) {
//    hdv::dim_t res = 0;
//    for (std::size_t i = 0; i < v._data.size(); i++) {
//        res += _popcount(v._data[i]);
//    }
//    return std::sqrt(res);
//}
//
//float hdv::cos(const hdv::HDV &v1, const hdv::HDV &v2) {
//    assert(v1.dim == v2.dim);
//    std::cout << _dot(v1, v2) << std::endl;
//    std::cout << _norm(v1) << std::endl;
//    std::cout << _norm(v2) << std::endl;
//    return (float)_dot(v1, v2) / (_norm(v1) * _norm(v2));
//}

hdv::dim_t hdv::dist(const HDV &op1, const HDV &op2) {
    assert(op1.dim == op2.dim);
    dim_t res = 0;
    for (int i = 0; i < op1._data.size(); i++) {
        res += _popcount(op1._data[i] ^ op2._data[i]);
    }
    return res;
}

hdv::HDV hdv::p(const HDV &op, std::uint32_t times) {
    HDV v = op;
    v.p(times);
    return v;
}

hdv::HDV hdv::maj(const HDV &op1, const HDV &op2) {
    assert(op1.dim == op2.dim);

    HDV zero(op1.dim, false);
    return maj(op1, op2, zero);
}

hdv::HDV hdv::maj(const HDV &op1, const HDV &op2, const HDV &op3) {
    const vec_t *a, *b, *c;
    dim_t dim = op1.dim;
    assert(dim == op2.dim && dim == op3.dim);
    HDV hdv(dim, false);

    for (int i = 0; i < hdv._data.size(); i++) {
        a = &op1._data[i];
        b = &op2._data[i];
        c = &op3._data[i];
        hdv._data[i] = (*a & *b) | (*b & *c) | (*c & *a);
    }

    return hdv;
}

hdv::HDV hdv::maj(const std::vector<hdv::HDV> &vectors) {
    int acc[sizeof(hdv::vec_t)*8];
    hdv::HDV res(vectors[0].dim, false);
    // The bit_group refers to the vec_t entries in the vector data
    const std::size_t bit_groups = vectors[0]._data.size();

    for (std::size_t i = 0; i < bit_groups; i++) {
        // Reset the accumulator
        for (int i = 0; i < sizeof(hdv::vec_t)*8; i++) { acc[i] = 0; }

        // Accumulate the bits of the same bit_group of all vectors
        for (const auto &hv : vectors) {
            hdv::vec_t bit_group = hv._data[i];
            // Unpack bits into accumulator
            for (std::size_t pos = 0; pos < _sizeof_vec_t(); pos++) {
                acc[pos] += _get_bit(bit_group, pos);
            }
        }

        // Write the result's vector bit_group
        hdv::vec_t res_bit_group = 0;
        for (std::size_t pos = 0; pos < _sizeof_vec_t(); pos++) {
            // Decide the bit majority of the accumulator entries
            int threshold = vectors.size() / 2;
            hdv::vec_t bit = acc[_sizeof_vec_t()-pos-1] > threshold;
            // Set bit
            res_bit_group <<= 1;
            res_bit_group |= bit;
        }
        res._data[i] = res_bit_group;
    }

    return res;
}

hdv::HDV hdv::mul(const std::vector<hdv::HDV> &vectors) {
    if (vectors.size() == 1) {
        return vectors[0];
    }

    hdv::HDV res = vectors[0];

    for (std::size_t i = 1; i < vectors.size(); i++) {
        res = res * vectors[i];
    }

    return res;
}
