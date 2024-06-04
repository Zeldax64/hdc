#include "Vector.hpp"
#include "types.hpp"
#include <array>
#include <cstdint>
#include "libbin/bitmanip.hpp"

// Binary HDC: Binary Spatter Code (BSC) VSA
namespace hdc {
Vector<bin_vec_t>::Vector(dim_t dim, bool random) {
        int bits = _sizeof_vec_t();
        // Find how many vec_t elements we need in the vector to support
        // the given dimension. The number of elements is ceiled to the
        // first multiple of vec_t.
        int elements = dim / bits + (dim % bits != 0);
        this->_data.resize(elements);
        this->_dim = this->_data.size()*bits;

        if (random) {
            this->_fillRandom(this->_data);
        }
    }


    Vector<bin_vec_t>::Vector(const std::string& str) {
        this->_data = _unhex<bin_vec_t>(str);
        int bits = _sizeof_vec_t();
        this->_dim = this->_data.size()*bits;
    }

    dim_t Vector<bin_vec_t>::hamming(const Vector<bin_vec_t>& rhs) const {
        _check_size(this->_data, rhs._data);

        dim_t res = 0;
        for (int i = 0; i < this->_data.size(); i++) {
            res += _popcount(this->_data[i] ^ rhs._data[i]);
        }
        return res;
    }

    float Vector<bin_vec_t>::dist(const Vector<bin_vec_t>& rhs) const {
        auto hamm_dist = this->hamming(rhs);

        return (float)hamm_dist/(float)this->size();
    }

    void Vector<bin_vec_t>::invert() {
        for (std::size_t i = 0; i < this->_data.size(); i++) {
            this->_data[i] = ~this->_data[i];
        }
    }

    void Vector<bin_vec_t>::invert(dim_t start, dim_t inversions) {
        for (auto i = 0; i < inversions; i++) {
            bool bit = this->get(start+i);
            this->set(start+i, (!bit) & ~0x0);
        }
    }

    int Vector<bin_vec_t>::get(dim_t pos) const {
        auto bit_group = pos / _sizeof_vec_t();
        auto bit_position = pos % _sizeof_vec_t();
        auto data_word = this->_data.at(bit_group);
        return _get_bit_at_dim(data_word, bit_position);
    }

    void Vector<bin_vec_t>::set(dim_t pos, int val) {
        auto bit_group = _get_bit_group(pos);
        auto bit_position = _get_bit_position(pos);
        //auto data_word = this->_data[bit_group];
        auto data_word = this->_data.at(bit_group);
        bool bit = _get_bit_at_dim(data_word, bit_position);
        // Clear bit
        auto clear_mask = ~(1 << (_sizeof_vec_t() - bit_position-1));
        auto cleared_val = data_word & clear_mask;
        data_word &= clear_mask;
        // Set bit
        data_word |= (val << (_sizeof_vec_t() - bit_position-1));
        this->_data[bit_group] = data_word;
    }

    void Vector<bin_vec_t>::p(std::uint32_t times) {
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

    void Vector<bin_vec_t>::add(const Vector& v1, const Vector& v2) {
        const bin_vec_t *a, *b, *c;

        for (int i = 0; i < this->_data.size(); i++) {
            a = &this->_data[i];
            b = &v1._data[i];
            c = &v2._data[i];
            this->_data[i] = (*a & *b) | (*b & *c) | (*c & *a);
        }
    }

    Vector<bin_vec_t> Vector<bin_vec_t>::add(
            const std::vector<Vector<bin_vec_t>> vectors
            ) {
        // Simple implementation
        //auto dim = vectors[0].size();
        //std::vector<unsigned int> acc(dim);
        //auto threshold = vectors.size() / 2;
        //auto res = Vector(dim, false);

        //for (std::size_t d = 0; d < dim; d++) {
        //    for (std::size_t i = 0; i < vectors.size(); i++) {
        //        acc[d] += vectors[i].get(d);
        //    }
        //    res.set(d, acc[d] > threshold);
        //}

        //return res;

        // Explore data locality implementation
        constexpr uint32_t vec_t_size = sizeof(hdc::bin_vec_t)*8;
        std::array<uint32_t, vec_t_size> acc;
        Vector<bin_vec_t> res(vectors[0].size(), false);
        // The bit_group refers to the vec_t entries in the vector data
        const std::size_t bit_groups = vectors[0]._data.size();

        for (std::size_t i = 0; i < bit_groups; i++) {
            // Reset the accumulator
            for (int i = 0; i < vec_t_size; i++) { acc[i] = 0; }

            // Accumulate the bits of the same bit_group of all vectors
            for (const auto &hv : vectors) {
                hdc::bin_vec_t bit_group = hv._data[i];
                // Unpack bits into accumulator
                //for (std::size_t pos = 0; pos < vec_t_size; pos++) {
                //    acc[pos] += bitmanip::get_bit(bit_group, pos);
                //}
                bitmanip::accumulate_unpacked(bit_group, acc);
            }

            // Write the result's vector bit_group
            hdc::bin_vec_t res_bit_group = 0;
            for (std::size_t pos = 0; pos < vec_t_size; pos++) {
                // Decide the bit majority of the accumulator entries
                int threshold = vectors.size() / 2;
                hdc::bin_vec_t bit = acc[vec_t_size-pos-1] > threshold;
                // Set bit
                res_bit_group <<= 1;
                res_bit_group |= bit;
            }
            //uint32_t threshold = vectors.size() / 2;
            //auto res_bit_group = bitmanip::threshold_pack(acc, threshold);
            res._data[i] = res_bit_group;
        }

        return res;
    }

    void Vector<bin_vec_t>::mul(const Vector& rhs) {
        for (int i = 0; i < this->_data.size(); i++) {
            this->_data[i] = this->_data[i] ^ rhs._data[i];
        }
    }

};
