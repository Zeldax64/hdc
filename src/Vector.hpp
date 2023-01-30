#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "types.hpp"

namespace hdc {

    // Built-in type vectors
    template<typename T>
    class Vector
    {
    public:
        Vector(dim_t dim, bool random=true) {
            this->_data.resize(dim);
            if (random) {
                this->_fillRandom(this->_data);
            }
        }
        virtual ~Vector(){};

        dim_t size() const { return this->_data.size(); }

        float dist(const Vector& v) const {
            float c = this->_cos(v);
            // Adjust cosine value to be between 0.0 and 1.0 as required by
            // BaseVector::dist(). A value close to 0 means that both vectors
            // are similar
            return std::abs((1.0-c)/2.0);
        }

        void invert() {
            this->invert(0, this->size());
        }

        void invert(dim_t start, dim_t inversions) {
            for (auto i = 0; i < inversions; i++) {
                this->_data[start+i] = -this->_data.at(start+i);
            }
        }

        T get(std::size_t pos) const {
            return this->_data.at(pos);
        }

        void p(std::uint32_t times=1) {
            for (std::uint32_t i = 0; i < times; i++) {
                // Rotate right
                std::rotate(
                        this->_data.rbegin(),
                        this->_data.rbegin()+1,
                        this->_data.rend()
                        );
            }
        }

        void add(const Vector& v1, const Vector& v2) {
            for (std::size_t i = 0; i < this->size(); i++) {
                this->_data[i] += v1._data[i] + v2._data[i];
            }
        }

        static Vector<T> add(
                const std::vector<Vector<T>> vectors
                ) {
            auto dim = vectors[0].size();
            Vector<T> res(dim, false);

            for (std::size_t d = 0; d < dim; d++) {
                for (std::size_t i = 0; i < vectors.size(); i++) {
                    res._data[d] += vectors[i].get(d);
                }
            }

            return res;
        }

        void mul(const Vector& rhs) {
            for (std::size_t i = 0; i < this->size(); i++) {
                this->_data[i] *= rhs._data[i];
            }
        }

    private:
        std::vector<T> _data;

        float _cos(const Vector& v) const {
            float magnitude_a = 0.0;
            float magnitude_b = 0.0;
            float dot_product = 0.0;

            for (auto i = 0; i < this->_data.size(); i++) {
                auto a = this->_data[i];
                auto b = v._data[i];
                dot_product += a*b;
                magnitude_a += a*a;
                magnitude_b += b*b;
            }

            return dot_product / (std::sqrt(magnitude_a) * std::sqrt(magnitude_b));
        }

        static T _generateRandomNumber() {
            // Generate -1 or 1 randomly
            int val = rand()%2;
            if (val) {
             return (T)1;
            }
            return (T)-1;
        }

        void _fillRandom(std::vector<T>& v) {
            //Generate unit vectors
            std::generate(v.begin(), v.end(), _generateRandomNumber);
        }
    };

    // Binary vectors
    template<>
    class Vector<bin_vec_t>
    {
    public:
        Vector(dim_t dim, bool random=true) {
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
        virtual ~Vector(){};

        dim_t size() const { return this->_dim; }

        dim_t hamming(const Vector<bin_vec_t>& rhs) const {
            dim_t res = 0;
            for (int i = 0; i < this->_data.size(); i++) {
                res += _popcount(this->_data[i] ^ rhs._data[i]);
            }
            return res;
        }

        float dist(const Vector<bin_vec_t>& rhs) const {
            auto hamm_dist = this->hamming(rhs);

            return (float)hamm_dist/(float)this->size();
        }

        void invert() {
            for (std::size_t i = 0; i < this->_data.size(); i++) {
                this->_data[i] = ~this->_data[i];
            }
        }

        void invert(dim_t start, dim_t inversions) {
            for (auto i = 0; i < inversions; i++) {
                bool bit = this->get(start+i);
                this->set(start+i, (!bit) & ~0x0);
            }
        }

        int get(dim_t pos) const {
            auto bit_group = pos / _sizeof_vec_t();
            auto bit_position = pos % _sizeof_vec_t();
            auto data_word = this->_data.at(bit_group);
            return _get_bit_at_dim(data_word, bit_position);
        }

        void set(dim_t pos, int val) {
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

        void p(std::uint32_t times) {
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

        void add(const Vector& v1, const Vector& v2) {
            const bin_vec_t *a, *b, *c;

            for (int i = 0; i < this->_data.size(); i++) {
                a = &this->_data[i];
                b = &v1._data[i];
                c = &v2._data[i];
                this->_data[i] = (*a & *b) | (*b & *c) | (*c & *a);
            }
        }

        static Vector<bin_vec_t> add(
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
            auto vec_t_size = sizeof(hdc::bin_vec_t)*8;
            int acc[vec_t_size];
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
                    for (std::size_t pos = 0; pos < vec_t_size; pos++) {
                        acc[pos] += _get_bit(bit_group, pos);
                    }
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
                res._data[i] = res_bit_group;
            }

            return res;
        }

        void mul(const Vector& rhs) {
            for (int i = 0; i < this->_data.size(); i++) {
                this->_data[i] = this->_data[i] ^ rhs._data[i];
            }
        }

    private:
        dim_t _dim;
        std::vector<bin_vec_t> _data;

        // Return the size of vec_t in bits
        int _sizeof_vec_t() const { return sizeof(bin_vec_t) * 8; }

        // TODO: Test this function because it is possibly wrong or
        // possibly too complicated to use with Vector::p()
        //bool _get_bit(bin_vec_t val, int bit) const {
        //    return (val & (1 << bit));
        //}

        static bool _get_bit(bin_vec_t val, int bit) {
            return (val & (1 << bit));
        }

        // Get the bit considering the data word's MSB as the lower
        // dimension of the vector and data word's LSB as the higher
        // dimension.
        bool _get_bit_at_dim(bin_vec_t val, int bit) const {
            return val & (1 << (_sizeof_vec_t() - bit -1)) ? 1 : 0;
        }

        bin_vec_t _get_bit_group(dim_t pos) const { return pos / _sizeof_vec_t(); }

        bin_vec_t _get_bit_position(dim_t pos) const { return pos % _sizeof_vec_t(); }

        int _popcount(bin_vec_t val) const {
            int pop = 0;
            for (int i = 0; i < _sizeof_vec_t(); i++) {
                pop += _get_bit(val, i) ? 1 : 0;
            }
            return pop;
        }

        void _fillRandom(std::vector<bin_vec_t>& v) const {
            std::generate(v.begin(), v.end(), rand);
        }
    };
}
