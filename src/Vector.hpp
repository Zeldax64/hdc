#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.hpp"

namespace hdc {
    template<typename T>
    static std::vector<T> _unhex(const std::string& s) {
        int bytes = sizeof(T)*2;

        if ((s.size()*2) % bytes) {
            std::string msg("Failed to create Vector from string. ");
            msg += "The string is " + std::to_string(s.size()) +
                   " bytes long and the vector is multiple of " +
                   std::to_string(bytes) + " bytes";
            throw std::runtime_error(msg);
        }

        std::vector<T> v;
        for (size_t i = 0; i < s.size(); i += bytes) {
          auto substr = s.substr(i, bytes);
          T chr_int = std::stol(substr, nullptr, 16);
          v.push_back(chr_int);
        }

        return v;
    }

    template<typename T>
    static void _check_size(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size()) {
          throw std::runtime_error("Attempt to perform operation on vectors "
                                   "with different dimensions.");
        }
    }

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

        Vector(const std::string& str) { this->_data = _unhex<T>(str); }

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

        auto cbegin() const { return std::cbegin(this->_data); }
        auto cend() const { return std::cend(this->_data); }

    private:
        std::vector<T> _data;

        float _cos(const Vector& v) const {
            float magnitude_a = 0.0;
            float magnitude_b = 0.0;
            float dot_product = 0.0;

            for (auto i = 0; i < this->_data.size(); i++) {
                float a = this->_data[i];
                float b = v._data[i];
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
        Vector(dim_t dim, bool random=true) ;
        Vector(const std::string& str);

        virtual ~Vector()=default;

        dim_t size() const { return this->_dim; }
        dim_t hamming(const Vector<bin_vec_t>& rhs) const;
        float dist(const Vector<bin_vec_t>& rhs) const;
        void invert();
        void invert(dim_t start, dim_t inversions);
        int get(dim_t pos) const;
        void set(dim_t pos, int val);
        void p(std::uint32_t times);
        void add(const Vector& v1, const Vector& v2);
        static Vector<bin_vec_t> add(
                const std::vector<Vector<bin_vec_t>> vectors
                );
        void mul(const Vector& rhs);

        auto cbegin() const { return std::cbegin(this->_data); }
        auto cend() const { return std::cend(this->_data); }

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

template<typename VectorType>
std::ostream& operator<<(std::ostream& os, const hdc::Vector<VectorType> v) {
    for (auto it = v.cbegin(); it != v.cend(); it++) {
        os << std::hex << std::setw(8) << std::setfill('0') << *it;
    }

    return os;
}
