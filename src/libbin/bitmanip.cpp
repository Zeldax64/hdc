#include <array>
#include <cstdint>
#include <immintrin.h>

namespace bitmanip {
    bool get_bit(uint32_t val, uint32_t pos) {
        return (val & (1 << pos)) ? 1 : 0;
    }

    std::array<uint32_t, 32> _unpack_gen(uint32_t val) {
        // Unpack bits into an accumulator
        constexpr uint32_t datatype_size = sizeof(val);
        constexpr uint32_t elements = datatype_size * 8;
        std::array<uint32_t, elements> acc;

        for (std::size_t pos = 0; pos < elements; pos++) {
            acc[pos] = get_bit(val, pos);
        }

        return acc;
    }

    std::array<uint32_t, 32> _unpack_asm(uint32_t val) {
        // Unpack bits into an accumulator
        constexpr uint32_t datatype_size = sizeof(val);
        constexpr uint32_t elements = datatype_size * 8;
        std::array<uint32_t, elements> acc;

        // Cast val to 64-bit to allow using _pdep64
        auto a = static_cast<uint64_t>(val);
        constexpr auto unpack_size = sizeof(a);
        uint64_t mask = 0x0101010101010101;
        uint64_t unp_bits[2] = {0, 0};
        for (std::size_t pos = 0; pos < elements; pos += unpack_size) {
            // Unpack 8 bits into 8 8-bit elements
            unp_bits[0] = _pdep_u64(a, mask);
            a >>= unpack_size;

            // Convert the 8 elements of 8-bit to a 8 32-bit elements
            __m128i unp_8bits = _mm_lddqu_si128((__m128i*)&unp_bits);
            __m256i unp_32bits = _mm256_cvtepu8_epi32(unp_8bits);

            // Store the 8 32-bit elements in the return array
            __m256i* addr = (__m256i*)(acc.data()+pos);
            _mm256_storeu_si256(addr, unp_32bits);
        }

        return acc;
    }

    std::array<uint32_t, 32> unpack(uint32_t val) {
#ifdef __ASM_LIBBIN
        return _unpack_asm(val);
#else
        return _unpack_gen(val);
#endif
    }

    // TODO: Create accumulate_unpacked(val, arr)
    // Unpack the values of "val" and accumulate them in "arr"
    void _accumulate_unpacked_asm(
            uint32_t val,
            std::array<uint32_t, 32> &acc
        ) {
        auto unp = unpack(val);

        auto acc_ptr = (__m256i*) acc.data();
        auto unp_ptr = (__m256i*) unp.data();

        constexpr int simd_size = sizeof(__m256i);
        constexpr int element_size = sizeof(decltype(*std::begin(acc)));
        std::size_t loops = acc.size() / (simd_size/element_size);
        for (std::size_t i = 0; i < loops; i++) {
            __m256i temp_acc = _mm256_lddqu_si256(acc_ptr+i);
            __m256i temp_unp = _mm256_lddqu_si256(unp_ptr+i);

            // Adopting 32b int add since AVX256 does not dispose unsigned add
            // for 32b
            temp_acc = _mm256_add_epi32(temp_acc, temp_unp);

            _mm256_storeu_si256(acc_ptr+i, temp_acc);
        }
    }
    void _accumulate_unpacked_gen(
            uint32_t val,
            std::array<uint32_t, 32> &acc
        ) {
        auto unp = unpack(val);

        for (std::size_t i = 0; i < acc.size(); i++) {
            acc[i] = acc[i] + unp[i];
        }
    }

    void accumulate_unpacked(
            uint32_t val,
            std::array<uint32_t, 32> &acc
        ) {
#ifdef __ASM_LIBBIN
        _accumulate_unpacked_asm(val, acc);
#else
        _accumulate_unpacked_gen(val, acc);
#endif
    }
}