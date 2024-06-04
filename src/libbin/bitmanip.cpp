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

    uint32_t _threshold_pack_gen(const std::array<uint32_t, 32>& acc, uint32_t threshold) {
        // Write the result's vector bit_group
        uint32_t word = 0;
        //constexpr int pack_size = acc.size();
        constexpr int pack_size = 32;
        for (std::size_t pos = 0; pos < pack_size; pos++) {
            // Decide the bit majority of the accumulator entries
            uint32_t bit = acc[pack_size-pos-1] > threshold;
            // Set bit
            word <<= 1;
            word |= bit;
        }

        return word;
    }

    uint32_t _threshold_pack_asm(const std::array<uint32_t, 32>& acc, uint32_t threshold) {
        constexpr int acc_size = 32;
        auto acc_ptr = (__m256i*) acc.data();

        constexpr int simd_size = sizeof(__m256i);
        constexpr int element_size = sizeof(decltype(*std::begin(acc)));
        constexpr int simd_elements = simd_size/element_size;
        std::array<uint32_t, simd_elements> avx_buffer;
        std::size_t loops = acc_size / simd_elements;

        // Hold the thresholded values. Each entry is either 32-bit 0xFFF...F or
        // 0x0.
        std::array<uint32_t, 32> threshold_buffer;
        auto threshold_buffer_ptr = (__m256i*) threshold_buffer.data();

        __m256i avx_threshold = _mm256_set1_epi32(threshold);
        auto buffer_ptr = (__m256i*) avx_buffer.data();
        uint64_t packed_word = 0;

        // Considering an AVX-256 register with 32-bit lanes. This shuffle mask
        // groups each LSB byte of each 32-bit word in the lowest word of
        // 128-bit lanes.
        // Example:
        // --- 128b --- --- 128b ---
        // L7|L6|L5|L4||L3|L2|L1|L0
        //           V           V
        // X|X|X|L7L6L5L4||X|X|X|L3L2L1L0
        // The "X" in the other lanes represent "don't care".
        constexpr int8_t shuffle_mask_data[32] = {
        // Lower 128 bits
             0,  4,  8, 12,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            // Upper 128 bits
             0,  4,  8, 12,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
            -1, -1, -1, -1,
        };
        __m256i shuffle_mask = _mm256_lddqu_si256((__m256i*)shuffle_mask_data);

        constexpr int32_t permute_mask_data[8] = {
           // Lower 128 bits
            0, 4, -1, -1,
            // Upper 128 bits
            -1, -1, -1, -1
        };
        __m256i permute_mask = _mm256_lddqu_si256((__m256i*)permute_mask_data);
        // Iterate from top to bottom to keep bit ordering
        for (int i = 7; i >= 0; i--) {
            __m256i temp_acc = _mm256_lddqu_si256(acc_ptr+i);

            // Apply the threshold
            // Divide all 32-bit elements in the SIMD lanes by 2 by shifting
            // them right then compare if their value is greater than the
            // threshold.
            // Shift right logic with immediate. Shifted in numbers are 0
            //__m256i temp = _mm256_srli_epi32(temp_acc, 1);
            __m256i res = _mm256_cmpgt_epi32(temp_acc, avx_threshold);

            // The threshold result has 32-bit lanes. We move a byte from each
            // word to the lower 32-bit lane. Since shuffle wokrs parallel in
            // the two halves of a 256 register. We create two int with valid
            // bytes at the upper and lower halves, then permute everything to
            // the lower lane in the 256 register.
            __m256i grouped_bytes_2x32 = _mm256_shuffle_epi8(res, shuffle_mask);

            __m256i grouped_bytes_1x64 = _mm256_permutevar8x32_epi32(grouped_bytes_2x32, permute_mask);

            // Move the lower 64-bit of data to a register
            _mm256_storeu_si256(buffer_ptr, grouped_bytes_1x64);
            uint64_t temp = *(uint64_t*)buffer_ptr;

            // Parallel pack
            uint64_t ext_mask = 0x0101010101010101;
            uint64_t extracted_bits = _pext_u64(temp, ext_mask);
            packed_word <<= simd_elements;
            packed_word = packed_word | extracted_bits;
        }

        return static_cast<uint32_t>(packed_word);
    }

    uint32_t threshold_pack(const std::array<uint32_t, 32>& acc, uint32_t threshold) {
#ifdef __ASM_LIBBIN
        return _threshold_pack_asm(acc, threshold);
#else
        return _threshold_pack_gen(acc, threshold);
#endif
    }
}