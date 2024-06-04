#include <array>
#include <cstdint>

namespace bitmanip {
    bool get_bit(uint32_t val, uint32_t pos);

    /**
     * @brief Unpack bits into an array with size equal to the number of bits in
     * the input datatype.
     *
     * @param val: Bit packed value.
     * @return An array contaning the bits expanded in each dimension.
     */
    std::array<uint32_t, 32> unpack(uint32_t val);
    uint32_t threshold_pack(const std::array<uint32_t, 32>& acc, uint32_t threshold);

    void accumulate_unpacked(
        uint32_t val,
        std::array<uint32_t, 32> &acc
    );
}

