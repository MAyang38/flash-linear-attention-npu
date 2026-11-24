#pragma once

#include <stdint.h>

namespace mega_recurrent_kda_mx_contract {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kMxGroup = 32;
constexpr uint32_t kMxPair = 2 * kMxGroup;
constexpr uint32_t kQkvPlanes = 3;
constexpr uint32_t kMaxProjectionK = 65536;

constexpr bool ValidProjectionK(int64_t columns)
{
    return columns >= kMxPair && columns <= kMaxProjectionK && columns % kMxPair == 0;
}

constexpr uint32_t ScalePairs(uint32_t columns)
{
    return columns / kMxPair;
}

constexpr uint64_t QkvWorkspaceBytes(uint32_t rows, uint32_t heads)
{
    return uint64_t(rows) * kQkvPlanes * heads * kHeadDim * sizeof(uint16_t);
}
} // namespace mega_recurrent_kda_mx_contract
