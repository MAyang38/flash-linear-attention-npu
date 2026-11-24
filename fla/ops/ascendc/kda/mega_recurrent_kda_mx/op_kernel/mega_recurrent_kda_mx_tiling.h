#pragma once
#include <stdint.h>

struct MegaRecurrentKdaMxTiling {
    uint64_t convStride, stateStride;
    uint32_t tokens, batch, heads, convSlots, stateSlots, history, candidates, projectionK;
    float scale, lower;
};
