#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mega_recurrent_kda_mx_tiling.h"
#include "../op_kernel/projection_contract.h"
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace {
using namespace mega_recurrent_kda_mx_contract;
constexpr int64_t dim = kHeadDim;

const gert::Shape *Shape(gert::TilingContext *ctx, size_t i) {
    const auto *shape = ctx->GetInputShape(i);
    if (!shape) return nullptr;
    const auto &origin = shape->GetOriginShape();
    return origin.GetDimNum() ? &origin : &shape->GetStorageShape();
}

bool Matches(gert::TilingContext *ctx, size_t i, std::initializer_list<int64_t> dims) {
    const auto *shape = Shape(ctx, i);
    if (!shape || shape->GetDimNum() != dims.size()) return false;
    size_t j = 0;
    for (auto d : dims) if (shape->GetDim(j++) != d) return false;
    return true;
}

ge::graphStatus Tile(gert::TilingContext *ctx) {
    if (!ctx) return ge::GRAPH_FAILED;
    const auto *x = Shape(ctx, 0);
    const auto *beta = Shape(ctx, 5);
    const auto *conv = Shape(ctx, 7);
    const auto *state = Shape(ctx, 8);
    const auto *slots = Shape(ctx, 10);
    if (!x || x->GetDimNum() != 2 || !beta || beta->GetDimNum() != 2 || !conv || conv->GetDimNum() != 3 ||
        !state || state->GetDimNum() != 4 || !slots || slots->GetDimNum() != 2) return ge::GRAPH_FAILED;
    const auto rows = beta->GetDim(0), heads = beta->GetDim(1);
    const auto k = x->GetDim(1);
    if (!ValidProjectionK(k) || x->GetDim(0) != rows) return ge::GRAPH_FAILED;
    const auto batch = slots->GetDim(0), candidates = slots->GetDim(1);
    const auto convSlots = conv->GetDim(0), history = conv->GetDim(1), stateSlots = state->GetDim(0);
    if (rows < 1 || rows > INT32_MAX || heads < 1 || heads > 128 ||
        batch < 1 || batch > INT32_MAX / heads || history < 3 || history > 32 ||
        candidates < 1 || candidates > history - 2 || convSlots < 2 || convSlots > INT32_MAX ||
        stateSlots < 2 || stateSlots > INT32_MAX) return ge::GRAPH_FAILED;
    const auto width = heads * dim;
    if (!Matches(ctx, 1, {3 * width, k}) || !Matches(ctx, 2, {rows, ScalePairs(k), 2}) ||
        !Matches(ctx, 3, {3 * width, ScalePairs(k), 2}) || !Matches(ctx, 4, {rows, width}) ||
        !Matches(ctx, 6, {4, 3 * width}) || !Matches(ctx, 7, {convSlots, history, 3 * width}) ||
        !Matches(ctx, 8, {stateSlots, heads, dim, dim}) || !Matches(ctx, 9, {batch + 1}) ||
        !Matches(ctx, 11, {batch}) || !Matches(ctx, 12, {batch}) || !Matches(ctx, 13, {batch}) ||
        !Matches(ctx, 14, {heads}) || !Matches(ctx, 15, {heads, dim})) return ge::GRAPH_FAILED;
    const auto *attrs = ctx->GetAttrs();
    if (!attrs) return ge::GRAPH_FAILED;
    const auto *convStride = attrs->GetAttrPointer<int64_t>(0);
    const auto *stateStride = attrs->GetAttrPointer<int64_t>(1);
    const auto *scale = attrs->GetAttrPointer<float>(2);
    const auto *lower = attrs->GetAttrPointer<float>(3);
    if (!convStride || *convStride < history * 3 * width || !stateStride || *stateStride < heads * dim * dim ||
        !scale || !std::isfinite(*scale) || !lower || !std::isfinite(*lower) ||
        *lower < -5 || *lower >= 0) return ge::GRAPH_FAILED;
    auto raw = ctx->GetRawTilingData();
    auto sizes = ctx->GetWorkspaceSizes(1);
    auto platform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    const auto cubeCores = platform.GetCoreNumAic();
    const auto vectorPairs = platform.GetCoreNumAiv() / 2;
    const auto cores = cubeCores < vectorPairs ? cubeCores : vectorPairs;
    if (!raw || !sizes || !cores || raw->GetCapacity() < sizeof(MegaRecurrentKdaMxTiling)) {
        return ge::GRAPH_FAILED;
    }
    MegaRecurrentKdaMxTiling tile{
        uint64_t(*convStride), uint64_t(*stateStride), uint32_t(rows), uint32_t(batch),
        uint32_t(heads), uint32_t(convSlots), uint32_t(stateSlots), uint32_t(history),
        uint32_t(candidates), uint32_t(k), *scale, *lower
    };
    std::memcpy(raw->GetData(), &tile, sizeof(tile));
    raw->SetDataSize(sizeof(tile));
    const auto projectionBytes = QkvWorkspaceBytes(rows, heads);
    sizes[0] = platform.GetLibApiWorkSpaceSize() + projectionBytes;
    ctx->SetBlockDim(cores);
    return ge::GRAPH_SUCCESS;
}

struct CompileInfo {};
ge::graphStatus Prepare(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }
} // namespace

IMPL_OP_OPTILING(MegaRecurrentKdaMx).Tiling(Tile).TilingParse<CompileInfo>(Prepare);
