#include "kernel_operator.h"
#include "mega_recurrent_kda_mx_tiling.h"
#include "projection_contract.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#define CATLASS_ARCH 3510
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "../../mega_recurrent_kda/op_kernel/arch35/recurrence.h"

namespace {
using namespace AscendC;
using kda::Bf;
constexpr uint32_t kDim = mega_recurrent_kda_mx_contract::kHeadDim;
constexpr uint32_t kState = kDim * kDim;
constexpr uint32_t kHistory = 3;

template <typename T>
__aicore__ inline GlobalTensor<T> At(GM_ADDR addr, uint64_t offset = 0) {
    GlobalTensor<T> tensor;
    tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addr) + offset);
    return tensor;
}

struct Inputs {
    using Address = GM_ADDR;
    Address qkv, gate, beta, convWeight;
    Address prefix, slots, convSlots, accepted, hasInitial, aLog, dtBias;
};

struct Outputs {
    using Address = GM_ADDR;
    Address values, conv, state;
};

__aicore__ inline bool ValidSlot(int32_t slot, uint32_t capacity) {
    return slot > 0 && uint32_t(slot) < capacity;
}

__aicore__ inline void Run(const Inputs &in, const Outputs &out, const MegaRecurrentKdaMxTiling &t) {
    Catlass::Arch::Resource<kda::Arch> resource;
    auto state = resource.ubBuf.GetBufferByByte<float>(0);
    auto qkv = resource.ubBuf.GetBufferByByte<float>(65536);
    auto correction = qkv[3 * kDim];
    auto decay = correction[kDim];
    auto result = decay[kDim];
    auto bias = result[kDim];
    auto weights = resource.ubBuf.GetBufferByByte<Bf>(69632);
    auto history = resource.ubBuf.GetBufferByByte<Bf>(72704);
    auto projected = resource.ubBuf.GetBufferByByte<Bf>(75008);
    auto gate = projected[3 * kDim];
    auto beta = gate[kDim];
    auto rounded = beta[16];
    const uint32_t width = t.heads * kDim;
    const uint32_t channels = 3 * width;
    const bool speculative = t.candidates > 1;
    const auto prefixes = At<int32_t>(in.prefix);
    const auto slots = At<int32_t>(in.slots);
    const auto convIds = At<int32_t>(in.convSlots);
    const auto counts = At<int32_t>(in.accepted);
    const auto initial = At<int32_t>(in.hasInitial);

    for (uint32_t item = GetBlockIdx(); item < t.batch * t.heads; item += 2 * GetBlockNum()) {
        const uint32_t request = item / t.heads;
        const uint32_t head = item % t.heads;
        const int32_t begin = prefixes.GetValue(request);
        const int32_t end = prefixes.GetValue(request + 1);
        const auto convId = convIds.GetValue(request);
        const auto accepted = counts.GetValue(request);
        if (begin < 0 || end <= begin || uint32_t(end) > t.tokens ||
            !ValidSlot(convId, t.convSlots) || accepted < 1 || uint32_t(accepted) > t.candidates ||
            (speculative && uint32_t(end - begin) > t.candidates)) {
            continue;
        }
        const uint64_t table = uint64_t(request) * t.candidates;
        const auto sourceId = slots.GetValue(table + accepted - 1);
        if (!ValidSlot(sourceId, t.stateSlots)) continue;
        const uint64_t convBase = uint64_t(convId) * t.convStride;
        const uint64_t stateHead = uint64_t(head) * kState;
        if (initial.GetValue(request)) {
            DataCopy(state, At<float>(out.state, uint64_t(sourceId) * t.stateStride + stateHead), kState);
            for (uint32_t row = 0; row < kHistory; ++row) {
                for (uint32_t kind = 0; kind < 3; ++kind) {
                    const uint64_t src = convBase + uint64_t(accepted - 1 + row) * channels +
                        kind * width + head * kDim;
                    DataCopy(history[(row * 3 + kind) * kDim], At<Bf>(out.conv, src), kDim);
                }
            }
        } else {
            Duplicate(state, 0.f, kState);
            Duplicate(history.ReinterpretCast<uint16_t>(), uint16_t(0), kHistory * 3 * kDim);
        }
        for (uint32_t kind = 0; kind < 3; ++kind) {
            DataCopyPad(weights[kind * 4 * kDim], At<Bf>(in.convWeight, kind * width + head * kDim),
                DataCopyExtParams{4, 2 * kDim, (channels - kDim) * 2, 0, 0}, DataCopyPadExtParams<Bf>{});
        }
        DataCopy(bias, At<float>(in.dtBias, head * kDim), kDim);
        const float aLog = At<float>(in.aLog).GetValue(head);
        PipeBarrier<PIPE_ALL>();

        // The speculative convolution stores two prior rows followed by all candidate projections.
        if (speculative) {
            for (uint32_t row = 0; row < 2; ++row) {
                for (uint32_t kind = 0; kind < 3; ++kind) {
                    const uint64_t dst = convBase + uint64_t(row) * channels + kind * width + head * kDim;
                    DataCopy(At<Bf>(out.conv, dst), history[((row + 1) * 3 + kind) * kDim], kDim);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t row = begin; row < end; ++row) {
            const uint32_t offset = uint32_t(row - begin);
            for (uint32_t kind = 0; kind < 3; ++kind) {
                const uint64_t src = uint64_t(row) * channels + kind * width + head * kDim;
                DataCopy(projected[kind * kDim], At<Bf>(in.qkv, src), kDim);
            }
            const uint64_t outOffset = uint64_t(row) * width + head * kDim;
            DataCopy(gate, At<Bf>(in.gate, outOffset), kDim);
            DataCopyPad(beta, At<Bf>(in.beta, uint64_t(row) * t.heads + head),
                DataCopyExtParams{1, 2, 0, 0, 0}, DataCopyPadExtParams<Bf>{});
            PipeBarrier<PIPE_ALL>();
            for (uint32_t kind = 0; kind < 3; ++kind) {
                kda::vec::convolve(
                    (__ubuf__ Bf *)projected[kind * kDim].GetPhyAddr(),
                    (__ubuf__ Bf *)history[kind * kDim].GetPhyAddr(),
                    (__ubuf__ Bf *)weights[kind * 4 * kDim].GetPhyAddr(),
                    (__ubuf__ float *)qkv[kind * kDim].GetPhyAddr(),
                    offset % kHistory, kind < 2, kind == 0 ? t.scale : 1.f);
                if (speculative) {
                    const uint64_t dst = convBase + uint64_t(2 + offset) * channels + kind * width + head * kDim;
                    DataCopy(At<Bf>(out.conv, dst), projected[kind * kDim], kDim);
                }
            }
            PipeBarrier<PIPE_V>();
            kda::vec::step((__ubuf__ float *)state.GetPhyAddr(), (__ubuf__ float *)qkv.GetPhyAddr(),
                (__ubuf__ Bf *)gate.GetPhyAddr(), (__ubuf__ Bf *)beta.GetPhyAddr(),
                (__ubuf__ float *)bias.GetPhyAddr(), aLog, t.lower, (__ubuf__ float *)correction.GetPhyAddr(),
                (__ubuf__ float *)decay.GetPhyAddr(), (__ubuf__ float *)result.GetPhyAddr());
            PipeBarrier<PIPE_V>();
            Cast(rounded, result, RoundMode::CAST_RINT, kDim);
            PipeBarrier<PIPE_ALL>();
            DataCopy(At<Bf>(out.values, outOffset), rounded, kDim);

            if (speculative || row + 1 == end) {
                const auto destId = slots.GetValue(table + (speculative ? offset : 0));
                if (ValidSlot(destId, t.stateSlots)) {
                    DataCopy(At<float>(out.state, uint64_t(destId) * t.stateStride + stateHead), state, kState);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
        if (!speculative) {
            const uint32_t oldest = uint32_t(end - begin) % kHistory;
            for (uint32_t row = 0; row < kHistory; ++row) {
                for (uint32_t kind = 0; kind < 3; ++kind) {
                    const uint64_t dst = convBase + uint64_t(row) * channels + kind * width + head * kDim;
                    DataCopy(At<Bf>(out.conv, dst), history[(((oldest + row) % kHistory) * 3 + kind) * kDim], kDim);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// The weight has logical CATLASS shape [K,N] and physical order [N,K].
__aicore__ inline void ProjectMx(GM_ADDR x, GM_ADDR weight, GM_ADDR xScale,
                                GM_ADDR weightScale, GM_ADDR qkv, const MegaRecurrentKdaMxTiling &t) {
    using namespace Catlass;
    using namespace tla;
    using Fp8 = float8_e4m3_t;
    using E8m0 = float8_e8m0_t;
    using Row = layout::RowMajor;
    using Col = layout::ColumnMajor;
    using Arch = Catlass::Arch::Ascend950;
    const uint32_t m = t.tokens, n = 3 * t.heads * kDim, k = t.projectionK;
    const auto layoutA = MakeLayout<Fp8, Row>(m, k);
    const auto layoutB = MakeLayout<Fp8, Col>(k, n);
    auto scaleA = MakeMxScaleLayout<E8m0, Row, false>(m, k / mega_recurrent_kda_mx_contract::kMxGroup);
    auto scaleB = MakeMxScaleLayout<E8m0, Col, true>(k / mega_recurrent_kda_mx_contract::kMxGroup, n);
    const auto layoutC = MakeLayout<Bf, Row>(m, n);
    using TileCopy = Gemm::Tile::PackedMxTileCopyTla<
        Arch, Fp8, Row, Fp8, Col, E8m0, decltype(scaleA), E8m0, decltype(scaleB), Bf, Row, void>;
    using L1 = tla::Shape<Int<128>, Int<128>, Int<256>>;
    using L0 = tla::Shape<Int<128>, Int<128>, Int<128>>;
    using Block = Gemm::Block::BlockMmadTla<Gemm::MmadMx<Arch, true, 16>, L1, L0, Fp8, Fp8, Bf, void, TileCopy>;
    using Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using Kernel = Gemm::Kernel::MxMatmulTla<Block, void, Scheduler>;
    const Kernel::Params args{GemmCoord(m, n, k), x, layoutA, weight, layoutB, xScale, scaleA,
                              weightScale, scaleB, qkv, layoutC};
    Kernel{}.template operator()<AscendC::AIC>(args);
}
} // namespace
#endif

extern "C" __global__ __aicore__ void mega_recurrent_kda_mx(
    GM_ADDR x, GM_ADDR weight, GM_ADDR xScale, GM_ADDR weightScale, GM_ADDR gate, GM_ADDR beta,
    GM_ADDR convWeight, GM_ADDR convState, GM_ADDR recurrentState,
    GM_ADDR prefix, GM_ADDR slots, GM_ADDR convSlots, GM_ADDR accepted, GM_ADDR hasInitial, GM_ADDR aLog,
    GM_ADDR dtBias, GM_ADDR out, GM_ADDR convOut, GM_ADDR recurrentOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    REGISTER_TILING_DEFAULT(MegaRecurrentKdaMxTiling);
    GET_TILING_DATA_WITH_STRUCT(MegaRecurrentKdaMxTiling, data, tiling);
    AscendC::InitSocState();
    GM_ADDR qkv = AscendC::GetUserWorkspace(workspace);
    constexpr uint32_t vectorReady = 2;
    constexpr uint32_t vectorDone = 3;
    constexpr uint32_t qkvReady = 4;
    if ASCEND_IS_AIV {
        CrossCoreSetFlag<4, PIPE_MTE3>(vectorReady);
    }
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<4, PIPE_FIX>(vectorReady);
        ProjectMx(x, weight, xScale, weightScale, qkv, data);
        // An AIV consumes a complete QKV row across all producer tiles.
        Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
        CrossCoreSetFlag<4, PIPE_FIX>(qkvReady);
        CrossCoreWaitFlag<4, PIPE_FIX>(vectorDone);
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<4, PIPE_MTE3>(qkvReady);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        Run({qkv, gate, beta, convWeight, prefix, slots, convSlots, accepted, hasInitial, aLog, dtBias},
            {out, convOut, recurrentOut}, data);
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<4, PIPE_MTE3>(vectorDone);
    }
#endif
}
