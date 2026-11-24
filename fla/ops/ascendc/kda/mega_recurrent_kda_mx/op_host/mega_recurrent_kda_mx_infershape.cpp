#include "register/op_impl_registry.h"

namespace {
constexpr size_t inputs[] = {4, 7, 8};

ge::graphStatus InferShape(gert::InferShapeContext *context) {
    for (size_t i = 0; i < 3; ++i) {
        const auto *src = context->GetInputShape(inputs[i]);
        auto *dst = context->GetOutputShape(i);
        if (!src || !dst) return ge::GRAPH_FAILED;
        *dst = *src;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferType(gert::InferDataTypeContext *context) {
    for (size_t i = 0; i < 3; ++i) {
        context->SetOutputDataType(i, context->GetInputDataType(inputs[i]));
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(MegaRecurrentKdaMx).InferShape(InferShape).InferDataType(InferType);
