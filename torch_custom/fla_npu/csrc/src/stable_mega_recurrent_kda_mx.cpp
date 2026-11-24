#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

namespace {
using torch::stable::Tensor;
using namespace fla_npu_stable::stable;

constexpr const char *kSchemaMegaRecurrentKdaMx =
    "npu_mega_recurrent_kda_mx(Tensor x, Tensor weight, Tensor x_scale, Tensor weight_scale, "
    "Tensor gate, Tensor beta, Tensor conv_weight, Tensor(a!) conv, Tensor(b!) state, "
    "Tensor prefix, Tensor slots, Tensor conv_slots, Tensor accepted, Tensor has_initial, "
    "Tensor a_log, Tensor dt_bias, Tensor(c!) out, float scale, float lower, int stream) -> Tensor(c!)";

Tensor run_mega_recurrent_kda_mx(
    Tensor x, Tensor weight, Tensor x_scale, Tensor weight_scale,
    Tensor gate, Tensor beta, Tensor conv_weight, Tensor conv, Tensor state,
    Tensor prefix, Tensor slots, Tensor conv_slots, Tensor accepted, Tensor has_initial,
    Tensor a_log, Tensor dt_bias, Tensor out, double scale, double lower, int64_t stream)
{
    const auto input = meta_of(x);
    const auto c = meta_of(conv);
    const auto s = meta_of(state);
    FLA_STABLE_EXEC("aclnnMegaRecurrentKdaMx", input, stream,
        nd_tensor(input), nd_tensor(meta_of(weight)), nd_tensor(meta_of(x_scale)), nd_tensor(meta_of(weight_scale)),
        nd_tensor(meta_of(gate)), nd_tensor(meta_of(beta)), nd_tensor(meta_of(conv_weight)),
        nd_tensor(c), nd_tensor(s), nd_tensor(meta_of(prefix)), nd_tensor(meta_of(slots)), nd_tensor(meta_of(conv_slots)),
        nd_tensor(meta_of(accepted)), nd_tensor(meta_of(has_initial)), nd_tensor(meta_of(a_log)), nd_tensor(meta_of(dt_bias)),
        scalar(c.strides[0]), scalar(s.strides[0]), scalar(scale), scalar(lower),
        nd_out_tensor(meta_of(out)), nd_out_tensor(c), nd_out_tensor(s));
    return out;
}
} // namespace
