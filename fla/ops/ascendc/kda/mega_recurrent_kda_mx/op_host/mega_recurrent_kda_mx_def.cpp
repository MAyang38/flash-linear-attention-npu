#include "register/op_def_registry.h"

namespace ops {
class MegaRecurrentKdaMx : public OpDef {
public:
    explicit MegaRecurrentKdaMx(const char *name) : OpDef(name) {
        const auto nd = {ge::FORMAT_ND};
        const auto bf = {ge::DT_BF16};
        const auto fp = {ge::DT_FLOAT};
        const auto ints = {ge::DT_INT32};
        // Raw MX bytes are E4M3FN activations/weights and E8M0 scales.
        for (auto name : {"x_mx", "weight_mx", "x_scale", "weight_scale"}) {
            this->Input(name).ParamType(REQUIRED).DataTypeList({ge::DT_UINT8}).FormatList(nd);
        }
        for (auto name : {"gate", "beta", "conv_weight"}) {
            this->Input(name).ParamType(REQUIRED).DataTypeList(bf).FormatList(nd);
        }
        this->Input("conv_state").ParamType(REQUIRED).DataTypeList(bf).FormatList(nd).IgnoreContiguous();
        this->Input("recurrent_state").ParamType(REQUIRED).DataTypeList(fp).FormatList(nd).IgnoreContiguous();
        for (auto name : {"prefix", "slots", "conv_slots", "accepted", "has_initial"}) {
            this->Input(name).ParamType(REQUIRED).DataTypeList(ints).FormatList(nd);
        }
        this->Input("a_log").ParamType(REQUIRED).DataTypeList(fp).FormatList(nd);
        this->Input("dt_bias").ParamType(REQUIRED).DataTypeList(fp).FormatList(nd);
        this->Output("out").ParamType(REQUIRED).DataTypeList(bf).FormatList(nd);
        this->Output("conv_out").ParamType(REQUIRED).DataTypeList(bf).FormatList(nd).IgnoreContiguous();
        this->Output("recurrent_out").ParamType(REQUIRED).DataTypeList(fp).FormatList(nd).IgnoreContiguous();
        this->Attr("conv_slot_stride").AttrType(REQUIRED).Int();
        this->Attr("recurrent_slot_stride").AttrType(REQUIRED).Int();
        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("lower").AttrType(REQUIRED).Float();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(MegaRecurrentKdaMx);
} // namespace ops
