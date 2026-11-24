# MegaRecurrentKdaMx

Ascend 950 experimental Stable ABI entry `npu_mega_recurrent_kda_mx`.
One mixed kernel runs the QKV MXFP8 GEMM on AIC and the causal convolution,
KDA recurrence, and paged state updates on AIV.

For M token positions, K input channels, and P local projection channels

| Input | Storage and layout |
| --- | --- |
| `x`, `weight` | E4M3FN bytes in UINT8 tensors, contiguous `[M,K]` and `[3P,K]` |
| `x_scale`, `weight_scale` | E8M0 bytes grouped by 32 channels in contiguous `[M,K/64,2]` and `[3P,K/64,2]` |
| `gate`, `beta` | BF16 `[M,P]` and `[M,heads]` from the caller's existing projection |
| `conv`, `state` | BF16 convolution storage and FP32 paged recurrence storage |
| `prefix`, `slots`, `conv_slots`, `accepted`, `has_initial` | Device INT32 metadata |

K is a multiple of 64 and P is 128 times the local head count. The operator
writes BF16 recurrence values to the supplied output buffer and updates
the provided convolution and recurrence storage in place. The caller
performs output normalization and the output projection.

## Execution

The Cube stage produces BF16 QKV into a workspace. Each AIV reads complete
QKV rows after the AIC producer barrier, then consumes its request/head
partition. The AIC waits for the paired AIVs to finish writing output and
state. The current replay's accepted counts and slot mappings are read
on device.
