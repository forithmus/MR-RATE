"""FSDP mixed-precision compatibility for the pinned DINOv3 FP8 linear.

The upstream custom autograd function assumes that its activation and weight
arrive in the same dtype. FSDP2 intentionally keeps non-parameter activations
in FP32 at some block boundaries while presenting BF16 parameters, so its
un-cast weight-gradient matmul fails before producing a gradient. This keeps
the upstream FP8 forward/activation-gradient kernels and makes every returned
gradient match the corresponding forward input dtype.
"""

from __future__ import annotations

import torch
from dinov3.layers import fp8_linear


@torch.compiler.allow_in_graph
class FSDPMixedPrecisionFp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b_t, bias):
        amax_a = a.abs().amax(dim=-1, keepdim=True)
        amax_b_t = b_t.abs().amax(dim=-1, keepdim=True)
        out = fp8_linear.matmul(a, amax_a, b_t, amax_b_t, bias)

        ctx.a_requires_grad = a.requires_grad
        ctx.b_requires_grad = b_t.requires_grad
        ctx.bias_requires_grad = bias.requires_grad if bias is not None else False
        ctx.a_dtype = a.dtype
        ctx.b_dtype = b_t.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.save_for_backward(a, b_t, amax_b_t.max())
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b_t, amax_b = ctx.saved_tensors

        if ctx.a_requires_grad:
            b = b_t.t().contiguous()
            amax_grad_out = grad_out.abs().amax(dim=-1, keepdim=True)
            amax_b = amax_b.repeat(b.shape[0], 1)
            grad_a = fp8_linear.matmul(grad_out, amax_grad_out, b, amax_b, None)
            grad_a = grad_a.to(ctx.a_dtype)
        else:
            grad_a = None

        if ctx.b_requires_grad:
            # BF16 is the intended FSDP parameter compute dtype. Casting both
            # operands avoids the upstream BF16 @ FP32 failure and returns a
            # gradient compatible with the materialized parameter input.
            grad_b = grad_out.to(ctx.b_dtype).t() @ a.to(ctx.b_dtype)
        else:
            grad_b = None

        if ctx.bias_requires_grad:
            grad_bias = grad_out.sum(dim=0).to(ctx.bias_dtype)
        else:
            grad_bias = None
        return grad_a, grad_b, grad_bias


def enable_fsdp_mixed_precision_fp8() -> None:
    """Install the compatibility autograd function before model conversion."""
    fp8_linear.Fp8LinearFn = FSDPMixedPrecisionFp8LinearFn
