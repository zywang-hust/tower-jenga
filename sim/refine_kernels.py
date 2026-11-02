"""Triton kernels for sim/refine.py's inference glue on channels-last bf16 activations [N, C, H, W]: the 2x average
pool and the 2x bilinear upsample (align_corners=False) written straight into the first channels of the skip
concat. Arithmetic follows ATen's kernels (fp32, exact weights, one rounding), so outputs are bit-identical to
F.avg_pool2d / F.interpolate + torch.cat; they read and write each byte once (ATen: 21 + 7.6 + 7 ms of the
87 ms net at 16 envs, now about 7 ms; see sim/refine.py for the net-level equivalence).
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jenga.paths import CUDA_HOME  # noqa: E402

BP, BC = 32, 32  # pixels x channels per program (8: a third of the bandwidth at 640x480)


def _nhwc(x: torch.Tensor):
    assert x.is_contiguous(memory_format=torch.channels_last) and x.dtype == torch.bfloat16, (x.shape, x.stride(), x.dtype)
    return x.shape


@triton.jit
def _up_cat_kernel(x_ptr, s_ptr, o_ptr, H, W, C1, C2, BP: tl.constexpr, BC: tl.constexpr):
    pid, cb = tl.program_id(0), tl.program_id(1)
    W2 = 2 * W
    row = pid // tl.cdiv(W2, BP)  # n * 2H + y
    xs = (pid % tl.cdiv(W2, BP)) * BP + tl.arange(0, BP)
    n, y = row // (2 * H), row % (2 * H)
    c = cb * BC + tl.arange(0, BC)
    out = o_ptr + (row * W2 + xs)[:, None] * (C1 + C2) + c[None, :]
    if cb * BC < C1:  # upsampled part. ATen: src = 0.5 (dst + 0.5) - 0.5 clamped at 0, i1 = i0 + (i0 < n - 1)
        sy = tl.maximum(0.5 * (y.to(tl.float32) + 0.5) - 0.5, 0.0)
        y0 = sy.to(tl.int32)
        ly = sy - y0.to(tl.float32)
        y1 = y0 + (y0 < H - 1).to(tl.int32)
        sx = tl.maximum(0.5 * (xs.to(tl.float32) + 0.5) - 0.5, 0.0)
        x0 = sx.to(tl.int32)
        lx = sx - x0.to(tl.float32)
        x1 = x0 + (x0 < W - 1).to(tl.int32)
        m = (xs < W2)[:, None] & (c < C1)[None, :]
        base = x_ptr + n * H * W * C1
        a = tl.load(base + (y0 * W + x0)[:, None] * C1 + c[None, :], mask=m, other=0.0).to(tl.float32)
        b = tl.load(base + (y0 * W + x1)[:, None] * C1 + c[None, :], mask=m, other=0.0).to(tl.float32)
        cc = tl.load(base + (y1 * W + x0)[:, None] * C1 + c[None, :], mask=m, other=0.0).to(tl.float32)
        d = tl.load(base + (y1 * W + x1)[:, None] * C1 + c[None, :], mask=m, other=0.0).to(tl.float32)
        w0, w1 = (1.0 - lx)[:, None], lx[:, None]
        val = (1.0 - ly) * (w0 * a + w1 * b) + ly * (w0 * cc + w1 * d)
        tl.store(out, val.to(tl.bfloat16), mask=m)
    else:  # skip part, copied
        cs = c - C1
        m = (xs < W2)[:, None] & (cs < C2)[None, :]
        tl.store(out, tl.load(s_ptr + (row * W2 + xs)[:, None] * C2 + cs[None, :], mask=m, other=0.0), mask=m)


def up_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """cat([interpolate(x, size=skip.shape[-2:], mode="bilinear"), skip], 1) for x [N, C1, H, W], skip [N, C2, 2H, 2W]."""
    N, C1, H, W = _nhwc(x)
    C2 = _nhwc(skip)[1]
    assert skip.shape[-2:] == (2 * H, 2 * W) and C1 % BC == 0 and C2 % BC == 0
    out = torch.empty((N, C1 + C2, 2 * H, 2 * W), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    _up_cat_kernel[(N * 2 * H * triton.cdiv(2 * W, BP), (C1 + C2) // BC)](x, skip, out, H, W, C1, C2, BP=BP, BC=BC)
    return out


@triton.jit
def _pool_kernel(x_ptr, o_ptr, H, W, C, BP: tl.constexpr, BC: tl.constexpr):
    pid, cb = tl.program_id(0), tl.program_id(1)
    Wo, Ho = W // 2, H // 2
    row = pid // tl.cdiv(Wo, BP)
    xs = (pid % tl.cdiv(Wo, BP)) * BP + tl.arange(0, BP)
    n, y = row // Ho, row % Ho
    c = cb * BC + tl.arange(0, BC)
    m = (xs < Wo)[:, None] & (c < C)[None, :]
    base = x_ptr + n * H * W * C + c[None, :]
    p00 = tl.load(base + ((2 * y) * W + 2 * xs)[:, None] * C, mask=m, other=0.0).to(tl.float32)
    p01 = tl.load(base + ((2 * y) * W + 2 * xs + 1)[:, None] * C, mask=m, other=0.0).to(tl.float32)
    p10 = tl.load(base + ((2 * y + 1) * W + 2 * xs)[:, None] * C, mask=m, other=0.0).to(tl.float32)
    p11 = tl.load(base + ((2 * y + 1) * W + 2 * xs + 1)[:, None] * C, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + (row * Wo + xs)[:, None] * C + c[None, :], ((((p00 + p01) + p10) + p11) / 4.0).to(tl.bfloat16), mask=m)


def pool2(x: torch.Tensor) -> torch.Tensor:
    """F.avg_pool2d(x, 2) for x [N, C, H, W] with even H, W."""
    N, C, H, W = _nhwc(x)
    assert H % 2 == 0 and W % 2 == 0 and C % BC == 0
    out = torch.empty((N, C, H // 2, W // 2), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    _pool_kernel[(N * (H // 2) * triton.cdiv(W // 2, BP), C // BC)](x, out, H, W, C, BP=BP, BC=BC)
    return out


@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, n, C, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs % C, mask=m, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)  # ATen: conv output += bias in bf16 (fp32 math, one rounding)
    y = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))  # ATen GeluCUDAKernelImpl, approximate="none"
    tl.store(x_ptr + offs, y.to(tl.bfloat16), mask=m)


def bias_gelu_(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """In place: gelu(x + bias) for a bias-free convolution's output x [N, C, H, W] (channels-last bf16), the two
    ATen passes that follow a biased nn.Conv2d and nn.GELU in one read and one write, same roundings."""
    N, C, H, W = _nhwc(x)
    assert bias.dtype == torch.bfloat16 and bias.shape == (C,)
    n = x.numel()
    _bias_gelu_kernel[(triton.cdiv(n, BG),)](x, bias, n, C, BLOCK=BG, enable_fp_fusion=False, extern_libs=LIBDEVICE)
    return x


BG = 4096  # elements per bias_gelu program
# erff as ATen's GELU gets it: torch cu130 is built with CUDA 13.0, whose libdevice differs from Triton's bundled one
# (1 ulp on some inputs, visible after bf16 rounding in about 1 of 15000 values of the 256-channel layers)
LIBDEVICE = {"libdevice": str(CUDA_HOME / "nvvm/libdevice/libdevice.10.bc")}  # env CUDA_HOME, else nvcc on PATH
