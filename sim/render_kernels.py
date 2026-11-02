"""Triton kernels for sim/render.py. transform(): owner-frame vertices under per-image, per-owner matrices, the
gather+contract that torch.compile turned into a size-4 reduction kernel writing [B, V, 4] at a fifth of the
bandwidth (3.58 ms per pass at B=64, V=507k; 0.72 ms here). The products pair up as in Inductor's tree reduction
of `(M[:, owner] * pos[:, None]).sum(-1)` (lanes 0+2, 1+3) and fp contraction is off, so the clip positions are
bit-identical (64 x 507k x 4 values of both passes on the fixed states; the other orders differ at 1 ulp).
"""
import torch
import triton
import triton.language as tl

BV = 256  # vertices per program


@triton.jit
def _transform_kernel(pos_ptr, owner_ptr, m_ptr, o_ptr, V, n_owners, BV: tl.constexpr):
    b, vb = tl.program_id(0), tl.program_id(1)
    v = vb * BV + tl.arange(0, BV)
    m = v < V
    own = tl.load(owner_ptr + v, mask=m, other=0)
    p0 = tl.load(pos_ptr + v * 4 + 0, mask=m, other=0.0)
    p1 = tl.load(pos_ptr + v * 4 + 1, mask=m, other=0.0)
    p2 = tl.load(pos_ptr + v * 4 + 2, mask=m, other=0.0)
    p3 = tl.load(pos_ptr + v * 4 + 3, mask=m, other=0.0)
    mat = m_ptr + (b * n_owners + own) * 16
    for r in tl.static_range(4):
        m0 = tl.load(mat + r * 4 + 0, mask=m, other=0.0)
        m1 = tl.load(mat + r * 4 + 1, mask=m, other=0.0)
        m2 = tl.load(mat + r * 4 + 2, mask=m, other=0.0)
        m3 = tl.load(mat + r * 4 + 3, mask=m, other=0.0)
        tl.store(o_ptr + (b * V + v) * 4 + r, (m0 * p0 + m2 * p2) + (m1 * p1 + m3 * p3), mask=m)


def transform(pos: torch.Tensor, owner: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Vertices [V, 4] (fp32) of owners [V] (int) under matrices [B, O, 4, 4] (fp32) -> clip positions [B, V, 4]."""
    B, n_owners = M.shape[:2]
    V = len(pos)
    out = torch.empty((B, V, 4), dtype=torch.float32, device=pos.device)
    _transform_kernel[(B, triton.cdiv(V, BV))](pos, owner, M.contiguous(), out, V, n_owners, BV=BV, enable_fp_fusion=False)
    return out
