"""Learned refinement from sim/render.py's raster images to the official RTX frames: the residual the raster
model leaves (indirect light and its colour, contact occlusion, surface relief). A small U-Net per camera image
on sRGB, the G-buffer (camera-frame normal, distance) and a camera one-hot; output = input + residual.
Trained on train-split dataset pairs by sim/train_refine.py.

Inference runs the net in bf16 channels-last with two Triton kernels for the memory-bound glue: the 2x average
pool and the 2x bilinear upsample fused with the skip concat (sim/refine_kernels.py). Both reproduce ATen's
arithmetic (fp32 with exact weights, one rounding to bf16): at batch 1 the output is bit-identical to the eager
module, and at batch 8-24 it differs from it by 2e-5 of the uint8 pixels (max 2 levels), the same fraction by
which the eager module differs from itself across batch sizes (cuDNN picks convolution algorithms per batch).
Cost per 640x480 image on the H800 at 8-24 envs: 5.40 -> 3.91 ms. Each conv's bias add runs fused with its GELU
(bias_gelu_, same roundings: 31.3 -> 22.4 ms per 8 envs with the wider pool / upsample programs, bit-identical).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CAMERAS = 4
CHUNK = 32  # images per U-Net pass: 128 at 640x480 overflow upsample_bilinear2d's int32 indexing; output unchanged


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())


class Refine(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        self.widths = widths
        self.down = nn.ModuleList(_block(i, o) for i, o in zip((3 + 4 + CAMERAS,) + widths[:-1], widths))
        self.up = nn.ModuleList(_block(o + i, i) for i, o in zip(widths[:-1], widths[1:]))
        self.out = nn.Conv2d(widths[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, rgb: torch.Tensor, gbuf: torch.Tensor) -> torch.Tensor:
        """rgb [E, C, H, W, 3] sRGB in [0, 1], gbuf [E, C, H, W, 4] -> refined rgb, same shape."""
        E, C, H, W, _ = rgb.shape
        cam = torch.eye(CAMERAS, device=rgb.device)[:C].view(1, C, 1, 1, CAMERAS).expand(E, C, H, W, CAMERAS)
        dist = gbuf[..., 3:].clamp(min=1e-3).log() * (gbuf[..., 3:] > 0)
        x = torch.cat([rgb * 2 - 1, gbuf[..., :3], dist, cam], -1).flatten(0, 1).permute(0, 3, 1, 2)
        res = torch.cat([self._residual(part) for part in x.split(CHUNK)])
        return (rgb + res.float().permute(0, 2, 3, 1).reshape(E, C, H, W, 3)).clamp(0, 1)

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:  # inference: bf16 weights (load), ATen convs, Triton pool / upsample + concat / bias + GELU
            from refine_kernels import bias_gelu_, pool2, up_cat

            def blk(seq, x):  # conv, GELU, conv, GELU; each conv's bias add fused into its GELU
                for conv in (seq[0], seq[2]):
                    x = bias_gelu_(F.conv2d(x, conv.weight, None, padding=1), conv.bias)
                return x
            x, skips = x.to(torch.bfloat16), []
            for k, seq in enumerate(self.down):
                x = blk(seq, x if k == 0 else pool2(x))
                skips.append(x)
            for seq, skip in zip(reversed(self.up), reversed(skips[:-1])):
                x = blk(seq, up_cat(x, skip))
            return self.out(x)
        with torch.autocast("cuda", torch.bfloat16):
            skips = []
            for k, blk in enumerate(self.down):
                x = blk(x if k == 0 else F.avg_pool2d(x, 2))
                skips.append(x)
            for blk, skip in zip(reversed(self.up), reversed(skips[:-1])):
                x = blk(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear"), skip], 1))
            return self.out(x)

    @classmethod
    def load(cls, path: str, device) -> "Refine":
        ck = torch.load(path, map_location=device)
        net = cls(tuple(ck["widths"])).to(device).eval()
        net.load_state_dict(ck["state"])
        net.requires_grad_(False)
        return net.to(torch.bfloat16).to(memory_format=torch.channels_last)
