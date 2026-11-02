"""Per-trial orientation of the two external cameras from the static station background.

The official evaluator sends the external cameras upside down in some trials and not in others under identical
headers (docs/research.md 6.1.2). The decision used to come from jenga.offset (the nominal block layout explains
one orientation better). That template is about the movable blocks: on empty scenes both confidences are 0, and a
warm, bright official lighting case (gain 1.14, about 4850 K) dropped the right orientation's confidence from 0.96
to 0.10 below the turned one's 0.15 (runs/simdev/analysis/lighting/notes.md section 2).

This rule ignores the blocks. configs/orient_reference.npz holds, per external camera at 160x120, the median grey
step-0 frame of the train split (dataset orientation) and a mask of pixels that are static across all train
episodes of all variants and differ from the reference's own 180-degree turn (walls, curtains, table edges, robot
bases). Each 20x20 block of the mask votes with the Spearman rank correlation of the frame against the reference,
turned minus as given; blocks where the frame is flat (occluder, clipped light) do not vote. Rank correlation is
invariant to any monotone grey-level map, so gain, offset, gamma and (on this mostly neutral station) colour
temperature do not move it, and the block median tolerates a large occluder. Turn iff the median vote exceeds
`MARGIN`; below it, or with fewer than `MIN_BLOCKS` votes, keep the default orientation.
Validation: runs/simdev/analysis/lighting/orient/ (notes.md there).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).resolve().parents[2] / "configs/orient_reference.npz"
EXTERNAL = ("left_back_camera", "right_front_camera")
SIZE = (160, 120)
MARGIN = 0.10


BLOCK = 20          # 160x120 -> 8 x 6 blocks per camera
MIN_PIXELS = 40     # masked pixels a block needs to vote
MIN_BLOCKS = 12     # fewer voting blocks over both cameras: abstain


@lru_cache(maxsize=1)
def _reference():
    z = np.load(REFERENCE)
    ref, mask = z["median"].astype(np.float32), z["mask"].astype(bool)
    blocks = []  # per camera: [(slice, block mask, ranked reference)]
    for c in range(len(EXTERNAL)):
        cam = []
        for y in range(0, SIZE[1], BLOCK):
            for x in range(0, SIZE[0], BLOCK):
                sl = np.s_[y:y + BLOCK, x:x + BLOCK]
                m = mask[c][sl]
                r = _ranks(ref[c][sl][m]) if m.sum() >= MIN_PIXELS else None
                if r is not None and np.any(r):
                    cam.append((sl, m, r))
        blocks.append(cam)
    return blocks


def _ranks(v: np.ndarray) -> np.ndarray:
    """Centred, unit-norm ranks with ties averaged: a flat patch (occluder, clipped highlight) gets one rank, so
    pixel order cannot leak into the score. All-equal input -> zeros."""
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    r = (start + (counts - 1) / 2.0)[inverse].astype(np.float32)
    r -= r.mean()
    n = np.linalg.norm(r)
    return r / n if n > 0 else r


def _grey(im) -> np.ndarray:
    im = np.asarray(im)
    if im.ndim == 3:
        im = cv2.cvtColor(np.ascontiguousarray(im[..., :3]).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.resize(im, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def scores(images: dict) -> dict:
    """Per block (20x20 px at 160x120) of both external cameras: Spearman correlation of the masked pixels with the
    reference, for the frame as given (a) and turned by 180 degrees (b). A block votes only if the frame's pixels
    there are not flat in both versions (an occluder or a clipped region carries no evidence). Returns the median
    vote b - a, the number of votes, and per-camera medians of a and b."""
    blocks = _reference()
    votes, per = [], []
    for c, cam in enumerate(EXTERNAL):
        g = _grey(images[cam])
        gt = g[::-1, ::-1]
        ca, cb = [], []
        for sl, m, r in blocks[c]:
            va, vb = g[sl][m], gt[sl][m]
            if np.ptp(va) < 2 or np.ptp(vb) < 2:
                continue
            a, b = float(_ranks(va) @ r), float(_ranks(vb) @ r)
            ca.append(a); cb.append(b); votes.append(b - a)
        per.append((float(np.median(ca)) if ca else 0.0, float(np.median(cb)) if cb else 0.0, len(ca)))
    return dict(vote=float(np.median(votes)) if votes else 0.0, n=len(votes), per_camera=per)


def decide_flip(variant: str, images: dict) -> tuple[bool, dict]:
    """images: the trial's first frame, cameras in dataset orientation as jenga.server receives them (external
    cameras already turned once). Returns (turn the external cameras again, diagnostics). `variant` is unused:
    the station background is the same for all variants."""
    try:
        s = scores(images)
    except Exception as exc:  # missing camera or malformed image: keep the default orientation
        return False, dict(rule="orient-static-v2", error=repr(exc))
    if s["n"] < MIN_BLOCKS:
        turn, why = False, "abstain_few_blocks"
    elif abs(s["vote"]) <= MARGIN:
        turn, why = False, "abstain_small_margin"
    else:
        turn, why = s["vote"] > 0, "vote"
    return bool(turn), dict(rule="orient-static-v2", vote=round(s["vote"], 4), blocks=s["n"],
                            per_camera=[[round(a, 4), round(b, 4), n] for a, b, n in s["per_camera"]], decision=why)
