"""Chunk post-processing for serving (docs/research.md §3).

The joint rows of each chunk are Savitzky-Golay smoothed along time (flow sampling leaves a second-difference
floor ~20x the demonstrations'), then the first `fade` commands are mixed linearly with the previous chunk's
prediction for the same frames. On R1 25000 held-out episodes the serving default (smooth 9, fade 2) takes the
boundary jump p95 from 19.9 to 13.8 mrad (demonstrations 13.3) and roughness from 5.6e-4 to 1.7e-4 rad, for
mae 0.82 -> 0.87 mrad (runs/eval/r1-25000/stitch.json). Keep one ChunkFader per session; reset it on episode reset.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

JOINT_COLUMNS = [*range(7), *range(8, 15)]


def smooth_joints(chunk: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay along time on the 14 joint columns; gripper columns are left untouched."""
    out = np.array(chunk, np.float32, copy=True)
    if window > order + 1 and len(out) >= window:
        out[:, JOINT_COLUMNS] = savgol_filter(out[:, JOINT_COLUMNS], window, order, axis=0, mode="interp")
    return out


def fade_weights(fade: int) -> np.ndarray:
    """Weight of the new chunk at steps 0..fade-1; it reaches 1 at step `fade`."""
    return np.arange(1, fade + 1, dtype=np.float32) / (fade + 1)


class ChunkFader:
    def __init__(self, fade: int = 2, smooth: int = 9, issued: int | None = None):
        """`issued`: rows of each chunk actually sent for execution (the server's k); None = the whole chunk."""
        if fade < 0:
            raise ValueError("fade must be non-negative")
        self.fade, self.smooth, self.issued = fade, smooth, issued
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, chunk: np.ndarray, executed: int) -> np.ndarray:
        """chunk [H, D] absolute commands; `executed` = commands of the previous chunk run since it was issued.

        The overlap with the previous chunk is only trusted for 1 <= executed <= issued (and < len(previous)):
        zero or negative (a re-sent frame, a step counter that went backwards) or more commands than were sent
        means the two chunks cannot be aligned, so the new chunk is used as is."""
        # Each chunk is smoothed on its own. Prepending the previous chunk's executed commands as left context
        # (rows 0..k-1 sit at the filter edge) was measured worse on R1 25000 (sg9fade2 mae 0.870 -> 0.893 mrad,
        # second difference 1.66e-4 -> 2.09e-4): the chunk-to-chunk jump (p95 ~20 mrad) becomes curvature.
        chunk = smooth_joints(chunk, self.smooth) if self.smooth else np.asarray(chunk, np.float32)
        aligned = (self._prev is not None and 1 <= executed < len(self._prev)
                   and (self.issued is None or executed <= self.issued))
        out = chunk.copy()
        if aligned:
            m = min(self.fade, len(self._prev) - executed, len(chunk))
            if m > 0:
                w = fade_weights(self.fade)[:m, None]
                out[:m] = (1 - w) * self._prev[executed : executed + m] + w * chunk[:m]
        self._prev = chunk  # overlap comes from the (smoothed) prediction, not the blended output
        return out
