"""Score-weighted variant sampling mix and contact emphasis (docs/recipe.md §5)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

# Official per-task score weights 1:1:1:3:3 (docs/evaluation.md), keyed by task number.
TASK_SCORE_WEIGHTS = {1: 1.0, 2: 1.0, 3: 1.0, 4: 3.0, 5: 3.0}


def variant_mix(frames: Mapping[str, float], score: Mapping[str, float]) -> dict[str, float]:
    """Target sampling probability per variant, proportional to sqrt(frame share x score share)."""
    total_frames, total_score = sum(frames.values()), sum(score[v] for v in frames)
    raw = {v: np.sqrt(frames[v] / total_frames * score[v] / total_score) for v in frames}
    norm = sum(raw.values())
    return {v: float(r / norm) for v, r in raw.items()}


def sample_weights(sample_variants: Sequence[str], score: Mapping[str, float]) -> np.ndarray:
    """Per-sample weights for a WeightedRandomSampler so variant v is drawn with variant_mix probability.

    Frame share is measured on the samples themselves, so it already reflects any window filtering.
    """
    variants, counts = np.unique(np.asarray(sample_variants), return_counts=True)
    mix = variant_mix(dict(zip(variants.tolist(), counts.tolist(), strict=True)), score)
    per_sample = {v: mix[v] / n for v, n in zip(variants.tolist(), counts.tolist(), strict=True)}
    return np.array([per_sample[v] for v in sample_variants], dtype=np.float64)


def emphasize(weights: np.ndarray, groups: np.ndarray, mask: np.ndarray, factor: float) -> np.ndarray:
    """Masked samples weigh `factor` times the others of their group; each group keeps its total weight."""
    out = np.where(mask, weights * factor, weights)
    for g in np.unique(groups):
        m = groups == g
        out[m] *= weights[m].sum() / out[m].sum()
    return out

