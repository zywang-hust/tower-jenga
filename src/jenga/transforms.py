"""Action and state representation (docs/recipe.md §3).

GRIPPER_FULL_WIDTH_M is the audited value for the local release (scripts/audit.py: 531 episodes,
leader gripper commands span [0, 0.1] m and equal sim/gripper_requested_width_m exactly). The lifter is
stationary in every audited episode, so there is no lifter filter.
Layout: 16-D = left 7 joints + gripper, back 7 joints + gripper.
"""
from __future__ import annotations

import numpy as np

GRIPPERS = np.array([7, 15])
GRIPPER_FULL_WIDTH_M = 0.1
JOINTS = np.setdiff1d(np.arange(16), GRIPPERS)


def to_model_actions(actions: np.ndarray, state: np.ndarray, gripper_max: float) -> np.ndarray:
    """Absolute [H,16] commands -> model space: joint delta vs measured state, gripper closure 1 - w/w_max."""
    if gripper_max <= 0:
        raise ValueError("gripper_max must come from the data audit and be positive")
    out = np.array(actions, dtype=np.float32, copy=True)
    out[..., JOINTS] -= state[..., None, JOINTS]
    out[..., GRIPPERS] = 1.0 - out[..., GRIPPERS] / gripper_max
    return out


def from_model_actions(model_actions: np.ndarray, state: np.ndarray, gripper_max: float) -> np.ndarray:
    """Inverse of to_model_actions; serving applies this before converting grippers to wire meters."""
    if gripper_max <= 0:
        raise ValueError("gripper_max must come from the data audit and be positive")
    out = np.array(model_actions, dtype=np.float32, copy=True)
    out[..., JOINTS] += state[..., None, JOINTS]
    out[..., GRIPPERS] = (1.0 - out[..., GRIPPERS]) * gripper_max
    return out


def to_model_state(state: np.ndarray, gripper_max: float) -> np.ndarray:
    """Measured [..., 16] state -> model space: joints unchanged, gripper closure 1 - w/w_max (same as actions)."""
    if gripper_max <= 0:
        raise ValueError("gripper_max must come from the data audit and be positive")
    out = np.array(state, dtype=np.float32, copy=True)
    out[..., GRIPPERS] = 1.0 - out[..., GRIPPERS] / gripper_max
    return out


def norm_stats(samples: list[np.ndarray]) -> dict[str, np.ndarray]:
    """OpenPI-format statistics (mean/std/q01/q99) over [..., D] arrays, via OpenPI's RunningStats."""
    from openpi.shared.normalize import RunningStats

    stats = RunningStats()
    for x in samples:
        stats.update(np.asarray(x).reshape(-1, np.asarray(x).shape[-1]))
    s = stats.get_statistics()
    return {"mean": s.mean, "std": s.std, "q01": s.q01, "q99": s.q99}
