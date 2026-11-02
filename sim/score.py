"""Local stand-in for the official per-episode metrics (docs/evaluation.md §1.4), from block poses alone.

LHPR: share of planned operations that succeed. An operation is one entry of the episode's target_mapping (block and
goal); it succeeds when the block ends within the generator's placement tolerance of its goal (placement_m,
placement_degrees in config_json). Pushdown entries have no goal: the block succeeds when it left its place (_pushed_off). TSR: every operation succeeds. ODR: share of operations during which some non-target movable block
moved more than 10 mm; approximated here as the number of non-target blocks displaced more than 10 mm by the episode end per operation,
capped at 1.
The official scorer is not public: these are [H] approximations, good for comparing runs against each other.
"""
import numpy as np


def _angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    return float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))))


def _pushed_off(start: np.ndarray, end: np.ndarray) -> bool:
    """Pushdown operation: the block left its place, i.e. dropped half a layer or slid half a block width."""
    return bool(end[2] < start[2] - 0.015 or np.linalg.norm(end[:2] - start[:2]) > 0.025)


def score(setup: dict, final_poses: np.ndarray, target_mapping: list, config: dict) -> dict:
    initial = setup["initial_block_poses"]
    targets = [m["block_index"] for m in target_mapping]
    ok = [np.linalg.norm(final_poses[i, :3] - np.asarray(m["goal"][:3])) <= config["placement_m"] * 2
          and _angle_deg(final_poses[i, 3:], np.asarray(m["goal"][3:])) <= config["placement_degrees"]
          if "goal" in m else _pushed_off(initial[i], final_poses[i])
          for i, m in zip(targets, target_mapping)]
    others = [i for i in np.flatnonzero(~setup["block_fixed"]) if i not in targets]
    disturbed = sum(np.linalg.norm(final_poses[i, :3] - initial[i, :3]) > 0.010 for i in others)
    return dict(lhpr=float(np.mean(ok)), tsr=float(all(ok)), odr=min(1.0, disturbed / max(len(targets), 1)),
                ops=[bool(x) for x in ok])


def reach(blocks: np.ndarray, target_mapping: list) -> list:
    """Closest approach (m) of each goal operation's block to its goal over a trajectory [T, B, 7]; None for pushdown.
    A finer progress signal than LHPR: placed-then-lost and near misses still count as getting there."""
    return [float(np.linalg.norm(blocks[:, m["block_index"], :3] - np.asarray(m["goal"][:3]), axis=-1).min())
            if "goal" in m else None for m in target_mapping]
