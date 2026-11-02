"""Fixed episode-level train/val split and OpenPI norm stats from local TOWER HDF5 (numeric only).

Split: per task variant, hold out 10% of episodes (seeded). Stats: model-space state (gripper closure)
and model-space action windows (joint delta vs measured state, gripper closure, tail hold labels) over
every valid start of every train episode. Only files whose size matches MANIFEST.jsonl are used.
Writes runs/prep/split.json, runs/prep/norm_stats/norm_stats.json (OpenPI format) and
runs/prep/prepare_report.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from jenga.hdf5_dataset import hold_window
from jenga.transforms import GRIPPER_FULL_WIDTH_M, to_model_actions, to_model_state

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 32


def episode_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        valid = f["sim/action_valid"][:]
        leader = f["leader/qpos"][:]
        state = f["observations/qpos"][:]
    last = int(np.flatnonzero(valid)[-1])
    starts = np.arange(last + 1)
    windows = np.stack([hold_window(leader, last, t, HORIZON) for t in starts])
    actions = to_model_actions(windows, state[starts], GRIPPER_FULL_WIDTH_M)
    return to_model_state(state[starts], GRIPPER_FULL_WIDTH_M), actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/prep")
    args = p.parse_args()
    by_task: dict[str, list[dict]] = {}
    skipped = []
    for line in (args.root / "MANIFEST.jsonl").read_text().splitlines():
        e = json.loads(line)
        f = args.root / e["path"]
        if f.is_file() and f.stat().st_size == e["bytes"]:
            by_task.setdefault(e["task"], []).append(e)
        else:
            skipped.append(e["path"])
    rng = np.random.default_rng(args.seed)
    split = {"dataset_root": str(args.root), "seed": args.seed, "val_frac": args.val_frac,
             "skipped_incomplete": skipped, "tasks": {}}
    for task, entries in sorted(by_task.items()):
        entries = sorted(entries, key=lambda e: e["path"])
        n_val = max(1, round(args.val_frac * len(entries)))
        val_idx = set(rng.choice(len(entries), n_val, replace=False).tolist())
        split["tasks"][task] = {
            "train": [e["path"] for i, e in enumerate(entries) if i not in val_idx],
            "val": [e["path"] for i, e in enumerate(entries) if i in val_idx],
            "train_frames": int(sum(e["frames"] for i, e in enumerate(entries) if i not in val_idx)),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")

    from openpi.shared import normalize

    train = [str(args.root / path) for t in split["tasks"].values() for path in t["train"]]
    state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
    n = 0
    with mp.Pool(args.workers) as pool:
        for state, actions in pool.imap_unordered(episode_arrays, train, chunksize=1):
            state_stats.update(state)
            action_stats.update(actions.reshape(-1, actions.shape[-1]))
            n += len(state)
    stats = {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    normalize.save(args.output_dir / "norm_stats", stats)
    report = {
        "train_episodes": len(train),
        "val_episodes": sum(len(t["val"]) for t in split["tasks"].values()),
        "train_frames": n,
        "skipped_incomplete": len(skipped),
        **{f"{k}_{s}": np.round(getattr(v, s), 4).tolist() for k, v in stats.items() for s in ("q01", "q99")},
    }
    (args.output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    print("PREPARE_DONE")


if __name__ == "__main__":
    main()
