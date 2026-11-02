"""Answer the data questions of docs/recipe.md §1 on local TOWER HDF5 files (docs/recipe.md).

Gripper units and full width, lifter motion, valid-action layout, head/tail stillness, lighting spread,
recovery segments, prompts per variant. Only files whose size matches MANIFEST.jsonl are read (skips
in-flight downloads). Read-only. Writes runs/audit/r1_episodes.jsonl and runs/audit/r1_summary.json.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from jenga.transforms import GRIPPERS, JOINTS

ROOT = Path(__file__).resolve().parents[1]
STILL_RAD = 1e-4  # per-frame leader joint change below this counts as still
EXTERNAL = ("left_back_camera", "right_front_camera")


def _attr(f, name):
    v = f.attrs.get(name)
    return v.decode() if isinstance(v, bytes) else v


def _image_stats(raw) -> tuple[float, float]:
    img = np.asarray(Image.open(io.BytesIO(raw.tobytes())).convert("RGB"), np.float32)
    mean = img.reshape(-1, 3).mean(0)
    return float(mean @ [0.299, 0.587, 0.114]), float(mean[0] / max(mean[2], 1e-3))


def audit(entry: dict) -> dict:
    path = Path(entry["root"]) / entry["path"]
    with h5py.File(path, "r") as f:
        valid = f["sim/action_valid"][:]
        leader = f["leader/qpos"][:]
        obs = f["observations/qpos"][:]
        n_valid = int(valid.sum())
        last = int(np.flatnonzero(valid)[-1]) if n_valid else -1
        moving = np.abs(np.diff(leader[: last + 1, JOINTS], axis=0)).max(1) > STILL_RAD
        head = int(np.argmax(moving)) if moving.any() else last
        tail = int(np.argmax(moving[::-1])) if moving.any() else last
        out = {
            "task": entry["task"],
            "path": entry["path"],
            "frames": len(valid),
            "valid": n_valid,
            "valid_is_prefix": bool(valid[: last + 1].all()) and not valid[last + 1 :].any(),
            "instruction": _attr(f, "instruction"),
            "lifter_control_mode": _attr(f, "lifter_control_mode"),
            "lifter_is_moving_frac": float(f["observations/lifter/is_moving"][:].any(1).mean()),
            "lifter_cmd_delta_nonzero_frac": float((f["lifter/command_delta_mm"][:] != 0).any(1).mean()),
            "lifter_position_span_mm": np.ptp(f["observations/lifter/position_mm"][:], axis=0).tolist(),
            "lifter_height_mm": f["observations/lifter/physical_height_mm"][0].tolist(),
            "leader_grip_min": leader[:, GRIPPERS].min(0).tolist(),
            "leader_grip_max": leader[:, GRIPPERS].max(0).tolist(),
            "obs_grip_min": obs[:, GRIPPERS].min(0).tolist(),
            "obs_grip_max": obs[:, GRIPPERS].max(0).tolist(),
            "head_still_frames": head,
            "tail_still_frames": tail,
            "recovery_frac": float((f["sim/recovery_attempt"][:] > 0).mean()),
            "recovery_max": int(f["sim/recovery_attempt"][:].max()),
            "success_final": bool(f["sim/success"][-1]),  # set on the terminal row, after the last valid action
            "collapse_any": bool(f["sim/collapse"][:].any()),
            # Action deltas vs measured state over 32 steps: the scale the model will learn.
            "delta32_absmax": np.abs(leader[32 : last + 1, JOINTS] - obs[: last + 1 - 32, JOINTS]).max(0).tolist()
            if last >= 32 else None,
        }
        if "sim/gripper_requested_width_m" in f:
            req = f["sim/gripper_requested_width_m"][:]
            out["requested_width_max"] = req.max(0).tolist()
            out["leader_minus_requested_absmax"] = float(np.abs(leader[:, GRIPPERS] - req).max())
        lum, rb = [], []
        for cam in EXTERNAL:
            ds = f[f"observations/images/{cam}"]
            for t in (0, len(valid) // 2):
                lu, r = _image_stats(ds[t])
                lum.append(lu)
                rb.append(r)
        out["external_luma"] = float(np.mean(lum))
        out["external_r_over_b"] = float(np.mean(rb))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--output-dir", type=Path, default=ROOT / "runs/audit")
    args = p.parse_args()
    entries, skipped = [], []
    for line in (args.root / "MANIFEST.jsonl").read_text().splitlines():
        e = json.loads(line)
        f = args.root / e["path"]
        if f.is_file() and f.stat().st_size == e["bytes"]:
            entries.append({**e, "root": str(args.root)})
        else:
            skipped.append(e["path"])
    with mp.Pool(args.workers) as pool:
        rows = pool.map(audit, entries, chunksize=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "r1_episodes.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def agg(key, fn):
        return fn(np.array([r[key] for r in rows if r.get(key) is not None], dtype=np.float64))

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    summary = {
        "question": "data audit: gripper units/full width, lifter motion, stillness, lighting, recovery",
        "episodes_audited": len(rows),
        "episodes_skipped_incomplete": len(skipped),
        "frames_audited": int(sum(r["frames"] for r in rows)),
        "valid_is_prefix_all": all(r["valid_is_prefix"] for r in rows),
        "lifter_control_modes": sorted({str(r["lifter_control_mode"]) for r in rows}),
        "lifter_is_moving_frac_max": agg("lifter_is_moving_frac", np.max),
        "lifter_cmd_delta_nonzero_frac_max": agg("lifter_cmd_delta_nonzero_frac", np.max),
        "lifter_position_span_mm_max": agg("lifter_position_span_mm", np.max),
        "leader_grip_min": agg("leader_grip_min", np.min),
        "leader_grip_max": agg("leader_grip_max", np.max),
        "obs_grip_min": agg("obs_grip_min", np.min),
        "obs_grip_max": agg("obs_grip_max", np.max),
        "requested_width_max": agg("requested_width_max", np.max) if any("requested_width_max" in r for r in rows) else None,
        "leader_minus_requested_absmax": agg("leader_minus_requested_absmax", np.max)
        if any("leader_minus_requested_absmax" in r for r in rows) else None,
        "delta32_absmax_rad": agg("delta32_absmax", np.max),
        "success_final_frac": agg("success_final", np.mean),
        "collapse_any_frac": agg("collapse_any", np.mean),
        "per_task": {
            t: {
                "episodes": len(rs),
                "frames": int(sum(r["frames"] for r in rs)),
                "instructions": sorted({r["instruction"] for r in rs}),
                "head_still_s_median": float(np.median([r["head_still_frames"] for r in rs]) / 30),
                "tail_still_s_median": float(np.median([r["tail_still_frames"] for r in rs]) / 30),
                "tail_still_s_max": float(np.max([r["tail_still_frames"] for r in rs]) / 30),
                "recovery_episode_frac": float(np.mean([r["recovery_max"] > 0 for r in rs])),
                "recovery_frame_frac": float(np.mean([r["recovery_frac"] for r in rs])),
                "luma_p5_p95": np.percentile([r["external_luma"] for r in rs], [5, 95]).tolist(),
                "r_over_b_p5_p95": np.percentile([r["external_r_over_b"] for r in rs], [5, 95]).tolist(),
                "lifter_height_mm_range": [float(np.min([r["lifter_height_mm"] for r in rs])),
                                           float(np.max([r["lifter_height_mm"] for r in rs]))],
            }
            for t, rs in sorted(by_task.items())
        },
    }
    (args.output_dir / "r1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1))
    print("AUDIT_DONE")


if __name__ == "__main__":
    main()
