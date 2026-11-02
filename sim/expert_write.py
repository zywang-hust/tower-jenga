"""Training episodes from expert rollouts (sim/expert_batch.py): every tick of each solved rollout rendered with the
four cameras (sim/render.py with the learned refinement), written in the demonstration layout so scripts/train.py
--synthetic-root mixes them in. Labels are the expert's commands (leader/qpos row t = the command sent at tick t,
30 Hz), the observation is the executed state: a self-consistent demonstration of the perturbed scene.

Output <out_root>/data/<variant>/<demo stem>_s<seed>.hdf5 (seed: the batch directory's s<seed>): observations/qpos
[T, 16], observations/images/<camera> [T] JPEG q95, leader/qpos [T, 16], sim/action_valid [T] (True),
sim/active_block [T], sim/block_released [T, B] (False), sim/block_poses [T, B, 7], sim/image_valid [T] (--frame-stride k: only every k-th frame has images); attrs instruction, source
(dataset-relative demo), expert (the rollout's result JSON). Resumable: finished files are skipped, files appear by
rename. Shard k of n takes rollouts with crc32(name) % n == k.

Run with the renderer's venv, one process per GPU:
  CUDA_VISIBLE_DEVICES=0 runs/simdev/render_venv/bin/python sim/expert_write.py runs/simdev/expert/offset_train/s0 \
      <synthetic out root, e.g. $JENGA_DATA_ROOT/../expert_offset_v1> --refine runs/simdev/refine/v2/refine.pt [--shard 0/4] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from writer import atomic, block_names, episode_dims, frame_poses, run_shard, write_images  # noqa: E402

from jenga.paths import DATA_ROOT as DATA  # noqa: E402  env JENGA_DATA_ROOT


def solved(batch: Path) -> list[str]:
    """Names of the batch's rollouts that completed every operation and logged the held block."""
    names = []
    for j in sorted(batch.glob("task*.json")):
        r = json.loads(j.read_text())
        if all(r["ops"]) and j.with_suffix(".npz").exists() and "active" in np.load(j.with_suffix(".npz")):
            names.append(j.stem)
    return names


def write_one(renderer, fk, batch: Path, out_root: Path, name: str, refine: str, stride: int = 1) -> dict:
    import h5py

    result = json.loads((batch / f"{name}.json").read_text())
    demo = Path(result["demo"])
    rel = demo.relative_to(DATA)
    rep = name.rsplit("_", 1)[1] if name.rsplit("_", 1)[1].startswith("r") else ""  # expert_batch.py --repeats
    dst = out_root / rel.parent / f"{demo.stem}_{batch.name}{'_' + rep if rep else ''}.hdf5"
    if dst.exists():
        return dict(name=name, frames=0, skipped=True)
    t0 = time.perf_counter()
    log = np.load(batch / f"{name}.npz")
    T = len(log["actions"])
    qpos, blocks, active = log["qpos"][:T].astype(np.float32), log["blocks"][:T], log["active"][:T]
    with h5py.File(demo, "r") as f:
        instruction = f.attrs["instruction"]
        names = block_names(tuple(json.loads(f["sim"].attrs["block_fixed_on_table"])))
        dims = episode_dims(f["sim"].attrs, names)
        jpeg_dtype = f["observations/images/left_back_camera"].dtype
    with atomic(dst) as tmp, h5py.File(tmp, "w") as g:
        g.attrs["instruction"] = instruction
        g.attrs["source"] = str(rel)
        g.attrs["expert"] = json.dumps(dict(result, refine=refine))
        g["observations/qpos"] = qpos
        g["leader/qpos"] = log["actions"].astype(np.float32)
        g["sim/action_valid"] = np.ones(T, bool)
        g["sim/active_block"] = active
        g["sim/block_released"] = np.zeros((T, blocks.shape[1]), bool)
        g["sim/block_poses"] = blocks.astype(np.float32)
        t_render = write_images(g, renderer, T, lambda t: frame_poses(
            renderer, fk.link_poses(qpos[t].astype(np.float64)), blocks[t], names, {}, dims), jpeg_dtype, stride)
    wall = time.perf_counter() - t0
    return dict(name=name, frames=T, wall_s=round(wall, 1), render_s=round(t_render, 1), fps=round(T / wall, 1))


def setup(batch: str, out_root: str, refine: str, stride: int = 1):
    from fk import ArmFK
    from render import STATION, Renderer

    renderer, fk = Renderer(STATION, refine=refine), ArmFK(STATION)
    renderer.render()  # compile the shading outside the timings
    return lambda name: write_one(renderer, fk, Path(batch), Path(out_root), name, refine, stride)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", type=Path, help="sim/expert_batch.py output directory, named s<seed>")
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--refine", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1, help="render every k-th frame (sim/image_valid marks them)")
    args = ap.parse_args()
    batch, out_root, refine = args.batch.resolve(), args.out_root.resolve(), str(Path(args.refine).resolve())
    assert Path(refine).exists(), refine
    k, n = map(int, args.shard.split("/"))
    names = solved(batch)
    mine = [x for x in names if zlib.crc32(x.encode()) % n == k][: args.limit or None]
    print(f"write shard {k}/{n}: {len(mine)} of {len(names)} solved rollouts, refine {refine}", flush=True)
    t0 = time.perf_counter()
    frames = run_shard(mine, args.workers, setup, (str(batch), str(out_root), refine, args.frame_stride))
    print(f"WRITE_DONE shard {k}/{n}: {len(mine)} rollouts, {frames} frames, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
