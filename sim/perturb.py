"""DART data with matching images: demonstration frames re-rendered with the end effectors pushed off the demo.

Per frame and per arm, independently with probability P, the tool point (held-block centre) is moved by a uniform
direction x U(0, MAX_SHIFT) and the hand turned about a random axis through it by U(0, MAX_TURN); batched IK
(jenga.kinematics) solves the arm joints, frames whose IK misses by more than 1 mm / 1 mrad stay on the demo. A
block clamped by a displaced hand moves rigidly with its gripper_base. All four cameras are rendered at the new
state (sim/render.py with the learned refinement, every frame, so render style does not mark perturbed frames),
JPEG quality 95 as the serving adapter. Labels stay the demonstration's leader/qpos: from an off-demo state the
action chunk leads back to the demo.

Output: one HDF5 per demo, same variant directory and file name, holding what jenga.hdf5_dataset.TowerFrames
reads (images, observations/qpos, leader/qpos, sim/action_valid, attr instruction) plus perturb/offset [T, 2, 6]
(world shift m, world rotation vector rad; zero = on demo) and perturb/held [T, 2]. Resumable: finished files are
skipped, files appear by rename. Episodes run round-robin over the variants; shard k of n takes those
with crc32(path) % n == k, its workers pull them in order.

Run with the renderer's venv, one process per GPU:
  CUDA_VISIBLE_DEVICES=3 runs/simdev/render_venv/bin/python sim/perturb.py --shard 0/3 [--workers 3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import zlib
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from fk import ARMS, ArmFK  # noqa: E402
from render import ROBOTS, STATION, Renderer, pose_matrix  # noqa: E402
from writer import atomic, block_names, episode_dims, frame_poses, run_shard, write_images  # noqa: E402

from jenga.kinematics import load_arms, rotvec_to_matrix  # noqa: E402
from jenga.paths import env_path  # noqa: E402

from jenga.paths import DATA_ROOT as DATA  # noqa: E402  env JENGA_DATA_ROOT
OUT = env_path("JENGA_DART_OUT", DATA.parent / "dart_v1")  # the DART copies
REFINE = str(ROOT / "runs/simdev/refine/v1/refine.pt")  # pinned: all of dart_v1 is rendered with v1
SPLIT = ROOT / "runs/prep/split.json"
P, MAX_SHIFT, MAX_TURN = 0.5, 0.020, np.radians(3.0)
IK_TOL = 1e-3, 1e-3                  # m, rad
BLOCK_WIDTH, CLAMP_TOL = 0.05, 0.004  # fingers on a block read its 50 mm width (pushes run at 0-20 mm)
RIGID_TOL = 0.002                    # held block's offset from the gripper_base within 2 mm of its run median
SEED = 0


def gripper_bases(fk: ArmFK, qpos: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """qpos [T, 16] -> per-frame link poses and gripper_base world poses [T, 2, 4, 4]."""
    links = [fk.link_poses(q) for q in qpos]
    return links, np.array([[x[f"{ROBOTS[a]}/gripper_base"] for a in ARMS] for x in links])


def held_mask(qpos, blocks, active, grip) -> np.ndarray:
    """blocks [T, B, 4, 4], grip [T, 2, 4, 4] -> [T, 2]: the active block sits clamped in that arm's fingers (nearest gripper, width at the block width) and
    keeps its offset from the gripper_base within RIGID_TOL of the median over its contiguous clamped run."""
    T, held = len(qpos), np.zeros((len(qpos), 2), bool)
    idx = np.flatnonzero(active >= 0)
    if not len(idx):
        return held
    b = blocks[idx, active[idx], :3, 3]
    rel = np.einsum("taji,taj->tai", grip[idx, :, :3, :3], b[:, None] - grip[idx, :, :3, 3])  # block in gripper frame
    arm = np.linalg.norm(rel, axis=-1).argmin(1)
    width = qpos[idx][np.arange(len(idx)), 8 * arm + 7]
    cand = np.zeros((T, 2), bool)
    cand[idx, arm] = np.abs(width - BLOCK_WIDTH) < CLAMP_TOL
    r = np.zeros((T, 2, 3))
    r[idx] = rel
    for a in range(2):
        edges = np.flatnonzero(np.diff(np.r_[0, cand[:, a].astype(int) * (active + 2), 0]))  # runs per block
        for s, e in zip(edges[:-1], edges[1:]):
            if cand[s, a]:
                held[s:e, a] = np.linalg.norm(r[s:e, a] - np.median(r[s:e, a], 0), axis=-1) < RIGID_TOL
    return held


def perturb(qpos: np.ndarray, rng: np.random.Generator, arms: dict) -> tuple[np.ndarray, np.ndarray]:
    """Displaced qpos [T, 16] and offsets [T, 2, 6] (zero where not perturbed or IK missed)."""
    T, out, offset = len(qpos), qpos.astype(np.float64), np.zeros((len(qpos), 2, 6))
    for i, name in enumerate(ARMS):
        arm, q = arms[name], qpos[:, 8 * i:8 * i + 7].astype(np.float64)
        on = np.flatnonzero(rng.random(T) < P)
        d = rng.normal(size=(T, 2, 3))
        if not len(on):
            continue
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        dp, dr = d[on, 0] * rng.uniform(0, MAX_SHIFT, (T, 1))[on], d[on, 1] * rng.uniform(0, MAX_TURN, (T, 1))[on]
        R, p = arm.fk_world(q[on])
        Rn = rotvec_to_matrix(dr) @ R
        pn = arm.tool_world(R, p) + dp - Rn @ arm.tool  # turn about the tool point, then shift it
        qn, ep, er = arm.ik_world(Rn, pn, q[on], q[on])
        ok = (ep < IK_TOL[0]) & (er < IK_TOL[1])
        out[on[ok], 8 * i:8 * i + 7] = qn[ok]
        offset[on[ok], i] = np.c_[dp, dr][ok]
    return out.astype(np.float32), offset


class Worker:
    def __init__(self):
        self.renderer, self.fk, self.arms = Renderer(STATION, refine=REFINE), ArmFK(STATION), load_arms()
        self.renderer.render()  # compile the shading outside the timings

    def episode(self, rel: str) -> dict:
        src, dst = DATA / rel, OUT / rel
        t0 = time.perf_counter()
        with h5py.File(src, "r") as f:
            qpos, blocks, active = f["observations/qpos"][:], f["sim/block_poses"][:], f["sim/active_block"][:]
            leader, valid, instruction = f["leader/qpos"][:], f["sim/action_valid"][:], f.attrs["instruction"]
            names = block_names(tuple(json.loads(f["sim"].attrs["block_fixed_on_table"])))
            dims = episode_dims(f["sim"].attrs, names)
            jpeg_dtype = f["observations/images/left_back_camera"].dtype
        T = len(qpos)
        rng = np.random.default_rng([SEED, zlib.crc32(rel.encode())])
        _, grip = gripper_bases(self.fk, qpos)
        bm = np.array([[pose_matrix(b) for b in fr] for fr in blocks])
        held = held_mask(qpos, bm, active, grip)
        new_q, offset = perturb(qpos, rng, self.arms)
        new_links, new_grip = gripper_bases(self.fk, new_q)
        t_cpu = time.perf_counter() - t0

        with atomic(dst) as tmp, h5py.File(tmp, "w") as g:
            g.attrs["instruction"] = instruction
            g.attrs["source"] = rel
            g.attrs["perturb"] = json.dumps(dict(p=P, max_shift_m=MAX_SHIFT, max_turn_rad=float(MAX_TURN), seed=SEED,
                                                 ik_tol=IK_TOL, refine=str(Path(REFINE).relative_to(ROOT))))
            g["observations/qpos"] = new_q
            g["leader/qpos"] = leader
            g["sim/action_valid"] = valid
            g["perturb/offset"] = offset.astype(np.float32)
            g["perturb/held"] = held

            def poses(t):  # a block clamped by a displaced hand moves with its gripper_base
                moved = {int(active[t]): new_grip[t, a] @ np.linalg.inv(grip[t, a]) @ bm[t, active[t]]
                         for a in range(2) if held[t, a] and offset[t, a].any()}
                return frame_poses(self.renderer, new_links[t], blocks[t], names, moved, dims)

            t_render = write_images(g, self.renderer, T, poses, jpeg_dtype)
        wall = time.perf_counter() - t0
        return dict(episode=rel, frames=T, wall_s=round(wall, 1), cpu_prep_s=round(t_cpu, 1), render_s=round(t_render, 1),
                    fps=round(T / wall, 1), perturbed=np.round((offset != 0).any(-1).mean(0), 3).tolist(),
                    held=held.sum(0).tolist(), held_moved=int((held & (offset != 0).any(-1)).sum()))


def setup():
    return Worker().episode


def episodes() -> list[str]:
    """Train episodes round-robin over the variants, so every prefix of the output covers all of them."""
    tasks = json.loads(SPLIT.read_text())["tasks"]
    queues = [sorted(v["train"]) for _, v in sorted(tasks.items())]
    return [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="k/n")
    ap.add_argument("--workers", type=int, default=3, help="renderer processes on this GPU")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many episodes (0: all)")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    mine = [e for e in episodes() if zlib.crc32(e.encode()) % n == k]  # not [k::n]: 10 variants, n even
    for e in mine:  # leftovers of a killed run of this shard
        for tmp in (OUT / e).parent.glob(f"{Path(e).name}.tmp.*"):
            tmp.unlink()
    todo = [e for e in mine if not (OUT / e).exists()][: args.limit or None]
    print(f"shard {k}/{n} on {socket.gethostname()} pid {os.getpid()}: {len(mine)} episodes, {len(todo)} to do", flush=True)
    t0 = time.perf_counter()
    frames = run_shard(todo, args.workers, setup, key="episode")
    print(f"PERTURB_DONE shard {k}/{n}: {len(todo)} episodes, {frames} frames, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
