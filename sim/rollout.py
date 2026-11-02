"""Closed-loop episode in the local simulator, driven the way the official worker drives a policy server over
tower-infer-v1: raw renders on the wire, k actions per infer, one per 30 Hz tick, 210 s.

Physics runs in an Isaac process (sim/physics_worker.py: demo parameters, evaluation controller); this process
renders its states with sim/render.py (FK of qpos through the USD joints, block poses) and talks to the policy.
The scene, start state and goals come from a demonstration HDF5 of the variant; sim/score.py scores the end state.
sim/fleet_driver.py runs many episodes of this loop in one process with batched rendering.

Run with the renderer's venv:
  CUDA_VISIBLE_DEVICES=3 runs/simdev/render_venv/bin/python sim/rollout.py <demo.hdf5> <out.json>
      [--url ws://127.0.0.1:9101] [--seconds 210] [--seed 0] [--offset DX DY DPSI] [--video-every N]
"""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch.cuda.nvtx as nvtx  # ranges for the nsys timeline (docs/throughput.md §8.2); torch is loaded by the renderer anyway
from websockets.sync.client import connect

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "external/tower-starter")]
from fk import ArmFK  # noqa: E402
from physics_client import Physics  # noqa: E402,F401  (re-exported: sim/fleet_driver.py, sim/expert_dagger.py)
from render import ROTATED, STATION, Renderer, episode_dims  # noqa: E402
from score import reach, score  # noqa: E402
from tower_protocol import pack, unpack  # noqa: E402

ACTION_HZ = 30
REFINE = str(ROOT / "runs/simdev/refine/v2/refine.pt")  # learned raster -> RTX residual (sim/refine.py)


def episode(demo: str, out: Path, physics: Physics, images, *, url: str, seconds: float = 210.0, seed: int = 0,
            offset=None, video_every: int = 0, session_id: str | None = None) -> dict:
    """One episode against the policy server at `url`; writes <out>.json, .npz (and _frames.npz) and returns the
    result. `images(q16, {block name: pose}, dims)` (dims: render.episode_dims) renders the four cameras in wire orientation. The physics worker is
    closed on return."""
    variant = Path(demo).parent.name
    try:
        with h5py.File(demo, "r") as f:
            a = f["sim"].attrs
            target_mapping, config = json.loads(a["target_mapping"]), json.loads(a["config_json"])
            instruction = f.attrs["instruction"]
            sim_attrs = dict(a)
        header, state0 = physics.connect()
        names, n_blocks = header["block_names"], len(header["block_names"])
        dims = episode_dims(sim_attrs, names)

        def observation(step: int, s: np.ndarray) -> dict:
            q, blocks = s[:16], s[16:].reshape(n_blocks, 7)
            return {"schema": "tower-challenge-observation-v1", "step_index": step, "sim_time_s": step / ACTION_HZ,
                    "task_variant": variant, "instruction": instruction,
                    "joint_positions": np.stack([q[:7], q[8:15]]).astype("<f4"),
                    "gripper_widths_m": q[[7, 15]].astype("<f4"), "images": images(q, dict(zip(names, blocks)), dims)}

        headers = {"X-Tower-Task-Variant": variant, "X-Tower-Track-Version": "local-sim",
                   "X-Tower-Evaluation-Version": "local-sim"}
        steps, frames, states = int(seconds * ACTION_HZ), [], [state0]
        timing, t_start = dict(physics=0.0, render=0.0, policy=0.0), time.perf_counter()
        with connect(url, additional_headers=headers, max_size=64 << 20, compression=None, proxy=None) as ws:
            meta = unpack(ws.recv())
            envelope = dict(protocol="tower-infer-v1", session_id=session_id or str(uuid.uuid4()),
                            trial_id=str(uuid.uuid4()), policy_id=meta["policy_id"])
            ws.send(pack({**envelope, "type": "reset", "request_id": str(uuid.uuid4()), "seed": seed}))
            ack = unpack(ws.recv())
            assert ack.get("type") == "reset_ack", ack
            step = 0
            while step < steps:
                t0 = time.perf_counter()
                obs = observation(step, states[-1])
                t1 = time.perf_counter()
                with nvtx.range("pack"):
                    msg = pack({**envelope, "type": "infer", "request_id": str(uuid.uuid4()), "observation": obs})
                with nvtx.range("ws_send"):
                    ws.send(msg)
                with nvtx.range("ws_recv"):
                    res = unpack(ws.recv())
                t2 = time.perf_counter()
                assert res.get("type") == "infer_result", res
                if video_every and len(states) // len(res["actions"]) % video_every == 0:
                    frames.append(obs["images"])
                actions = np.asarray(res["actions"], float)[: steps - step]
                with nvtx.range("physics"):
                    states += list(physics.step(actions))
                step += len(actions)
                timing["render"] += t1 - t0
                timing["policy"] += t2 - t1
                timing["physics"] += time.perf_counter() - t2
    finally:
        physics.close()

    states = np.array(states)
    setup = dict(initial_block_poses=np.array(header["initial_block_poses"]),
                 block_fixed=np.array(header["block_fixed"]))
    result = dict(demo=demo, variant=variant, seconds=seconds, seed=seed, offset=offset,
                  wall_s=time.perf_counter() - t_start, timing_s=timing, plugin=meta.get("plugin"),
                  **score(setup, states[-1, 16:].reshape(n_blocks, 7), target_mapping, config),
                  reach_m=reach(states[:, 16:].reshape(len(states), n_blocks, 7), target_mapping))
    np.savez_compressed(out.with_suffix(".npz"), qpos=states[:, :16], blocks=states[:, 16:].reshape(len(states), n_blocks, 7))
    if frames:
        np.savez_compressed(out.with_name(out.stem + "_frames.npz"),
                            **{c: np.stack([f[c] for f in frames]) for c in frames[0]})
    out.write_text(json.dumps(result, indent=1))  # last: its presence marks the episode done
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("out", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:9101")
    ap.add_argument("--seconds", type=float, default=210.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=float, nargs=3, metavar=("DX", "DY", "DPSI"),
                    help="move the movable assembly rigidly (m, m, rad about its centre), as jenga.offset measures it")
    ap.add_argument("--video-every", type=int, default=0, help="keep every Nth set of camera frames (0: none)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    physics = Physics(args.demo, args.offset, args.out.with_suffix(".physics.log"))  # boots while the renderer loads
    renderer, fk = Renderer(STATION, refine=REFINE), ArmFK(STATION)
    renderer.render()  # the first call compiles the shading: keep it out of the episode timing

    def images(q: np.ndarray, blocks: dict, dims: dict | None = None) -> dict:
        return {c: np.ascontiguousarray(im[::-1, ::-1]) if c in ROTATED else im
                for c, im in renderer.render(fk.link_poses(q), blocks, dims).items()}

    result = episode(args.demo, args.out, physics, images, url=args.url, seconds=args.seconds, seed=args.seed,
                     offset=args.offset, video_every=args.video_every)
    print("ROLLOUT_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
