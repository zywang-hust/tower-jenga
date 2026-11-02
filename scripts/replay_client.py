"""Fake TOWER worker: replay held-out dataset episodes through a running policy server over tower-infer-v1.

Question: does the server honour the protocol (metadata, identity echo, seeds, dt, shapes, bounds, idempotent
reconnect) under 8+1 concurrent sessions, and do its executed actions match the demonstrations as closely as the
offline evaluation says? Observations are dataset frames (open loop: the demo, not our actions, drives the
state); step_index advances by the k actions returned. Writes runs/replay/<tag>.json; prints REPLAY_DONE.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/tower-starter"))
from tower_protocol import pack, sample_seed, unpack  # noqa: E402

from jenga.paths import DATA_ROOT  # noqa: E402
from jenga.policy import to_dataset_orientation  # noqa: E402
from jenga.transforms import JOINTS  # noqa: E402

CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--url", default="ws://127.0.0.1:9001")
    p.add_argument("--track", type=Path, default=ROOT / "configs/track_local.json")
    p.add_argument("--dataset-root", type=Path, default=DATA_ROOT, help="TOWER dataset root (env JENGA_DATA_ROOT)")
    p.add_argument("--split", type=Path, default=ROOT / "runs/prep/split.json")
    p.add_argument("--per-variant", type=int, default=1, help="val episodes per variant")
    p.add_argument("--chunks", type=int, default=40, help="requests per episode from the start (stops at the episode end)")
    p.add_argument("--sessions", type=int, default=9, help="concurrent sessions")
    p.add_argument("--policy-id", default="jenga-r1")
    return p.parse_args()


class Check:
    def __init__(self):
        self.failures: list[str] = []

    def __call__(self, ok: bool, what: str):
        if not ok and len(self.failures) < 50:
            self.failures.append(what)


def frame_observation(f, t: int, variant: str, instruction: str) -> dict:
    q = f["observations/qpos"][t].astype("<f4")
    # Send the images as the official worker does: the 180-degree turn is its own inverse.
    return to_dataset_orientation({
        "schema": "tower-challenge-observation-v1", "step_index": t, "sim_time_s": t / 30.0,
        "task_variant": variant, "instruction": instruction,
        "joint_positions": np.stack([q[:7], q[8:15]]), "gripper_widths_m": q[[7, 15]].copy(),
        "images": {c: np.asarray(Image.open(io.BytesIO(f[f"observations/images/{c}"][t].tobytes())).convert("RGB"))
                   for c in CAMERAS},
    })


async def episode(args, track, path: Path, check: Check, official: dict) -> dict:
    from websockets.asyncio.client import connect

    variant = path.parent.name
    headers = {"X-Tower-Task-Variant": variant, "X-Tower-Track-Version": "local",
               "X-Tower-Evaluation-Version": "local"}
    session, trial, base_seed = str(uuid.uuid4()), str(uuid.uuid4()), 1234
    envelope = dict(protocol="tower-infer-v1", session_id=session, trial_id=trial, policy_id=args.policy_id)
    lat, executed, demo, timing = [], [], [], []
    with h5py.File(path, "r") as f:
        n_valid = int(f["sim/action_valid"][:].sum())
        k = None
        async with connect(args.url, additional_headers=headers, max_size=64 << 20, compression=None, proxy=None) as ws:
            meta = unpack(await asyncio.wait_for(ws.recv(), 10))
            check(meta.get("type") == "metadata" and meta.get("ready") is True, "metadata")
            check(str(meta.get("artifact_identity", "")).startswith("sha256:")
                  and len(meta["artifact_identity"]) == 71, "artifact_identity format")
            k = meta["config_metadata"]["k"]
            await ws.send(pack({**envelope, "type": "reset", "request_id": str(uuid.uuid4()), "seed": base_seed}))
            ack = unpack(await ws.recv())
            check(ack.get("type") == "reset_ack" and ack.get("next_sample_index") == 0, "reset_ack")
            # From the episode start, as a worker does after reset: a server may decide per trial from the first
            # observation (jenga.orient does), and a mid-episode frame has the blocks already moved.
            t = 0
            for i in range(args.chunks):
                if t + k > n_valid:
                    break
                request_id = str(uuid.uuid4())
                req = pack({**envelope, "type": "infer", "request_id": request_id,
                            "observation": frame_observation(f, t, variant, official[variant[:5]])})
                t0 = time.perf_counter()
                await ws.send(req)
                res = unpack(await ws.recv())
                lat.append((time.perf_counter() - t0) * 1e3)
                if res.get("type") != "infer_result":
                    check(False, f"{variant} error {res.get('error')}")
                    break
                a = res["actions"]
                check(all(res.get(x) == v for x, v in {**envelope, "request_id": request_id}.items()), "identity echo")
                check(res["sample_index"] == i and res["sample_seed"] == sample_seed(base_seed, i), "seed fields")
                check(res["plugin"] == meta["plugin"] and res["dt_s"] == track["action_dt_s"], "plugin/dt_s")
                check(a.dtype == np.dtype("<f4") and a.shape == (k, 16) and np.isfinite(a).all(), "action shape")
                check(bool((a >= np.asarray(track["action_lower"], "<f4")).all()
                           and (a <= np.asarray(track["action_upper"], "<f4")).all()), "action bounds")
                if i == min(args.chunks, n_valid // k) // 2:  # transport drop: resend identical bytes on a new socket
                    async with connect(args.url, additional_headers=headers, max_size=64 << 20, proxy=None,
                                       compression=None) as ws2:
                        unpack(await ws2.recv())
                        await ws2.send(req)
                        replay = unpack(await ws2.recv())
                        check(replay.get("sample_index") == i
                              and np.array_equal(replay.get("actions"), a), "idempotent replay")
                timing.append(res.get("server_timing") or {})
                executed.append(a)
                demo.append(f["leader/qpos"][t : t + k])
                t += k
    if not executed:
        # first request failed; recorded in protocol_failures
        return {"variant": variant[:7], "requests": 0, "latency_ms": lat, "server_timing": timing}
    e, d = np.concatenate(executed), np.concatenate(demo)
    step = np.abs(np.diff(e[:, JOINTS], axis=0)).max(-1)
    dstep = np.abs(np.diff(d[:, JOINTS], axis=0)).max(-1)
    bounds = np.arange(k, len(e), k) - 1
    return {"variant": variant[:7], "requests": len(executed), "mae_rad": float(np.abs(e - d)[:, JOINTS].mean()),
            "grip_mm": float(np.abs(e - d)[:, [7, 15]].mean() * 1e3),
            "jump_p95_rad": float(np.percentile(step[bounds], 95)),
            "demo_jump_p95_rad": float(np.percentile(dstep[bounds], 95)), "latency_ms": lat, "server_timing": timing}


async def main():
    args = parse()
    from jenga.instructions import OFFICIAL_INSTRUCTIONS

    track = json.loads(args.track.read_text())
    split = json.loads(args.split.read_text())
    paths = [args.dataset_root / ep for v in sorted(split["tasks"]) for ep in sorted(split["tasks"][v]["val"])
             [: args.per_variant]]
    check, sem = Check(), asyncio.Semaphore(args.sessions)

    async def run(p):
        async with sem:
            return await episode(args, track, p, check, OFFICIAL_INSTRUCTIONS)

    t0 = time.perf_counter()
    rows = await asyncio.gather(*(run(p) for p in paths))
    lat = np.concatenate([r.pop("latency_ms") for r in rows])
    timing = [t for r in rows for t in r.pop("server_timing")]
    # adapter_ms and model_ms per request
    server_timing = {key: {q: float(np.percentile([t[key] for t in timing if key in t], q)) for q in (50, 90, 99, 100)}
                     for key in sorted({key for t in timing for key in t})}
    result = {
        "question": "protocol conformance and action fidelity of the policy server under concurrent sessions",
        "args": {k: str(v) for k, v in vars(args).items()}, "wall_s": time.perf_counter() - t0,
        "protocol_failures": check.failures, "latency_ms": {q: float(np.percentile(lat, q)) for q in (50, 90, 99)},
        "server_timing_ms": server_timing,
        "macro": {m: float(np.mean([r[m] for r in rows if m in r])) for m in ("mae_rad", "grip_mm", "jump_p95_rad",
                                                                  "demo_jump_p95_rad")},
        "episodes": rows,
    }
    out = ROOT / "runs/replay" / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("protocol_failures", "latency_ms", "macro", "wall_s")}))
    print("REPLAY_DONE", args.tag)


if __name__ == "__main__":
    asyncio.run(main())
