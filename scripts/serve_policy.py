"""Run the tower-infer-v1 policy server on 127.0.0.1 (exposing it publicly needs the user's authorization).

Example: python scripts/serve_policy.py --track configs/track_local.json --params runs/r2/params_40000 --devices 0,1
Prints SERVER_READY once every replica has compiled. Each replica is a spawned process holding one device
(jenga.server.ModelPool): the forward's host side never shares this process's GIL with the event loop.

Everything else is fixed by the snapshot or by measurement: prompt and norm stats come from the snapshot's run
(jenga.model.snapshot_inputs); post-processing (8 rows per request, SG9 smoothing, 2-step fade, docs/research.md §3)
and the q95 JPEG round trip are the server defaults; --chunk overrides the post-processing for closed-loop
comparisons in the local simulator. The test-time frame transform (jenga.warp, research.md §2.4) is not applied: on
layouts offset by 10-30 mm it lowered the local closed-loop score (§2.4). Every reset and infer is recorded to
runs/serve_records/ for failure analysis (jenga.record; diagnosis only, never training data).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # this checkout's jenga, ahead of the editable install (worktrees)
K, FADE, SMOOTH = 8, 2, 9


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--track", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True, help="snapshot dir containing params/")
    p.add_argument("--devices", default="0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--policy-id", default="jenga-r1")
    p.add_argument("--records", default=str(ROOT / "runs/serve_records"),
                   help="diagnostic record root; '' disables the recorder (local simulator fleets)")
    p.add_argument("--chunk", type=int, nargs=3, default=(K, FADE, SMOOTH), metavar=("K", "FADE", "SMOOTH"),
                   help="rows per request, fade steps, Savitzky-Golay window (0: off)")
    p.add_argument("--start-qpos", type=Path, help="configs/start_qpos.json: trials that start away from the training "
                   "start state are first moved there (jenga.server.homing_plan)")
    p.add_argument("--buckets", default="1,2,4", help="batch sizes compiled at startup (local simulator fleets: "
                   "up to 16); a batch is padded to the next one")
    return p.parse_args()


async def main():
    args = parse()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("jenga.server").setLevel(logging.INFO)  # the replicas' batch statistics
    import numpy as np
    import openpi.shared.normalize as normalize
    from websockets.asyncio.server import serve

    from jenga.model import snapshot_inputs
    from jenga.policy import ObservationAdapter
    from jenga.record import Recorder
    from jenga.server import ModelPool, artifact_identity, handler_factory

    prompt, prep = snapshot_inputs(args.params)
    track = json.loads(args.track.read_text())
    adapter = ObservationAdapter(normalize.load(prep / "norm_stats"), prompt=prompt)
    k, fade, smooth = args.chunk
    config = {"k": k, "fade": fade, "smooth": smooth, "prompt": prompt, "jpeg_quality": adapter.jpeg_quality}
    config["orient"] = "orient-static-v2:" + hashlib.sha256((ROOT / "configs/orient_reference.npz").read_bytes()).hexdigest()[:16]
    start_qpos = json.loads(args.start_qpos.read_text())["home"] if args.start_qpos else None
    if start_qpos is not None:
        config["start_qpos"] = start_qpos
    identity = artifact_identity(args.params, prep / "norm_stats", config)
    warm = adapter({
        "task_variant": "task1-2_tower_transfer_2", "instruction": "warmup",
        "joint_positions": np.zeros((2, 7), np.float32), "gripper_widths_m": np.full(2, 0.1, np.float32),
        "images": {cam: np.zeros((*hw, 3), np.uint8) for cam, hw in track["cameras"].items()},
    })
    pool = ModelPool(args.params, [int(d) for d in args.devices.split(",")], warm,
                     buckets=[int(b) for b in args.buckets.split(",")])
    handler = handler_factory(track, args.policy_id, identity, adapter, pool, k=k, fade=fade, smooth=smooth,
                              recorder=Recorder(args.records), orient_per_trial=True,
                              start_qpos=start_qpos)
    async with serve(handler, args.host, args.port, max_size=64 * 1024 * 1024, compression=None,
                     ping_timeout=None):
        print("SERVER_READY", json.dumps({"port": args.port, "artifact_identity": identity, **config}), flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
