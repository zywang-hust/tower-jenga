"""Protocol regression for jenga.server over a real websocket, with a fake model (CPU, no weights)."""
import asyncio
import json
import types
import uuid
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "external/tower-starter/tower_protocol.py").exists():
    pytest.skip("needs external/tower-starter", allow_module_level=True)

from openpi.shared.normalize import NormStats  # noqa: E402

from jenga.policy import ObservationAdapter  # noqa: E402
from jenga.server import handler_factory  # noqa: E402  (puts the starter codec on sys.path)
from tower_protocol import pack, sample_seed, unpack  # noqa: E402,I001

TRACK = json.loads((ROOT / "configs/track_local.json").read_text())


def stats():
    s = NormStats(mean=np.zeros(16), std=np.ones(16), q01=-np.ones(16, np.float32), q99=np.ones(16, np.float32))
    return {"state": s, "actions": s}


class FakeModel:
    """Returns a constant model-space chunk whose first joint encodes the call count."""

    def expect(self):  # ModelPool.expect: a token the handler closes once the request is queued or failed
        return types.SimpleNamespace(close=lambda: None)

    def __init__(self):
        self.calls = 0

    async def __call__(self, model_input, noise, pending=None, trial=None):
        self.calls += 1
        out = np.zeros((32, 32), np.float32)
        out[:, 0] = 0.01 * self.calls
        return out


def observation(step):
    return {"step_index": step, "task_variant": "task4-2_full_tower_pushdown_2", "instruction": "x",
            "joint_positions": np.zeros((2, 7), "<f4"), "gripper_widths_m": np.full(2, 0.1, "<f4"),
            "images": {c: np.zeros((*hw, 3), np.uint8) for c, hw in TRACK["cameras"].items()}}


def test_protocol_flow_reconnect_rejections_and_record(tmp_path):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    from jenga.record import Recorder, read_records

    model, recorder = FakeModel(), Recorder(tmp_path)
    adapter = ObservationAdapter(stats(), prompt="train", jpeg_quality=None)
    handler = handler_factory(TRACK, "p", "sha256:" + "0" * 64, adapter, model, k=8, fade=4, smooth=0,
                              recorder=recorder)

    async def scenario():
        async with serve(handler, "127.0.0.1", 0, compression=None, max_size=64 << 20) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            env = dict(protocol="tower-infer-v1", session_id=str(uuid.uuid4()), trial_id="t1", policy_id="p")
            async with connect(url, max_size=64 << 20, proxy=None) as ws:
                meta = unpack(await ws.recv())
                assert meta["seed_mode"] == "sample_seed" and meta["config_metadata"]["k"] == 8
                await ws.send(pack({**env, "type": "reset", "request_id": "r0", "seed": 7}))
                assert unpack(await ws.recv())["type"] == "reset_ack"
                first = pack({**env, "type": "infer", "request_id": "r1", "observation": observation(0)})
                await ws.send(first)
                a = unpack(await ws.recv())
                assert a["actions"].shape == (8, 16) and a["sample_seed"] == sample_seed(7, 0)
                await ws.send(pack({**env, "type": "infer", "request_id": "r2", "observation": observation(8)}))
                b = unpack(await ws.recv())
                assert b["sample_index"] == 1
                # Cross-fade: the second chunk starts between the two model outputs and reaches the new one.
                assert a["actions"][0, 0] < b["actions"][0, 0] < b["actions"][4, 0]
            async with connect(url, max_size=64 << 20, proxy=None) as ws:  # reconnect: identical bytes replay the cached response
                unpack(await ws.recv())
                await ws.send(pack({**env, "type": "infer", "request_id": "r2", "observation": observation(8)}))
                assert unpack(await ws.recv())["sample_index"] == 1 and model.calls == 2
                await ws.send(first)  # an older request id is stale
                assert unpack(await ws.recv())["error"]["code"] == "stale_request"
            async with connect(url, max_size=64 << 20, proxy=None) as ws:  # infer on an unknown session never creates one
                unpack(await ws.recv())
                await ws.send(pack({**env, "session_id": "new", "type": "infer", "request_id": "x",
                                    "observation": observation(0)}))
                assert unpack(await ws.recv())["error"]["code"] == "session_expired"

    asyncio.run(scenario())
    recorder.flush()  # one record per reset and per new infer; the cached replay of r2 is not recorded again
    (path,) = tmp_path.rglob("t1.rec")
    reset, r1, r2 = read_records(path)
    assert reset["type"] == "reset" and reset["seed"] == 7
    assert [r1["step_index"], r2["step_index"]] == [0, 8] and r2["actions"].shape == (8, 16)
    assert set(r1["images"]) == set(TRACK["cameras"]) and isinstance(r1["images"]["left_back_camera"], bytes)


def test_rejects_bad_requests_before_any_model_work():
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    model = FakeModel()
    adapter = ObservationAdapter(stats(), prompt="train", jpeg_quality=None)
    handler = handler_factory(TRACK, "p", "sha256:" + "0" * 64, adapter, model, k=8, fade=4, smooth=0)

    async def one(obs):
        async with serve(handler, "127.0.0.1", 0, compression=None, max_size=64 << 20) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            env = dict(protocol="tower-infer-v1", session_id=str(uuid.uuid4()), trial_id="t1", policy_id="p")
            async with connect(url, max_size=64 << 20, proxy=None) as ws:
                unpack(await ws.recv())
                await ws.send(pack({**env, "type": "reset", "request_id": "r0", "seed": 1}))
                unpack(await ws.recv())
                await ws.send(pack({**env, "type": "infer", "request_id": "r1", "observation": obs}))
                return unpack(await ws.recv())["error"]["code"]

    assert asyncio.run(one(observation(0) | {"instruction": "  "})) == "invalid_instruction"
    assert asyncio.run(one({k: v for k, v in observation(0).items() if k != "step_index"})) == "invalid_step_index"
    assert asyncio.run(one(observation(0) | {"images": None})) == "invalid_observation"
    assert asyncio.run(one(observation(0) | {"task_variant": "task9-9_unknown"})) == "invalid_task_variant"
    assert model.calls == 0
    with pytest.raises(ValueError, match="30 Hz"):  # a 10 Hz track would play every 30 Hz row for 0.1 s
        handler_factory(TRACK | {"action_dt_s": 0.1}, "p", "sha256:" + "0" * 64, adapter, FakeModel())
