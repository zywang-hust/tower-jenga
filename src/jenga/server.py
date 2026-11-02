"""tower-infer-v1 policy server: per-session state, reconnect-safe, dynamic batching over GPU replicas.

Session handling follows the official reference server (external/tower-starter/server.py): identity echo,
idempotent replay of the last request, stale-request rejection, no implicit session creation on infer.
Per request: ObservationAdapter (same transforms as training) -> batched flow sampling with noise drawn from
the request's sample_seed -> absolute decode -> joint smoothing and cross-fade against the previous chunk (ChunkFader) -> first `k` rows,
clipped to the release bounds.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import queue
import sys
import threading
import time
import traceback
from math import ceil
from pathlib import Path

import numpy as np

from jenga import nvtx
from jenga.policy import ObservationAdapter, clip_to_bounds, state16, to_dataset_orientation
from jenga.stitching import ChunkFader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external/tower-starter"))
from tower_protocol import action_spec, observation_spec, pack, sample_seed, unpack  # noqa: E402

log = logging.getLogger("jenga.server")

# Adapter code: the observation transforms and post-processing. plugin_version carries their hash so it changes
# whenever the adapter does (contract: "Update plugin_version when your adapter changes").
ADAPTER_SOURCES = ("server.py", "policy.py", "transforms.py", "stitching.py")
PLUGIN = "jenga-pi05"
PLUGIN_VERSION = "1+" + hashlib.sha256(b"".join(
    (Path(__file__).parent / f).read_bytes() for f in ADAPTER_SOURCES)).hexdigest()[:12]
CONTROL_DT_S = 1 / 30  # demonstrations and action chunks are 30 Hz
JOINT_COLUMNS = [*range(7), *range(8, 15)]  # state16 layout: left 7 joints, left width, back 7 joints, back width
START_TOL_RAD = 0.02
HOMING_RAD_PER_S = 0.25
HOMING_HOLD_S = 0.5


def homing_plan(q: np.ndarray, target) -> np.ndarray | None:
    """Commands [T, 16] from state q to the training start state `target`: a cosine ramp at most HOMING_RAD_PER_S
    (at least 1 s), then HOMING_HOLD_S of hold so the arms settle before the policy sees them. None when every joint
    is already within START_TOL_RAD."""
    target = np.asarray(target, dtype=np.float32)
    dist = float(np.abs(target - q)[JOINT_COLUMNS].max())
    if dist <= START_TOL_RAD:
        return None
    n = max(30, ceil(dist / HOMING_RAD_PER_S / CONTROL_DT_S))
    s = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2
    ramp = q[None] + s[:, None] * (target - q)[None]
    return np.concatenate([ramp, np.repeat(target[None], round(HOMING_HOLD_S / CONTROL_DT_S), axis=0)]).astype(np.float32)
IDENTITY_FIELDS = ("protocol", "session_id", "trial_id", "request_id", "policy_id")


def artifact_identity(params_dir: Path, norm_stats_dir: Path, config: dict) -> str:
    """sha256 over every weight and norm-stats file (path + bytes) and the serving config. Only params/ of the
    snapshot is hashed: it is all load_policy reads, so stray files next to it cannot move the identity."""
    h = hashlib.sha256()
    for root in (Path(params_dir) / "params", norm_stats_dir):
        for f in sorted(p for p in Path(root).rglob("*") if p.is_file()):
            h.update(str(f.relative_to(root)).encode() + b"\0")
            with f.open("rb") as fh:
                while block := fh.read(1 << 24):
                    h.update(block)
    h.update(json.dumps(config, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()


def _resolve(fut: asyncio.Future, result, exc) -> None:
    if fut.done():  # the request was cancelled (client gone) while the batch ran
        return
    if exc is None:
        fut.set_result(result)
    else:
        fut.set_exception(exc)


STAGING_CAP_S = 0.25  # longest a replica waits for requests the server has accepted but not yet staged


def _replica_main(conn, params_dir: str, index: int, warm_input: dict, buckets: tuple, num_steps: int) -> None:
    """A model replica process: fast-path switches, model load, bucket compilation, then one batch at a time over
    the pipe. Images cross the pipe and the PCIe bus as uint8 and become [-1, 1] float32 on the device (the same
    expression Observation.from_dict applies on the host, inside one fused kernel). Messages in: (seq, items); out:
    ("ok", seq, actions), ("error", seq, traceback) or ("log", None, text)."""
    try:
        from jenga.perf import enable_fast_path

        enable_fast_path(training=False)
        import jax
        import openpi.models.model as _model
        from openpi.shared import nnx_utils

        from jenga.dataset import collate
        from jenga.model import load_policy

        device = jax.devices()[index]
        with jax.default_device(device):
            policy = load_policy(Path(params_dir), device)
        sample = nnx_utils.module_jit(policy.sample_actions, static_argnames=("num_steps",))
        to_obs = jax.jit(_model.Observation.from_dict)
        phases = []  # (n, collate ms, upload ms, forward ms), logged every 50 batches

        def forward(items):
            n = len(items)
            pad = next(b for b in buckets if b >= n)
            items = items + [items[-1]] * (pad - n)
            t0 = time.perf_counter()
            with nvtx.range(f"collate {n}"):
                data = collate([inp for inp, _ in items])
                noise = np.stack([nz for _, nz in items]).astype(np.float32)
            t1 = time.perf_counter()
            with nvtx.range("upload"):
                data, noise = jax.device_put((data, noise), device)
            t2 = time.perf_counter()
            with nvtx.range(f"forward {n}"):
                out = sample(jax.random.key(0), to_obs(data), num_steps=num_steps, noise=noise)
            with nvtx.range("readback"):
                acts = np.asarray(out)[:n]
            phases.append((n, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (time.perf_counter() - t2) * 1e3))
            if len(phases) == 50:
                m = np.array(phases)
                conn.send(("log", None, "replica %d: %d batches, mean size %.1f (min %d), per batch collate %.1f ms, "
                           "upload %.1f ms, forward %.1f ms" % (index, len(m), m[:, 0].mean(), m[:, 0].min(), *m[:, 1:].mean(0))))
                phases.clear()
            return acts

        for b in buckets:  # compile every bucket before accepting traffic
            forward([(warm_input, np.zeros((32, 32), np.float32))] * b)
        phases.clear()
    except BaseException:  # noqa: BLE001
        conn.send(("error", None, traceback.format_exc()))
        return
    conn.send(("ready", None, None))
    while True:
        try:
            with nvtx.range("pipe_recv"):
                seq, items = conn.recv()
        except EOFError:
            return
        try:
            acts = forward(items)
            with nvtx.range("pipe_send"):
                conn.send(("ok", seq, acts))
        except BaseException:  # noqa: BLE001
            conn.send(("error", seq, traceback.format_exc()))


class _Pending:
    """A request the server accepted and is staging (frame estimate, adapter); closed once it is queued or failed."""

    def __init__(self, pool):
        self.pool, self.open = pool, True

    def close(self) -> None:
        if self.open:
            self.open = False
            with self.pool.lock:
                self.pool.pending -= 1


class ModelPool:
    """One model replica per device, each in its own spawned process: the forward's dispatch (collate, upload,
    launch, readback) needs the GIL at every step and in the server process shared it with the event loop, adapter
    and recorder threads (a 24-sample batch took 1.4 s against a busy thread, 315 ms alone). A shared queue gives
    each idle replica the next batch; the batch closes as soon as no accepted request is still being staged
    (expect()), so a lone request waits for nothing and a lockstep group arrives as one forward (a quiet window
    after the last arrival was measured and lost steady req/s, docs/throughput.md §8.1). Replicas on the same
    device each hold XLA's memory fraction."""

    def __init__(self, params_dir: Path, devices: list[int], warm_input: dict, *, buckets=(1, 2, 4), num_steps: int = 10):
        import multiprocessing as mp

        self.buckets = tuple(sorted(buckets))
        self.work: queue.Queue = queue.Queue()
        self.pending, self.lock = 0, threading.Lock()
        ctx, conns = mp.get_context("spawn"), []
        for d in devices:
            conn, child = ctx.Pipe()
            ctx.Process(target=_replica_main, args=(child, str(params_dir), d, warm_input, self.buckets, num_steps),
                        daemon=True).start()
            child.close()
            conns.append(conn)
        for conn in conns:
            status, _, detail = conn.recv()
            if status != "ready":
                raise RuntimeError(f"model replica failed to start:\n{detail}")
        for conn in conns:
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn) -> None:
        for seq in itertools.count():
            batch = [self.work.get()]
            nvtx.push("batch_close")
            t0 = time.perf_counter()
            while len(batch) < self.buckets[-1] and time.perf_counter() - t0 < STAGING_CAP_S:
                try:
                    batch.append(self.work.get_nowait())
                except queue.Empty:
                    if not self.pending:
                        break
                    time.sleep(0.0005)
            nvtx.pop()
            with nvtx.range(f"batch_rpc {len(batch)}"):
                conn.send((seq, [(inp, nz) for inp, nz, _, _ in batch]))
                try:
                    while (msg := conn.recv())[0] == "log":
                        log.info(msg[2])
                    status, _, payload = msg
                except EOFError:  # the replica process died: fail this batch, every later one fails at send
                    log.error("model replica exited")
                    status, payload = "error", "model replica exited"
            outs = payload if status == "ok" else [None] * len(batch)
            for (_, _, loop, fut), out in zip(batch, outs, strict=True):
                loop.call_soon_threadsafe(_resolve, fut, out, None if status == "ok" else RuntimeError(payload))
            if status == "error" and payload == "model replica exited":
                return

    def expect(self) -> _Pending:
        """Announce an accepted request before staging it; close() the token once queued (__call__ does) or failed."""
        with self.lock:
            self.pending += 1
        return _Pending(self)

    async def __call__(self, model_input: dict, noise: np.ndarray, pending: _Pending | None = None,
                       trial=None) -> np.ndarray:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.work.put((model_input, noise, loop, fut))
        if pending is not None:
            pending.close()
        return await fut


def handler_factory(track: dict, policy_id: str, identity: str, adapter: ObservationAdapter, model, *,
                    k: int = 8, fade: int = 2, smooth: int = 9, config_extra: dict | None = None, recorder=None,
                    seed_mode: str = "sample_seed", orient_per_trial: bool = False, start_qpos: dict | None = None):
    """`model(model_input, noise, trial)` is awaited per request; `trial` = (session_id, trial_id) lets a stateful
    policy keep per-trial state. `config_extra` is merged into the advertised
    config_metadata. `recorder`: jenga.record.Recorder, given every reset and infer (diagnosis only).
    `orient_per_trial`: the official evaluator sends the external cameras upside down in some trials and not in others
    under identical headers (docs/method.md 7.2); the first frame of each trial decides by the static station background (jenga.orient.decide_flip), and the external cameras are
    turned once more for the rest of that trial when it says so.
    `start_qpos`: variant key -> 16-D training start state (configs/start_qpos.json). The official task1-2 trials
    start 165 mrad away from it, a state no training episode is near, and the policy then grasps the second block
    low and off-centre (docs/distill.md 5.2). A trial whose first state differs by more than START_TOL_RAD is first
    moved there (homing_plan) and handed to the policy afterwards; other trials are unchanged."""
    if not 1 <= k <= track["max_chunk_actions"]:
        raise ValueError("k must fit the release's max_chunk_actions")
    if abs(track["action_dt_s"] - CONTROL_DT_S) > 1e-9:  # a 10 Hz track would play every 30 Hz row for 0.1 s
        raise ValueError(f"model is trained at 30 Hz; track action_dt_s is {track['action_dt_s']}")
    sessions: dict[str, dict] = {}

    def flipped(observation: dict) -> tuple[bool, dict]:
        """(turn the external cameras again, decision record) from a trial's first frame in dataset orientation, by
        the static station background (jenga.orient: blocks are not needed, so empty scenes and relit frames decide
        too)."""
        from jenga.orient import decide_flip
        return decide_flip(observation.get("task_variant"), observation["images"])

    def turn_external(observation: dict) -> dict:
        from jenga.offset import EXTERNAL
        images = {c: (np.ascontiguousarray(np.asarray(im)[::-1, ::-1]) if c in EXTERNAL else im)
                  for c, im in observation["images"].items()}
        return {**observation, "images": images}
    retention = max(300, ceil(track["inference_timeout_s"] + 60))

    def session_for(request, ws):
        now = time.monotonic()
        for key in [s for s, v in sessions.items() if v["connections"] == 0 and now - v["detached_at"] > retention]:
            del sessions[key]
        session_id = request["session_id"]
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError("invalid_session")
        binding = tuple(ws.request.headers.get(h) for h in
                        ("X-Tower-Task-Variant", "X-Tower-Track-Version", "X-Tower-Evaluation-Version"))
        state = sessions.get(session_id)
        if state is None:
            if request["type"] != "reset":
                raise ValueError("session_expired")
            if len(sessions) >= 64:
                raise ValueError("session_capacity")
            state = dict(lock=asyncio.Lock(), binding=binding, connections=0, detached_at=now, trial=None,
                         trials=set(), seed=None, index=0, seen=set(), last_id=None, last_digest=None,
                         response=None, invalid=False, fader=ChunkFader(fade, smooth, issued=k), last_step=None)
            sessions[session_id] = state
        if state["binding"] != binding:
            raise ValueError("session_binding_mismatch")
        return state

    def prepare(observation: dict) -> dict:
        try:
            with nvtx.range("adapter"):
                return adapter(observation)
        except (TypeError, AttributeError, IndexError, KeyError) as exc:  # malformed fields, not a server fault
            raise ValueError("invalid_observation") from exc

    async def infer(observation: dict, seed: int, pending, trial: tuple) -> tuple[np.ndarray, dict]:
        t0 = time.perf_counter()
        model_input = await asyncio.to_thread(prepare, observation)
        noise = np.random.default_rng(seed).standard_normal((32, 32)).astype(np.float32)
        t1 = time.perf_counter()
        out = await model(model_input, noise, pending, trial)
        t2 = time.perf_counter()
        return out, {"adapter_ms": (t1 - t0) * 1e3, "model_ms": (t2 - t1) * 1e3}

    def stage(r: dict, raw: bytes) -> tuple[bytes, dict | None]:
        """Per-request byte work off the event loop (3.7 MB raw renders per request): the payload digest and the
        images turned to the dataset orientation; hashlib and numpy release the GIL for both."""
        with nvtx.range("stage"):
            content = hashlib.sha256(raw).digest()
            if r.get("type") != "infer":
                return content, None
            observation = r["observation"]
            if not isinstance(observation, dict) or not isinstance(observation.get("images"), dict):
                raise ValueError("invalid_observation")
            return content, to_dataset_orientation(observation)  # estimator, adapter and records all see this

    async def process(r, raw, state):
        if state["invalid"]:
            raise ValueError("session_unavailable")
        base = {key: r[key] for key in IDENTITY_FIELDS}
        request_id = r["request_id"]
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("invalid_request_id")
        content, observation = await asyncio.to_thread(stage, r, raw)
        if request_id == state["last_id"]:
            if content != state["last_digest"]:
                raise ValueError("request_payload_mismatch")
            return state["response"]
        if request_id in state["seen"]:
            raise ValueError("stale_request")
        if len(state["seen"]) >= 100000:
            raise ValueError("session_request_limit")
        if r["type"] == "reset":
            if r["trial_id"] in state["trials"]:
                raise ValueError("repeated_trial")
            seed = r["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("invalid_seed")
            response = dict(**base, type="reset_ack", base_seed=seed, seed_scheme="tower-seed-stream-v1",
                            next_sample_index=0)
            state.update(trial=r["trial_id"], seed=seed, index=0, last_step=None, flip=None, homing=None)
            if recorder is not None:
                recorder.put(r["trial_id"], state["binding"][0], dict(
                    type="reset", t=time.time(), session_id=r["session_id"], seed=seed, binding=list(state["binding"])))
            state["fader"].reset()
            state["trials"].add(r["trial_id"])
        elif r["type"] == "infer":
            if state["trial"] is None or r["trial_id"] != state["trial"]:
                raise ValueError("reset_required")
            instruction = observation.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():  # as the reference server
                raise ValueError("invalid_instruction")
            # step_index counts control actions already executed (tower-starter contract.md); checked before any
            # model work so a malformed request costs nothing.
            step = observation.get("step_index")
            if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
                raise ValueError("invalid_step_index")
            step = int(step)
            if orient_per_trial:
                if state.get("flip") is None:
                    state["flip"], est = await asyncio.to_thread(flipped, observation)
                    log.info("trial %s %s: external cameras %s %s", r["trial_id"], observation.get("task_variant"),
                             "turned" if state["flip"] else "as is", est)
                if state["flip"]:
                    observation = await asyncio.to_thread(turn_external, observation)
            seed, index = state["seed"], state["index"]
            s_seed = sample_seed(seed, index)
            try:
                q = state16(observation)
            except (TypeError, AttributeError, IndexError, KeyError) as exc:  # as prepare(): malformed fields
                raise ValueError("invalid_observation") from exc
            if start_qpos is not None and state.get("homing") is None:
                target = start_qpos.get(str(observation.get("task_variant", "")).split("_")[0])
                plan = homing_plan(q, target) if target is not None else None
                state["homing"] = (step, plan)
                if plan is not None:
                    log.info("trial %s %s: start state %.0f mrad from training start, homing for %d steps",
                             r["trial_id"], observation.get("task_variant"),
                             1e3 * float(np.abs(q - target)[JOINT_COLUMNS].max()), len(plan))
            step0, plan = state.get("homing") or (None, None)
            if plan is not None and step - step0 < len(plan):
                rows = plan[min(step - step0, len(plan) - 1):][:k]
                chunk = np.concatenate([rows, np.repeat(rows[-1:], k - len(rows), axis=0)])
                timing = {"adapter_ms": 0.0, "model_ms": 0.0}
                state["fader"].reset()  # the first model chunk after homing is not faded against the plan
            else:
                pending = model.expect()
                try:
                    out, timing = await infer(observation, s_seed, pending, (r["session_id"], r["trial_id"]))
                finally:
                    pending.close()  # a request that failed before reaching the queue
                # Commands of the previous chunk run since it was issued; ChunkFader skips the fade if this is not
                # in [1, len(previous chunk)) (a re-sent frame, a step that went backwards, or a gap).
                executed = k if state["last_step"] is None else step - state["last_step"]
                chunk = state["fader"](adapter.decode(out, q), executed)[:k]
            actions = clip_to_bounds(chunk, track["action_lower"], track["action_upper"])
            response = dict(**base, type="infer_result", plugin=PLUGIN, plugin_version=PLUGIN_VERSION,
                            base_seed=seed, sample_index=index, sample_seed=s_seed,
                            seed_scheme="tower-seed-stream-v1", actions=actions, dt_s=track["action_dt_s"],
                            action_metadata={}, server_timing=timing)
            state["index"] += 1
            state["last_step"] = step
            if recorder is not None:
                recorder.put(r["trial_id"], state["binding"][0], dict(
                    type="infer", t=time.time(), sample_index=index, step_index=step,
                    sim_time_s=observation.get("sim_time_s"), task_variant=observation.get("task_variant"),
                    instruction=instruction, joint_positions=observation.get("joint_positions"),
                    gripper_widths_m=observation.get("gripper_widths_m"), images=observation.get("images") or {},
                    actions=actions, server_timing=timing))
        else:
            raise ValueError("unsupported_request")
        with nvtx.range("pack"):
            encoded = pack(response)
        state["seen"].add(request_id)
        state.update(last_id=request_id, last_digest=content, response=encoded)
        return encoded

    metadata = pack(dict(
        type="metadata", protocol="tower-infer-v1", ready=True, policy_id=policy_id, plugin=PLUGIN,
        plugin_version=PLUGIN_VERSION, plugin_api_version=1, seed_scheme="tower-seed-stream-v1",
        seed_mode=seed_mode, supports_seed_reset=True, artifact_identity=identity,
        observation_spec=observation_spec(track), action_spec=action_spec(track), policy_metadata={},
        config_metadata={"k": k, "fade": fade, "smooth": smooth, "prompt": adapter.prompt,
                         "jpeg_quality": adapter.jpeg_quality, **(config_extra or {})},
        reconnect=dict(protocol="tower-reconnect-v1", idempotent_requests=True, retention_s=retention),
    ))

    async def handle(ws):
        from websockets.exceptions import ConnectionClosed

        state = session_id = None
        try:
            await ws.send(metadata)
            async for raw in ws:
                with nvtx.range("unpack"):
                    r = unpack(raw)
                try:
                    if r.get("protocol") != "tower-infer-v1" or r.get("policy_id") != policy_id:
                        raise ValueError("identity_mismatch")
                    if state is None:
                        state = session_for(r, ws)
                        session_id = r["session_id"]
                        state["connections"] += 1
                    if r["session_id"] != session_id:
                        raise ValueError("session_mismatch")
                    async with state["lock"]:
                        try:
                            response = await process(r, raw, state)
                        except BaseException:
                            state["invalid"] = True
                            raise
                    await ws.send(response)
                except (ValueError, KeyError) as exc:
                    code = "missing_field" if isinstance(exc, KeyError) else str(exc)
                    await ws.send(pack(dict(protocol="tower-infer-v1", type="error", request_id=r.get("request_id"),
                                            error={"code": code, "message": "Protocol request rejected"})))
                    await ws.close(code=1008)
                    return
        except ConnectionClosed:
            pass
        finally:
            if state is not None:
                state["connections"] -= 1
                state["detached_at"] = time.monotonic()

    return handle
