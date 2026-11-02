"""Physics half of a closed-loop episode: one sim.scene.Station with the evaluation worker's controller (demo
parameters, gripper drive = commanded width: sim/replay.py --eval-gripper), served to sim/rollout.py over a local socket.

The two halves run in different interpreters (Isaac's Python 3.11, the renderer's 3.12), so only bytes cross:
  worker -> JSON header {block_names, block_fixed, initial_block_poses}, then the reset state;
  driver -> float64 actions [n, 16]; worker -> float64 states [n, 16 + 7 * blocks] after each tick;
  driver -> b"INIT:" + JSON {"npz": ..., "row": ...} (length never a multiple of 128, unlike a block of 16-float64
            action rows): the arms and free blocks move to that logged state, velocities zero; worker -> that state;
  driver -> b"EPISODE:" + JSON {"demo", "offset", "init", "env"} (persistent worker, started with demo "-"): drop the
            current station, open the stage afresh and build that episode as a worker started with these arguments and
            JENGA_* variables `env` would; worker -> its header, then its reset state (b"ERROR:..." for a bad request);
  driver -> empty message: quit.
A state row is qpos16 then the block poses (x y z qx qy qz qw, metres).

--init NPZ ROW starts from a logged state (qpos, blocks of a rollout npz: sim/rollout.py, sim/expert_rollout.py)
instead of the demo's first row; the header still carries the episode's own initial block poses, which scoring judges
against. Several takeovers then share one Isaac start-up through INIT messages (sim/expert_dagger.py).

Run (started by sim/rollout.py): sim/isaac.sh sim/physics_worker.py <demo.hdf5> <socket> [--offset DX DY DPSI]
    [--init NPZ ROW]
The socket key is read from JENGA_PHYSICS_KEY; JENGA_PHYS_FIDELITY=1 builds the blocks from the episode attributes,
leaves the task5-3 tower dynamic and runs the reset settle (sim/scene.py fidelity_setup); JENGA_FRICTION="<static>,<dynamic>" and JENGA_MASS_SCALE override the block friction and mass; JENGA_LAYER_JITTER="<seed>,<mm>,<deg>" shifts each movable layer independently (jitter_layers)
(robustness probes, docs/research.md §8.1).
"""
import argparse
import json
import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("demo", help='"-": persistent worker, episodes come as EPISODE messages')
ap.add_argument("socket")
ap.add_argument("--offset", type=float, nargs=3, metavar=("DX", "DY", "DPSI"))
ap.add_argument("--init", nargs=2, metavar=("NPZ", "ROW"))
args = ap.parse_args()
sys.argv = sys.argv[:1]  # Kit would parse the rest

app = SimulationApp({"headless": True, "limit_cpu_threads": 2},  # 1 deadlocks carb.tasking at startup
                    experience=str(Path(__file__).with_name("physics_only.kit")))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from render import STATION  # noqa: E402
from scene import MM, Station, episode_setup  # noqa: E402


def offset_source(setup: dict, dx: float, dy: float, dpsi: float) -> np.ndarray:
    """Initial poses with the movable assembly turned by dpsi about its centre, then shifted: the offset as
    jenga.offset estimates it (source blocks except the source foundation, about their xy mean)."""
    poses = setup["initial_block_poses"].copy()
    src = np.array([b.startswith("source_") and not b.startswith("source_base") for b in setup["block_ids"]])
    c = poses[src, :2].mean(0)
    rot = np.array([[np.cos(dpsi), -np.sin(dpsi)], [np.sin(dpsi), np.cos(dpsi)]])
    poses[src, :2] = (poses[src, :2] - c) @ rot.T + c + (dx, dy)
    half = np.array([0, 0, np.sin(dpsi / 2), np.cos(dpsi / 2)])  # z rotation, xyzw
    poses[src, 3:] = [quat_mul(half, q) for q in poses[src, 3:]]
    return poses


def jitter_layers(setup: dict, seed: int, mm: float, deg: float) -> np.ndarray:
    """Robustness probe (the release's "Irregular layers" case: each layer independently up to 8 mm and 5 deg): every
    layer of movable source blocks (same height, foundation excluded) shifted by U(-mm, mm) per axis and turned by
    U(-deg, deg) about its own xy mean, independently of the other layers; seeded per episode."""
    poses = setup["initial_block_poses"].copy()
    src = np.array([b.startswith("source_") and not b.startswith("source_base") for b in setup["block_ids"]])
    rng = np.random.default_rng(seed)
    for z in sorted({round(float(z) * 1000 / 5) for z in poses[src, 2]}):
        lay = src & (np.round(poses[:, 2] * 1000 / 5) == z)
        dx, dy = rng.uniform(-mm, mm, 2) / 1000
        dpsi = np.radians(rng.uniform(-deg, deg))
        c = poses[lay, :2].mean(0)
        rot = np.array([[np.cos(dpsi), -np.sin(dpsi)], [np.sin(dpsi), np.cos(dpsi)]])
        poses[lay, :2] = (poses[lay, :2] - c) @ rot.T + c + (dx, dy)
        half = np.array([0, 0, np.sin(dpsi / 2), np.cos(dpsi / 2)])
        poses[lay, 3:] = [quat_mul(half, q) for q in poses[lay, 3:]]
    return poses


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    (ax, ay, az, aw), (bx, by, bz, bw) = a, b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def logged(npz: str, row: int) -> tuple[np.ndarray, np.ndarray]:
    log = np.load(npz)
    return log["qpos"][int(row)].astype(float), log["blocks"][int(row)].astype(float)


def init(npz: str, row: int) -> None:
    """Move the arms and the free blocks to a logged state, velocities zero."""
    q, b = logged(npz, row)
    free = ~setup["block_fixed"]
    if station.free is not None:
        idx = np.arange(free.sum(), dtype=np.int32)
        station.free.set_transforms(np.c_[b[free, :3] * MM, b[free, 3:]].astype(np.float32), idx)
        station.free.set_velocities(np.zeros((free.sum(), 6), np.float32), idx)
    station.reset(q)


def state() -> np.ndarray:
    q, blocks = station.state()
    return np.r_[q, blocks.ravel()]


def build(demo: str, offset=None, init_state=None, env=None) -> tuple[dict, np.ndarray, np.ndarray]:
    """The episode's station setup as a worker started with these arguments (and JENGA_* variables `env`) builds it
    -> setup, initial block poses (the header's), reset qpos16."""
    env = os.environ if env is None else env
    with h5py.File(demo, "r") as f:
        setup = episode_setup(f["sim"].attrs, fidelity=env.get("JENGA_PHYS_FIDELITY", "0") == "1")
        q0 = f["observations/qpos"][0]
    setup["gripper"] = None  # evaluation worker: no contact latch
    if env.get("JENGA_FRICTION"):
        setup["block_friction"] = [float(x) for x in env["JENGA_FRICTION"].split(",")]
    if env.get("JENGA_MASS_SCALE"):  # robustness probe: the authored block mass scaled
        setup["block_mass_scale"] = float(env["JENGA_MASS_SCALE"])
    if offset:
        setup["initial_block_poses"] = offset_source(setup, *offset)
    if env.get("JENGA_DIAG_TOWER_DZ_MM"):  # diagnosis only: the frozen task5-3 tower moved up/down (mm)
        setup["initial_block_poses"] = setup["initial_block_poses"].copy()
        setup["initial_block_poses"][np.asarray(setup["frozen"], bool), 2] += float(env["JENGA_DIAG_TOWER_DZ_MM"]) / 1000
    if env.get("JENGA_LAYER_JITTER"):  # "<seed>,<mm>,<deg>": per-layer independent offsets (jitter_layers)
        import zlib
        base, mm, deg = (float(x) for x in env["JENGA_LAYER_JITTER"].split(","))
        seed = zlib.crc32(f"{Path(demo).name} {int(base)}".encode())
        setup["initial_block_poses"] = jitter_layers(setup, seed, mm, deg)
    initial = setup["initial_block_poses"]
    if init_state:
        q0, setup["initial_block_poses"] = logged(*init_state)
    return setup, initial, q0


def start(setup: dict, q0: np.ndarray, from_init: bool) -> Station:
    station = Station(STATION, setup)
    station.reset(q0)
    if setup.get("reset_settle_s") and not from_init:  # JENGA_PHYS_FIDELITY=1: the generator's reset settle
        station.settle(q0, setup["reset_settle_s"])
    return station


def release() -> None:
    """Drop the current episode's station: stop the timeline (PhysX releases the scene), forget the simulation
    context, close the stage and revert every layer the episode edited (block placement, activation, colliders,
    physics scene), so the next open_stage reads the station as a fresh process does."""
    import omni.usd
    from isaacsim.core.api import SimulationContext
    from pxr import Sdf
    station.sim.stop()
    SimulationContext.clear_instance()
    ctx = omni.usd.get_context()
    dirty = [layer.identifier for layer in ctx.get_stage().GetUsedLayers() if layer.dirty]
    ctx.close_stage()
    for ident in dirty:
        layer = Sdf.Layer.Find(ident)
        if layer is not None and layer.dirty:
            layer.Reload(force=True)
            assert not layer.dirty, ident


def header(initial: np.ndarray) -> bytes:
    return json.dumps(dict(block_names=[p.rsplit("/", 1)[1] for p in station.block_paths],
                           block_fixed=setup["block_fixed"].tolist(), initial_block_poses=initial.tolist())).encode()


station = None
if args.demo != "-":  # one episode from the command line; a persistent worker ("-") waits for EPISODE messages
    setup, initial, q0 = build(args.demo, args.offset, args.init)
    station = start(setup, q0, bool(args.init))

with Listener(args.socket, "AF_UNIX", authkey=bytes.fromhex(os.environ["JENGA_PHYSICS_KEY"])) as listener, \
        listener.accept() as conn:
    if station is not None:
        conn.send_bytes(header(initial))
        conn.send_bytes(state().tobytes())
    while msg := conn.recv_bytes():
        if msg.startswith(b"EPISODE:"):
            cmd = json.loads(msg[8:])
            print(f"EPISODE_START {cmd['demo']} {cmd.get('offset')} {cmd.get('env')}", flush=True)
            try:
                env = {k: v for k, v in os.environ.items() if not k.startswith("JENGA_") or k == "JENGA_PHYSICS_KEY"}
                setup, initial, q0 = build(cmd["demo"], cmd.get("offset"), cmd.get("init"), {**env, **cmd["env"]})
            except Exception as e:  # noqa: BLE001  a bad request, not a broken worker
                conn.send_bytes(b"ERROR:" + repr(e).encode())
                continue
            if station is not None:
                release()
            station = start(setup, q0, bool(cmd.get("init")))
            conn.send_bytes(header(initial))
            conn.send_bytes(state().tobytes())
            continue
        if msg.startswith(b"INIT:"):
            cmd = json.loads(msg[5:])
            init(cmd["npz"], cmd["row"])
            conn.send_bytes(state().tobytes())
            continue
        rows = []
        for action in np.frombuffer(msg).reshape(-1, 16):
            station.tick(action)
            rows.append(state())
        conn.send_bytes(np.array(rows).tobytes())
print("PHYSICS_WORKER_DONE", flush=True)
app.close()
