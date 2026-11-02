"""Open-loop replay of a demonstration's commands (leader/qpos) through sim.scene.Station, compared row by row with
what the official simulator recorded. The official replay of the same commands is exact (replay_comparison_json:
all maximum errors 0), so any difference here is ours: parameters, controller, or engine configuration.

Run: GPU=2 sim/isaac.sh sim/replay.py <episode.hdf5> <out.npz> [--rows N] [--device cpu|cuda:0] [--friction S D]
     [--eval-gripper]  (the evaluation worker's gripper: drive target = commanded width, no contact latch)
     [--start-frame0]; JENGA_PHYS_FIDELITY=1 builds the blocks from the episode attributes (sim/scene.py)
"""
import argparse
import json
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("out", type=Path)
ap.add_argument("--rows", type=int)
ap.add_argument("--device", default="cpu")
ap.add_argument("--friction", type=float, nargs=2)
ap.add_argument("--eval-gripper", action="store_true")
ap.add_argument("--scene", nargs="*", default=[], help="diagnosis: physxScene:<name>=<value>")
ap.add_argument("--body", nargs="*", default=[], help="diagnosis: physxRigidBody:<name>=<value> on the blocks")
ap.add_argument("--offsets", type=float, nargs=2, metavar=("CONTACT", "REST"), help="diagnosis: block collider offsets, mm")
ap.add_argument("--role-friction", nargs=3, action="append", metavar=("PREFIX", "S", "D"),
                help="diagnosis: friction of the blocks whose id starts with PREFIX")
ap.add_argument("--start-frame0", action="store_true", help="blocks start at the recorded first frame (after the official "
                "reset settle) instead of initial_block_poses")
args = ap.parse_args()
episode, out, rows = args.episode, args.out, args.rows
sys.argv = sys.argv[:1]  # Kit would parse the rest

app = SimulationApp({"headless": True, "limit_cpu_threads": 2}, experience=str(Path(__file__).with_name("physics_only.kit")))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from render import STATION  # noqa: E402
from scene import Station, episode_setup  # noqa: E402
from score import score  # noqa: E402

with h5py.File(episode, "r") as f:
    setup = episode_setup(f["sim"].attrs)
    leader = f["leader/qpos"][:rows]
    qpos = f["observations/qpos"][:rows]
    blocks = f["sim/block_poses"][:rows]
    target_mapping, config = (json.loads(f["sim"].attrs[k]) for k in ("target_mapping", "config_json"))

setup["block_friction"] = args.friction
def _kv(items, prefix):
    out = {}
    for it in items:
        k, v = it.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                v = {"true": True, "false": False}.get(v.lower(), v)
        out[prefix + k] = v
    return out


setup["scene_attrs"], setup["body_attrs"] = _kv(args.scene, "physxScene:"), _kv(args.body, "physxRigidBody:")
if args.offsets:
    setup["block_offsets"] = args.offsets
setup["role_friction"] = [(p, float(a), float(b)) for p, a, b in args.role_friction or []]
if args.eval_gripper:
    setup["gripper"] = None
if args.start_frame0:
    setup["initial_block_poses"] = blocks[0].astype(float)
station = Station(STATION, setup, args.device)
mismatch = {arm: {k: float(np.abs(v - setup["readback"][arm][k]).max() / (np.abs(v).max() + 1e-9))
                  for k, v in rb.items()} for arm, rb in station.readback().items()}
print("READBACK_REL_ERR", mismatch, flush=True)

station.reset(qpos[0])
if setup.get("reset_settle_s") and not args.start_frame0:  # fidelity: the generator's reset settle from initial_block_poses
    station.settle(qpos[0], setup["reset_settle_s"])
sim_q, sim_b = [], []
t0 = time.perf_counter()
for row in range(len(leader)):
    q, b = station.state()
    sim_q.append(q)
    sim_b.append(b)
    station.tick(leader[row])
wall = time.perf_counter() - t0
sim_q, sim_b = np.array(sim_q), np.array(sim_b)

err_q = np.abs(sim_q - qpos)
err_b = np.linalg.norm(sim_b[..., :3] - blocks[..., :3], axis=-1)
np.savez_compressed(out, sim_qpos=sim_q, sim_blocks=sim_b, qpos=qpos, blocks=blocks, leader=leader)
summary = dict(episode=episode, device=args.device, fidelity=setup.get("block_dims") is not None, start_frame0=args.start_frame0, friction=args.friction, role_friction=setup["role_friction"], scene_attrs=setup["scene_attrs"], offsets=args.offsets, body_attrs=setup["body_attrs"], eval_gripper=args.eval_gripper, rows=len(leader), ms_per_row=round(wall / len(leader) * 1e3, 2),
               arm_rad_max=np.round(err_q[:, [*range(7), *range(8, 15)]].max(0), 4).tolist(),
               arm_rad_p95=np.round(np.percentile(err_q[:, [*range(7), *range(8, 15)]], 95, 0), 4).tolist(),
               gripper_m_max=np.round(err_q[:, [7, 15]].max(0), 4).tolist(),
               block_m_max=np.round(err_b.max(0), 4).tolist(),
               sim_score=score(setup, sim_b[-1], target_mapping, config), demo_score=score(setup, blocks[-1], target_mapping, config))
out.with_suffix(".json").write_text(json.dumps(summary))
print("REPLAY_DONE", summary, flush=True)
app.close()
