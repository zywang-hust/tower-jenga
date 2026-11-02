"""Settle test: hold both arms at the demo's first command and let the blocks rest for N seconds; report each block's
z drift. Answers whether the local engine keeps a tower still from the state the official one recorded.

Run: sim/isaac.sh sim/settle.py <demo.hdf5> <out.json> [--seconds 10] [--start frame0|initial] [--fidelity]
     [--dims] [--mass] [--collider cuboid|convexHull|...] [--offsets CONTACT REST (mm)] [--unfreeze] [--dump]
--fidelity is JENGA_PHYS_FIDELITY=1 (sim/scene.py fidelity_setup); the other switches set its parts one by one.
"""
import argparse
import json
import sys
from pathlib import Path

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("demo")
ap.add_argument("out", type=Path)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--start", choices=("frame0", "initial"), default="frame0")
ap.add_argument("--fidelity", action="store_true")
ap.add_argument("--dims", action="store_true")
ap.add_argument("--mass", action="store_true")
ap.add_argument("--collider")
ap.add_argument("--offsets", type=float, nargs=2)
ap.add_argument("--unfreeze", action="store_true")
ap.add_argument("--scene", nargs="*", default=[], help="physxScene:<name>=<value> (diagnosis)")
ap.add_argument("--body", nargs="*", default=[], help="physxRigidBody:<name>=<value> on the blocks (diagnosis)")
ap.add_argument("--dump", action="store_true")
args = ap.parse_args()
sys.argv = sys.argv[:1]

app = SimulationApp({"headless": True, "limit_cpu_threads": 2}, experience=str(Path(__file__).with_name("physics_only.kit")))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from render import STATION  # noqa: E402
from scene import ACTION_HZ, Station, episode_setup, fidelity_setup  # noqa: E402

with h5py.File(args.demo, "r") as f:
    setup = episode_setup(f["sim"].attrs, fidelity=args.fidelity)
    fid = fidelity_setup(f["sim"].attrs)
    q0 = f["observations/qpos"][0]
    lead0 = f["leader/qpos"][0]
    demo_b = f["sim/block_poses"][:]
setup["gripper"] = None
if args.dims:
    setup["block_dims"] = fid["block_dims"]
if args.mass:
    setup["block_mass"] = fid["block_mass"]
if args.collider:
    setup["block_collider"] = args.collider
if args.offsets:
    setup["block_offsets"] = args.offsets
def _kv(items, prefix):
    out = {}
    for it in items:
        k, v = it.split("=")
        out[prefix + k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.lstrip("-").isdigit() else
                                                             (float(v) if v.replace(".", "").replace("-", "").replace("e", "").isdigit() else v))
    return out


setup["scene_attrs"], setup["body_attrs"] = _kv(args.scene, "physxScene:"), _kv(args.body, "physxRigidBody:")
if args.unfreeze:
    setup["frozen"] = fid["frozen"]
if args.start == "frame0":
    setup["initial_block_poses"] = demo_b[0].astype(float)
station = Station(STATION, setup)

if args.dump:
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if p.GetTypeName() == "PhysicsScene" or any("Physx" in s for s in p.GetAppliedSchemas()) and "ReplayTaskBlocks" in str(p.GetPath()):
            attrs = {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.HasAuthoredValue() or p.GetTypeName() == "PhysicsScene"}
            print("DUMP", p.GetPath(), p.GetTypeName(), p.GetAppliedSchemas(), json.dumps(attrs)[:3000], flush=True)
    blk = stage.GetPrimAtPath(station.block_paths[-1])
    for c in Usd.PrimRange(blk):
        print("DUMP_BLOCK", c.GetPath(), c.GetTypeName(), c.GetAppliedSchemas(),
              {a.GetName(): str(a.Get()) for a in c.GetAttributes() if a.HasAuthoredValue() and "phys" in a.GetName().lower()}, flush=True)

placed = setup["initial_block_poses"][:, 2] * 1000  # the engine already stepped in play(): measure from the placed pose
station.reset(q0)
zs = [station.block_poses()[:, 2].copy()]
for _ in range(int(args.seconds * ACTION_HZ)):
    station.tick(lead0)
    zs.append(station.block_poses()[:, 2].copy())
zs = np.array(zs) * 1000
names = [p.rsplit("/", 1)[-1] for p in station.block_paths]
with h5py.File(args.demo, "r") as f:
    ids = json.loads(f["sim"].attrs["block_ids"])
one_s = ACTION_HZ
res = dict(demo=args.demo, args={k: v for k, v in vars(args).items() if k not in ("demo", "out")}, seconds=args.seconds,
           blocks={b: dict(prim=n, placed=round(float(placed[i]), 3), z0=round(float(zs[0, i]), 3), demo_f0=round(float(demo_b[0, i, 2] * 1000), 3),
                           z_1s=round(float(zs[one_s, i]), 3), z_end=round(float(zs[-1, i]), 3),
                           drift=round(float(zs[-1, i] - placed[i]), 3), drift_after_1s=round(float(zs[-1, i] - zs[one_s, i]), 3),
                           max_abs=round(float(np.abs(zs[:, i] - placed[i]).max()), 3))
                   for i, (b, n) in enumerate(zip(ids, names))})
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
np.save(args.out.with_suffix(".npy"), zs)
print("SETTLE_DONE", json.dumps({b: (v["placed"], v["z_end"], v["drift"]) for b, v in res["blocks"].items()}), flush=True)
app.close()
