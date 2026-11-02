"""Where do goal operations stop? Per rollout dir, the share of goal operations (pushdown excluded) whose block
  lifted (rose 30 mm above its start at some point) -> reached (came within 20 mm of its goal) -> placed (scored),
for the first operation of each episode and for all of them; plus placed-then-lost (placed at some frame, not at the end).
runs/simdev/render_venv/bin/python sim/funnel.py <dir> [...]"""
import glob
import json
import sys
import h5py
import numpy as np


def stages(p):
    r, blocks = json.load(open(p)), np.load(p[:-5] + ".npz")["blocks"]
    with h5py.File(r["demo"]) as f:
        mapping = json.loads(f["sim"].attrs["target_mapping"])
    rows = []
    for k, m in enumerate(mapping):
        if "goal" not in m:
            continue
        b = blocks[:, m["block_index"]]
        d = np.linalg.norm(b[:, :3] - np.asarray(m["goal"][:3]), axis=-1)
        rows.append(dict(first=not rows, lifted=(b[:, 2] - b[0, 2]).max() > 0.03, reached=d.min() <= 0.02,
                         placed=r["ops"][k], lost=bool((d <= 0.01).any()) and not r["ops"][k]))
    return rows


for d in sys.argv[1:]:
    rows = [x for p in sorted(glob.glob(f"{d}/*.json")) for x in stages(p)]
    print(d, f"{len(rows)} goal ops")
    for name, sel in (("first op", [x for x in rows if x["first"]]), ("all ops", rows)):
        n = len(sel)
        print(f"  {name:9s} n={n:3d} " + " ".join(f"{k} {sum(x[k] for x in sel) / n:.2f}" for k in ("lifted", "reached", "placed", "lost")))
