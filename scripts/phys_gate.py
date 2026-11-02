"""Compare two expert batches rollout by rollout (old sim/expert_batch.py vs new sim/expert_pool.py): scores equal,
max |d qpos|, |d blocks|, |d actions| over the episode.  python3 scripts/phys_gate.py <dir A> <dir B> [--tol 1e-6]"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a", type=Path)
ap.add_argument("b", type=Path)
ap.add_argument("--tol", type=float, default=1e-6)
args = ap.parse_args()
bad, n, worst = 0, 0, dict(qpos=0.0, blocks=0.0, actions=0.0, executed=0.0)
for ja in sorted(args.a.glob("task*.json")):
    jb = args.b / ja.name
    if not jb.exists():
        print("MISSING", ja.name)
        bad += 1
        continue
    ra, rb = json.loads(ja.read_text()), json.loads(jb.read_text())
    same = all(ra[k] == rb[k] for k in ("ops", "lhpr", "tsr", "ticks", "done_s", "offset", "noise", "final_err_mm"))
    za, zb = np.load(ja.with_suffix(".npz")), np.load(jb.with_suffix(".npz"))
    d = {}
    for k in ("qpos", "blocks", "actions", "executed"):
        if za[k].shape != zb[k].shape:
            d[k] = float("inf")
        else:
            d[k] = float(np.abs(za[k] - zb[k]).max())
        worst[k] = max(worst[k], d[k])
    ok = same and max(d.values()) <= args.tol and np.array_equal(za["active"], zb["active"])
    n += 1
    bad += not ok
    print("OK  " if ok else "DIFF", ja.stem, ra["ops"], ra["tsr"], rb["tsr"], {k: f"{v:.2e}" for k, v in d.items()})
print(f"GATE {n - bad}/{n} identical (tol {args.tol}); worst {worst}")
