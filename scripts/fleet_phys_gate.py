"""Physics gate for sim/fleet_driver.py's persistent workers (no GPU): the same episodes (demo, layout offset) driven by
the same action rows, once with a worker per episode (the old --fresh-workers path) and once through persistent
workers that each run several episodes in a row, set up the way the driver does (Physics.episode with the driver's
JENGA_* variables, wrapped in fleet_driver.Started). Prints the max |state difference| per episode; 0 means identical.
Actions: the demo's leader qpos rows (an arbitrary but realistic command stream), `--ticks` of them.

  runs/simdev/render_venv/bin/python scripts/fleet_phys_gate.py runs/simdev/rollout/val_offset.txt <out_dir>
      [--episodes 6] [--workers 2] [--ticks 1200]
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from fleet_driver import Started, probe_env  # noqa: E402
from physics_client import Physics  # noqa: E402


def drive(ph, actions: np.ndarray) -> np.ndarray:
    head, s0 = ph.connect()
    rows = [s0] + [ph.step(actions[i:i + 8]) for i in range(0, len(actions), 8)]
    ph.close()
    return np.concatenate([r.reshape(-1, len(s0)) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=1200)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [w for w in (line.split() for line in args.jobs.read_text().splitlines()) if w]
    by_var = {}
    for w in lines:  # one episode per variant first, so the persistent workers switch variants between episodes
        by_var.setdefault(Path(w[0]).parent.name, []).append(w)
    picks = [v[0] for v in by_var.values()][: args.episodes]
    jobs = [(w[0], [float(x) for x in w[1:4]] or None) for w in picks]
    acts = []
    for d, _ in jobs:
        with h5py.File(d, "r") as f:
            acts.append(np.asarray(f["leader/qpos"][: args.ticks], float))

    fresh, pers = [None] * len(jobs), [None] * len(jobs)

    def one_fresh(k):
        d, o = jobs[k]
        fresh[k] = drive(Physics(d, o, args.out / f"fresh_{k}.physics.log"), acts[k])  # as the old driver: no shim

    def one_worker(w):
        p, env = Physics(None, log=args.out / f"pers_{w}.physics.log", nosmi=True), probe_env()
        for k in range(w, len(jobs), args.workers):
            d, o = jobs[k]
            pers[k] = drive(Started(p, p.episode(d, o, env=env)), acts[k])
        p.close()

    ts = [threading.Thread(target=one_fresh, args=(k,)) for k in range(len(jobs))] + \
         [threading.Thread(target=one_worker, args=(w,)) for w in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    res = []
    for k, (d, o) in enumerate(jobs):
        a, b = fresh[k], pers[k]
        res.append(dict(demo=Path(d).parent.name + "/" + Path(d).name.split("_")[1], offset=o, rows=len(a),
                        same_shape=a.shape == b.shape, max_abs=float(np.abs(a - b).max()) if a.shape == b.shape else None))
        print(json.dumps(res[-1]), flush=True)
    ok = all(r["same_shape"] and r["max_abs"] == 0 for r in res)
    (args.out / "gate.json").write_text(json.dumps(dict(ok=ok, episodes=res), indent=1))
    print("PHYS_GATE", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
