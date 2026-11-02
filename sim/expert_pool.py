"""sim/expert_batch.py with persistent physics workers: N driver processes, each owning one Isaac worker
(sim/physics_worker.py "-") that boots once and runs every episode it is handed on a freshly opened stage. Same jobs
(same random draws for the same arguments), same per-rollout code (sim/expert_rollout.py run()), same output layout:
<out>/<variant>_<ep>[_r<k>].{json,npz,log} + summary.json + EXPERT_BATCH_DONE. The Isaac logs are
<out>/workers/w<k>.physics.log. JENGA_* probe variables set for the batch apply to every episode, as they do for
sim/expert_batch.py; per-job variables (--perturb d2: JENGA_LAYER_JITTER, JENGA_PHYS_FIDELITY) go with each episode.

Why (docs/throughput.md 8.4): of a 30-90 s expert rollout the episode loop is about 9 s; the rest is the worker's Kit
start-up (gpu.foundation / CUDA probes, extension load), its shutdown and the driver's own start-up, all paid once
per worker here instead of once per rollout.

  .venv/bin/python sim/expert_pool.py <out_dir> --workers 40 [all sim/expert_batch.py options except --jobs]
"""
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import expert_rollout  # noqa: E402
from expert_batch import build_jobs, parser, summarize  # noqa: E402
from physics_client import Physics  # noqa: E402

_phys, _log, _k = None, None, None


def _start(out: Path, slot):
    global _phys, _log, _k
    if _k is None:
        with slot.get_lock():
            _k = slot.value
            slot.value += 1
    (out / "workers").mkdir(exist_ok=True)
    _log = out / "workers" / f"w{_k}.physics.log"
    if _log.exists():
        _log.rename(_log.with_suffix(f".{int(time.time())}.log"))
    _phys = Physics(None, log=_log, nosmi=os.environ.get("JENGA_NOSMI", "1") != "0")
    mp.util.Finalize(_phys, _phys.close, exitpriority=10)


def _run(job_k: str, job):
    var, demo, out, extra, policy_ops, job_env = job
    if out.exists():
        r = json.loads(out.read_text())
        r["policy_ops"] = policy_ops
        return var, r
    t0 = time.perf_counter()
    args = expert_rollout.parser().parse_args([demo, str(out), "--k", job_k, *extra])
    env = {**{k: v for k, v in os.environ.items()
              if k.startswith("JENGA_") and k not in ("JENGA_PHYSICS_KEY", "JENGA_NOSMI")}, **job_env}
    with out.with_suffix(".log").open("w") as log:
        try:
            with h5py.File(args.demo, "r") as f:
                a = f["sim"].attrs
                tm, config = json.loads(a["target_mapping"]), json.loads(a["config_json"])
            header, state0 = _phys.episode(args.demo, args.offset, args.init, env)
            t1 = time.perf_counter()
            result, npz = expert_rollout.run(args, _phys, header, state0, tm, config)
            t2 = time.perf_counter()
            np.savez_compressed(out.with_suffix(".npz"), **npz)
            out.write_text(json.dumps(result, indent=1))
            t3 = time.perf_counter()
        except Exception:  # noqa: BLE001  the rollout counts as not run; the worker is replaced
            traceback.print_exc(file=log)
            _phys.close()
            _start(out.parent, None)
            return var, None
        print("EXPERT_DONE", json.dumps({k: v for k, v in result.items() if k != "segments"}), file=log)
        print("POOL_TIMING", json.dumps(dict(worker=_k, setup_s=t1 - t0, episode_s=result["wall_s"],
                                             plan_s=result["plan_s"], post_s=t2 - t1 - result["wall_s"],
                                             save_s=t3 - t2, total_s=t3 - t0)), file=log)
    result["policy_ops"] = policy_ops
    return var, result


def main():
    ap = parser()
    ap.add_argument("--workers", type=int, default=40, help="persistent Isaac workers (driver processes)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    ctx = mp.get_context("spawn")
    slot = ctx.Value("i", 0)
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=ctx, initializer=_start,
                             initargs=(args.out, slot)) as pool:
        results = list(pool.map(_run, [str(args.k)] * len(jobs), jobs))
    summarize(args, results)


if __name__ == "__main__":
    main()
