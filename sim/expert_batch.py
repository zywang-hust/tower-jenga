"""Run sim/expert_rollout.py over training-split episodes and tabulate per-variant success.

  .venv/bin/python sim/expert_batch.py <out_dir> [--variants task1-2 task1-3] [--start 0] [--episodes 5] [--jobs 6]
      [--perturb none|offset|grasp|kick|policy] [--seed 0] [--k 4] [--fleet <rollout dir>]

Perturbations draw one magnitude per episode, uniform 10-30 mm, in a random horizontal direction:
  offset  the movable assembly (dx, dy) plus up to +-3 deg yaw at reset
  wide    distillation layouts: offset magnitude uniform 0-40 mm, yaw +-5 deg; with --noise-max, each episode
          executes with DART noise of std uniform 0..noise-max rad (sim/expert_rollout.py --noise)
  d2      wide with offsets 0-25 mm (beyond, layer-1 blocks leave the base); tasks 1-4 also get irregular layers
          (JENGA_LAYER_JITTER, 8 mm / 5 deg, seeded per rollout) and task5-2 runs JENGA_PHYS_FIDELITY=1 (task5-3
          keeps the frozen tower: the expert fails its seat on the dynamic one)
  grasp   every grasp off the block centre: along the block's long axis, plus 15 mm deeper on half the episodes
  kick    near the goal a stand-in policy moves the holding hand off by the magnitude (and up to 10 mm in z)
  policy  no magnitude: start from a logged policy rollout (--fleet, sim/rollout.py npz) at the first tick the block of
          its first failed operation comes within 60 mm of its goal (else its last tick); --episodes counts rollouts
          (successful rollouts are skipped)
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jenga.paths import DATA_ROOT as DATA  # noqa: E402  env JENGA_DATA_ROOT


def perturb_args(kind: str, rng: np.random.Generator) -> list[str]:
    if kind == "none":
        return []
    m, th = rng.uniform(0.010, 0.030), rng.uniform(0, 2 * np.pi)
    if kind in ("wide", "d2"):
        m, psi = rng.uniform(0, 0.040 if kind == "wide" else 0.025), np.radians(rng.uniform(-5, 5))
        return ["--offset", f"{m * np.cos(th):.4f}", f"{m * np.sin(th):.4f}", f"{psi:.4f}"]
    if kind == "offset":
        return ["--offset", f"{m * np.cos(th):.4f}", f"{m * np.sin(th):.4f}", f"{np.radians(rng.uniform(-3, 3)):.4f}"]
    if kind == "grasp":
        deeper = 0.015 if rng.random() < 0.5 else 0.0
        return ["--grasp-offset", f"{m * rng.choice([-1, 1]):.4f}", "0", f"{deeper:.4f}"]
    if kind == "kick":
        return ["--kick", f"{m * np.cos(th):.4f}", f"{m * np.sin(th):.4f}", f"{rng.uniform(0, 0.010):.4f}"]
    raise ValueError(kind)


def policy_row(npz: Path, tm: list, ops_ok: list) -> int:
    """Init row of a policy rollout: where its first failed operation's block first gets within 60 mm of the goal."""
    blocks = np.load(npz)["blocks"]
    for m, ok in zip(tm, ops_ok):
        if not ok and "goal" in m:
            near = np.linalg.norm(blocks[:, m["block_index"], :3] - np.asarray(m["goal"][:3]), axis=1) < 0.06
            return int(np.argmax(near)) if near.any() else len(blocks) - 1
    return len(blocks) - 1


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--start", type=int, default=0, help="first training episode per variant")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--perturb", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--noise-max", type=float, default=0.0, help="DART noise std drawn per episode from U(0, max) rad")
    ap.add_argument("--noise-place", type=float, default=0.0, help="with --noise-max: DART std in the place segment")
    ap.add_argument("--noise-insert", type=float, default=0.0, help="with --noise-max: DART std before insertion")
    ap.add_argument("--repeats", type=int, default=1, help="rollouts per training episode (distinct layouts)")
    ap.add_argument("--fleet", type=Path, default=ROOT / "runs/simdev/fleet/r2-40000_v2", help="--perturb policy: rollouts")
    return ap


def build_jobs(args) -> list[tuple]:
    """(variant, demo, out json, sim/expert_rollout.py arguments, policy ops, JENGA_* env) per rollout, in a fixed
    order; the random draws depend only on the arguments and the job order (sim/expert_pool.py runs the same list)."""
    split = json.loads((ROOT / "runs/prep/split.json").read_text())
    rng = np.random.default_rng(args.seed)
    jobs = []
    if args.perturb == "policy":
        import h5py
        per = {}
        for rj in sorted(args.fleet.glob("task*.json")):
            r = json.loads(rj.read_text())
            var = r["variant"]
            if all(r["ops"]) or (args.variants and not any(var.startswith(v) for v in args.variants)) or \
                    len(per.setdefault(var, [])) >= args.episodes or not rj.with_suffix(".npz").exists():
                continue
            with h5py.File(r["demo"]) as f:
                tm = json.loads(f["sim"].attrs["target_mapping"])
            row = policy_row(rj.with_suffix(".npz"), tm, r["ops"])
            per[var].append(rj)
            jobs.append((var, r["demo"], args.out / rj.name, ["--init", str(rj.with_suffix(".npz")), str(row)],
                         r["ops"], {}))
    for var, sp in ([] if args.perturb == "policy" else split["tasks"].items()):
        if args.variants and not any(var.startswith(v) for v in args.variants):
            continue
        for rel in sp["train"][args.start:args.start + args.episodes]:
            ep = Path(rel).stem.split("_")[1]
            for rep in range(args.repeats):
                out = args.out / (f"{var}_{ep}.json" if args.repeats == 1 else f"{var}_{ep}_r{rep}.json")
                extra = perturb_args(args.perturb, rng)
                if args.noise_max > 0:
                    extra += ["--noise", f"{rng.uniform(0, args.noise_max):.4f}", str(int(rng.integers(1 << 30)))]
                    if args.noise_place > 0:
                        extra += ["--noise-place", f"{args.noise_place:.4f}"]
                    if args.noise_insert > 0:
                        extra += ["--noise-insert", f"{args.noise_insert:.4f}"]
                env = {}
                if args.perturb == "d2" and var[:5] in ("task1", "task2", "task3", "task4"):
                    env["JENGA_LAYER_JITTER"] = f"{int(rng.integers(1 << 30))},8,5"
                if args.perturb == "d2" and var.startswith("task5-2"):
                    env["JENGA_PHYS_FIDELITY"] = "1"
                jobs.append((var, str(DATA / rel), out, extra, None, env))
    return jobs


def main():
    args = parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)

    def run(job):
        var, demo, out, extra, policy_ops, env = job
        if not out.exists():
            cmd = [sys.executable, str(ROOT / "sim/expert_rollout.py"), demo, str(out), "--k", str(args.k), *extra]
            with out.with_suffix(".log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env})
        r = json.loads(out.read_text()) if out.exists() else None
        if r is not None:
            r["policy_ops"] = policy_ops
        return var, r

    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(run, jobs))
    summarize(args, results)


def summarize(args, results: list[tuple]) -> None:
    """Per-variant table of (variant, result json or None) -> <out>/summary.json, then EXPERT_BATCH_DONE."""
    table = {}
    for var, r in results:
        t = table.setdefault(var, dict(n=0, failed_to_run=0, tsr=0, ops=0, ops_ok=0, err_mm=[], done_s=[],
                                         policy_ok=0, fixed=0, broken=0))
        t["n"] += 1
        if r is None:
            t["failed_to_run"] += 1
            continue
        t["tsr"] += r["tsr"]
        t["ops"] += len(r["ops"])
        t["ops_ok"] += sum(r["ops"])
        t["err_mm"] += [e for e in r["final_err_mm"] if e is not None]
        t["done_s"].append(r["done_s"])
        if r["policy_ops"] is not None:  # operations the policy failed that the expert completed from its state
            t["policy_ok"] += sum(r["policy_ops"])
            t["fixed"] += sum(e and not p for e, p in zip(r["ops"], r["policy_ops"]))
            t["broken"] += sum(p and not e for e, p in zip(r["ops"], r["policy_ops"]))
    summary = {}
    for var, t in table.items():
        e = np.array(t["err_mm"]) if t["err_mm"] else np.array([np.nan])
        done = [d for d in t["done_s"] if d is not None]
        summary[var] = dict(episodes=t["n"], not_run=t["failed_to_run"], tsr=f"{int(t['tsr'])}/{t['n']}",
                            lhpr=round(t["ops_ok"] / max(t["ops"], 1), 3), err_mm_median=round(float(np.median(e)), 1),
                            err_mm_max=round(float(np.max(e)), 1),
                            done_s_median=round(float(np.median(done)), 1) if done else None, not_done=t["n"] - len(done))
        if args.perturb == "policy":
            summary[var].update(policy_ops_ok=f"{t['policy_ok']}/{t['ops']}", ops_fixed=t["fixed"],
                                ops_broken=t["broken"])
        print(var, summary[var])
    (args.out / "summary.json").write_text(json.dumps(dict(perturb=args.perturb, seed=args.seed, k=args.k,
                                                           variants=summary), indent=1))
    print("EXPERT_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
