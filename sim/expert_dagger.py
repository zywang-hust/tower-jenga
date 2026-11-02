"""DAgger data from closed-loop policy rollouts (scripts/fleet.sh: qpos and block poses every tick):
at each kept visited state, the expert's next HORIZON command rows (sim/expert.py, Expert.plan from that state alone),
with the four cameras rendered at that state.

Labels match the training target: TowerFrames (tail_hold, horizon 32) trains state t on leader/qpos[t:t+32], 30 Hz,
row 0 = the command sent at tick t. A policy state has no single future trajectory, so the chunk is stored per state
in dagger/actions [N, 32, 16]; reading it needs TowerFrames to take dagger/actions[t] instead of the leader window.
There is deliberately no leader/qpos: an unpatched reader fails instead of training on a window of first rows.

Kept states (per rollout, ticks 0..T-1):
  rules      no non-finite or |x| > 2 m block (explosion), no block moved > 0.5 m in xy from its start or below the table
             (thrown; demos move blocks <= 0.372 m): either cuts the rest of the rollout. Outside pushdown, no block
             resting > 5 deg off flat / side / upside down unless held (leaning). A rollout the policy solved ends
             END_PAD ticks after its first tick with every operation scored done.
  takeover   sim/expert_rollout.py --init from checkpoints every CHECK ticks of the rule-kept states and the last one;
             one bisection where a success is followed by a failure. Kept: rule-kept ticks between two successful
             checkpoints (tick 0, the reset state, counts as one: nominal fresh-draw takeover 70/70).
HDF5 per rollout (<out>/data/<variant>/<demo stem>_s<seed>.hdf5, written with sim/writer.py):
  observations/qpos [N, 16], observations/images/<camera> [N] JPEG q95 (dataset orientation, refinement --refine),
  dagger/actions [N, 32, 16], dagger/tick [N], dagger/segment [N], sim/action_valid [N] (all True),
  sim/block_poses [N, B, 7], sim/active_block [N] (target block held by either arm, else -1), sim/block_released
  [N, B] (False); attrs instruction, source (dataset-relative demo), dagger (JSON: policy rollout, checkpoints, refine).

Stages (each resumable: finished outputs are skipped):
  .venv/bin/python sim/expert_dagger.py select <fleet_dir> <work_dir> [--procs 16]
  CUDA_VISIBLE_DEVICES=7 runs/simdev/render_venv/bin/python sim/expert_dagger.py write <work_dir> <out_root> \
      --refine runs/simdev/refine/v2/refine.pt [--shard 0/2] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # before numpy: threaded BLAS on the expert's
    os.environ.setdefault(_v, "1")  # small batched inverses costs 6x the CPU and is 2x slower (62 vs 30 ms per label)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "src")]
from jenga.paths import DATA_ROOT as DATA  # noqa: E402  env JENGA_DATA_ROOT
HORIZON = 32
CHECK, END_PAD, TAKEOVER_S = 600, 150, 180.0
EXPLODE_M, THROWN_M, LEAN_DEG = 2.0, 0.5, 5.0


def tilt_dev(quat_xyzw: np.ndarray) -> np.ndarray:
    """[..., 4] -> degrees from the nearest resting tilt (flat, on its side, upside down)."""
    x, y = quat_xyzw[..., 0], quat_xyzw[..., 1]
    t = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
    return np.min(np.abs(t[..., None] - np.array([0.0, 90.0, 180.0])), -1)


def rules(ex, qpos, blocks, pushdown: bool) -> tuple[np.ndarray, dict]:
    """Rule mask over ticks 0..T-1 and the first tick of each cutting reason."""
    T = len(qpos) - 1
    q, b = qpos[:T], blocks[:T]
    keep, why = np.ones(T, bool), {}
    explode = ~np.isfinite(b).all((1, 2)) | ~np.isfinite(q).all(1) | (np.abs(b[..., :3]) > EXPLODE_M).any((1, 2))
    thrown = (np.linalg.norm(b[..., :2] - b[:1, :, :2], axis=-1) > THROWN_M).any(1) | (b[..., 2] < 0.0).any(1)
    for name, m in (("explode", explode), ("thrown", thrown)):
        if m.any():
            why[name] = int(np.argmax(m))
            keep[why[name]:] = False
    if not pushdown:
        lean = tilt_dev(b[..., 3:]) > LEAN_DEG  # [T, B]
        for t in np.flatnonzero(lean.any(1) & keep):
            if any(not any(ex.held(a, q[t], b[t, j]) for a in ("left", "back")) for j in np.flatnonzero(lean[t])):
                keep[t] = False
        why["lean_ticks"] = int((lean.any(1) & ~keep).sum())
    return keep, why


def solved_tick(demo_attrs, blocks) -> int | None:
    """First tick from which every operation scores done to the end (sim/score.py), or None."""
    from score import score
    tm, config = json.loads(demo_attrs["target_mapping"]), json.loads(demo_attrs["config_json"])
    setup = dict(initial_block_poses=blocks[0], block_fixed=np.array(json.loads(demo_attrs["block_fixed_on_table"])))
    T = len(blocks) - 1
    ok = np.array([score(setup, blocks[t], tm, config)["tsr"] for t in range(0, T, 15)]) > 0
    if not ok[-1]:
        return None
    bad = np.flatnonzero(~ok)
    return int(15 * (bad[-1] + 1)) if len(bad) else 0


def checkpoints(ticks: np.ndarray) -> list[int]:
    out, last = [], 0
    for t in ticks:
        if t >= last + CHECK:
            out.append(int(t))
            last = t
    if len(ticks) and (not out or out[-1] != ticks[-1]):
        out.append(int(ticks[-1]))
    return out


def takeovers(plan: dict, work: Path) -> dict:
    """Expert takeovers from the checkpoints of one rollout in one Isaac worker (sim/physics_worker.py --init,
    re-initialised per checkpoint), then one bisection after the first success-to-failure step. Cached in work/takeover/<name>.json.
    -> {tick: {ok, ops, ticks, wall_s}}; a worker error leaves {"error": ...} and no checkpoint passes."""
    import h5py
    import expert_rollout as er
    from rollout import Physics
    dst = work / "takeover" / f"{plan['name']}.json"
    if dst.exists():
        return {int(k) if k.isdigit() else k: v for k, v in json.loads(dst.read_text()).items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    cps, out = plan["checkpoints"], {}
    if cps:
        with h5py.File(plan["demo"], "r") as f:
            tm, config = json.loads(f["sim"].attrs["target_mapping"]), json.loads(f["sim"].attrs["config_json"])
        phys = Physics(plan["demo"], None, dst.with_suffix(".physics.log"), (plan["npz"], cps[0]))
        try:
            header, state0 = phys.connect()

            def run(c, state):
                a = argparse.Namespace(demo=plan["demo"], seconds=TAKEOVER_S, k=4, kick=None, grasp_offset=None,
                                       offset=None, init=[plan["npz"], c])
                r, _ = er.run(a, phys, header, state, tm, config)
                out[c] = dict(ok=bool(r["tsr"] > 0), ops=r["ops"], ticks=r["ticks"], wall_s=round(r["wall_s"], 1))

            for i, c in enumerate(cps):
                run(c, state0 if i == 0 else phys.reinit(plan["npz"], c))
            prev, prev_ok = 0, True
            for c in cps:
                if prev_ok and not out[c]["ok"]:
                    if c - prev > 2:
                        m = prev + (c - prev) // 2
                        run(m, phys.reinit(plan["npz"], m))
                    break
                prev, prev_ok = c, out[c]["ok"]
        except Exception as e:  # noqa: BLE001
            out = dict(error=repr(e))
        finally:
            phys.close()
    dst.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def plan_rollout(fleet: Path, work: Path, name: str) -> dict:
    """Rules, end trim and checkpoints of one rollout -> work/plan/<name>.json."""
    import h5py
    from expert import Expert
    dst = work / "plan" / f"{name}.json"
    if dst.exists():
        return json.loads(dst.read_text())
    res = json.loads((fleet / f"{name}.json").read_text())
    log = np.load(fleet / f"{name}.npz")
    qpos, blocks = log["qpos"], log["blocks"]
    ex = Expert.from_hdf5(res["demo"])
    with h5py.File(res["demo"], "r") as f:
        attrs = dict(f["sim"].attrs)
    keep, why = rules(ex, qpos, blocks, ex.task.startswith("full_tower_pushdown"))
    end = len(keep)
    if res.get("tsr", 0) > 0:
        s = solved_tick(attrs, blocks)
        if s is not None:
            end = min(end, s + END_PAD)
            why["solved"] = s
    keep[end:] = False
    ticks = np.flatnonzero(keep)
    plan = dict(name=name, demo=res["demo"], npz=str((fleet / f"{name}.npz").resolve()), T=len(qpos) - 1,
                policy_tsr=res.get("tsr"), policy_ops=res.get("ops"), rule_kept=len(ticks), why=why,
                checkpoints=checkpoints(ticks))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(plan))
    return plan


def keep_mask(plan: dict, ok: dict) -> np.ndarray:
    """Rule-kept ticks lying between two successful checkpoints (tick 0 counts as one)."""
    log = np.load(plan["npz"])
    from expert import Expert
    ex = Expert.from_hdf5(plan["demo"])
    keep, _ = rules(ex, log["qpos"], log["blocks"], ex.task.startswith("full_tower_pushdown"))
    keep[np.arange(len(keep)) >= (plan["why"].get("solved", 10 ** 9) + END_PAD)] = False
    cps = sorted(ok)
    good = np.zeros(len(keep), bool)
    prev_t, prev_ok = 0, True
    for c in cps:
        if prev_ok and ok[c]:
            good[prev_t:c + 1] = True
        prev_t, prev_ok = c, ok[c]
    return keep & good


def label_rollout(job) -> dict:
    """Labels at the kept ticks -> work/sel/<name>.npz."""
    plan, ok, work = job
    from expert import ARMS, Expert
    dst = Path(work) / "sel" / f"{plan['name']}.npz"
    if dst.exists():
        return json.loads(dst.with_suffix(".json").read_text())
    t0 = time.perf_counter()
    keep = keep_mask(plan, ok)
    log = np.load(plan["npz"])
    qpos, blocks = log["qpos"], log["blocks"]
    ex = Expert.from_hdf5(plan["demo"])
    ex.initial = blocks[0]
    ticks = np.flatnonzero(keep)
    labels = np.zeros((len(ticks), HORIZON, 16), np.float32)
    segment, active = [], np.full(len(ticks), -1, np.int16)
    for i, t in enumerate(ticks):
        q, b = qpos[t].astype(float), blocks[t].astype(float)
        seg = ex.plan(q, b, fresh=True)
        r = seg.rows[:HORIZON]
        labels[i] = np.vstack([r, np.repeat(r[-1:], HORIZON - len(r), 0)]) if len(r) < HORIZON else r
        segment.append(seg.name)
        for op in ex.ops:
            if ex.held(ARMS[op["arm"]], q, b[op["block_index"]]):
                active[i] = op["block_index"]
                break
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, ticks=ticks.astype(np.int32), labels=labels, segment=np.array(segment), active=active)
    summary = dict(name=plan["name"], variant=Path(plan["demo"]).parent.name, T=plan["T"], rule_kept=plan["rule_kept"],
                   kept=int(len(ticks)), checkpoints={str(k): bool(v) for k, v in sorted(ok.items())},
                   policy_tsr=plan["policy_tsr"], why=plan["why"], label_s=round(time.perf_counter() - t0, 1))
    dst.with_suffix(".json").write_text(json.dumps(summary))
    tmp.rename(dst)
    return summary


def process(job) -> dict:
    fleet, work, name = job
    try:
        plan = plan_rollout(fleet, work, name)
        res = takeovers(plan, work)
        ok = {} if "error" in res else {c: v["ok"] for c, v in res.items()}
        if "error" in res:
            plan = {**plan, "why": {**plan["why"], "takeover_error": res["error"]}}
        return label_rollout((plan, ok, str(work)))
    except Exception as e:  # noqa: BLE001  report and go on; the rollout stays undone for the next run
        return dict(name=name, error=repr(e))


def select(args):
    fleet, work = args.fleet.resolve(), args.work.resolve()
    names = sorted(p.stem for p in fleet.glob("*.json") if (fleet / f"{p.stem}.npz").exists())
    names = [n for n in names if not (work / "sel" / f"{n}.npz").exists()][: args.limit or None]
    print(f"select: {len(names)} rollouts to do with {args.procs} processes", flush=True)
    t0, states = time.perf_counter(), 0
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for i, s in enumerate(pool.imap_unordered(process, [(fleet, work, n) for n in names])):
            states += s.get("kept", 0)
            keys = ("name", "T", "rule_kept", "kept", "policy_tsr", "label_s", "error")
            print(json.dumps({**{k: s[k] for k in keys if k in s}, "done": i + 1, "of": len(names),
                              "states_per_min": round(states / (time.perf_counter() - t0) * 60)}), flush=True)
    print(f"SELECT_DONE {len(names)} rollouts {states} states {time.perf_counter() - t0:.0f} s", flush=True)


# ---------- write (renderer venv)
def write_one(renderer, fk, work: Path, out_root: Path, name: str, refine: str) -> dict:
    import h5py
    from writer import atomic, block_names, episode_dims, frame_poses, write_images
    sel = np.load(work / "sel" / f"{name}.npz")
    plan = json.loads((work / "plan" / f"{name}.json").read_text())
    summary = json.loads((work / "sel" / f"{name}.json").read_text())
    demo = Path(plan["demo"])
    rel = demo.relative_to(DATA)
    dst = out_root / rel.parent / f"{demo.stem}_s{name.rsplit('_s', 1)[1]}.hdf5"
    if dst.exists():
        return dict(name=name, frames=0, skipped=True)
    ticks = sel["ticks"]
    if not len(ticks):
        return dict(name=name, frames=0, empty=True)
    t0 = time.perf_counter()
    log = np.load(plan["npz"])
    qpos, blocks = log["qpos"][ticks].astype(np.float32), log["blocks"][ticks]
    with h5py.File(demo, "r") as f:
        instruction = f.attrs["instruction"]
        names = block_names(tuple(json.loads(f["sim"].attrs["block_fixed_on_table"])))
        dims = episode_dims(f["sim"].attrs, names)
        jpeg_dtype = f["observations/images/left_back_camera"].dtype
    N = len(ticks)
    with atomic(dst) as tmp, h5py.File(tmp, "w") as g:
        g.attrs["instruction"] = instruction
        g.attrs["source"] = str(rel)
        g.attrs["dagger"] = json.dumps(dict(
            rollout=plan["npz"], policy_tsr=plan["policy_tsr"], policy_ops=plan["policy_ops"], horizon=HORIZON,
            checkpoints=summary["checkpoints"], rules=plan["why"], refine=refine, check=CHECK, end_pad=END_PAD,
            takeover_s=TAKEOVER_S, lean_deg=LEAN_DEG, thrown_m=THROWN_M))
        g["observations/qpos"] = qpos
        g["dagger/actions"] = sel["labels"]
        g["dagger/tick"] = ticks
        g["dagger/segment"] = sel["segment"].astype("S")
        g["sim/action_valid"] = np.ones(N, bool)
        g["sim/block_poses"] = blocks.astype(np.float32)
        g["sim/active_block"] = sel["active"]
        g["sim/block_released"] = np.zeros((N, blocks.shape[1]), bool)
        t_render = write_images(g, renderer, N, lambda t: frame_poses(
            renderer, fk.link_poses(qpos[t].astype(np.float64)), blocks[t], names, {}, dims), jpeg_dtype)
    wall = time.perf_counter() - t0
    return dict(name=name, frames=N, wall_s=round(wall, 1), render_s=round(t_render, 1), fps=round(N / wall, 1))


def write_setup(work: str, out_root: str, refine: str):
    from fk import ArmFK
    from render import STATION, Renderer
    renderer, fk = Renderer(STATION, refine=refine), ArmFK(STATION)
    renderer.render()  # compile the shading outside the timings
    return lambda name: write_one(renderer, fk, Path(work), Path(out_root), name, refine)


def write(args):
    from writer import run_shard
    work, out_root = args.work.resolve(), args.out_root.resolve()  # files go to data/<variant>/ as in the demo path
    refine = str(Path(args.refine).resolve())
    assert Path(refine).exists(), refine
    k, n = map(int, args.shard.split("/"))
    names = sorted(p.stem for p in (work / "sel").glob("*.npz") if ".tmp" not in p.name)
    mine = [x for x in names if zlib.crc32(x.encode()) % n == k][: args.limit or None]
    print(f"write shard {k}/{n}: {len(mine)} rollouts, refine {refine}", flush=True)
    t0 = time.perf_counter()
    frames = run_shard(mine, args.workers, write_setup, (str(work), str(out_root), refine))
    print(f"WRITE_DONE shard {k}/{n}: {len(mine)} rollouts, {frames} frames, {time.perf_counter() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("fleet", type=Path)
    s.add_argument("work", type=Path)
    s.add_argument("--procs", type=int, default=16)
    s.add_argument("--limit", type=int, default=0)
    w = sub.add_parser("write")
    w.add_argument("work", type=Path)
    w.add_argument("out_root", type=Path)
    w.add_argument("--refine", required=True)
    w.add_argument("--shard", default="0/1")
    w.add_argument("--workers", type=int, default=3)
    w.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    select(args) if args.cmd == "select" else write(args)


if __name__ == "__main__":
    main()
