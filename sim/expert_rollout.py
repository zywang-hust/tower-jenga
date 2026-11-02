"""Run sim/expert.py in closed loop against the local physics (sim/physics_worker.py: demo parameters, evaluation
controller), no rendering and no policy. Scores the end state with sim/score.py.

Perturbations (recovery tests):
  --offset DX DY DPSI     move the movable assembly at reset (worker option, as jenga.offset measures it)
  --grasp-offset X Y Z    grasp every block off its centre (block frame, m; +Z = deeper in the hand)
  --kick DX DY DZ         the first time a held block is within 60 mm of its goal, a stand-in for the policy moves the
                          holding arm's tool point by (DX, DY, DZ) m over 1 s and holds 0.5 s; then the expert resumes
  --init NPZ ROW          start from a logged state (qpos, blocks of a rollout npz; sim/physics_worker.py --init)
  --noise SIGMA [SEED]    DART: the arm joints of every executed row get smooth noise (Ornstein-Uhlenbeck, 1 s time
                          constant, stationary std SIGMA rad) while the expert is in a transit segment (TRANSIT),
                          faded in and out over ~0.3 s; fine segments (descend, place, release, push, insert-*) run
                          clean, since 10 mrad there already fails most pick-and-place episodes. The logged `actions`
                          stay the expert's clean commands
                          (absolute joint targets, so they pull the arm back), `executed` holds what was sent
  --noise-place SIGMA     with --noise: the same noise also runs, at std SIGMA rad, through the place segment (the
                          lowering onto the goal, before release), so the data holds recovery labels at the goal
  --noise-insert SIGMA    with --noise: likewise at std SIGMA rad through the task5 pre-insertion segments
                          (INSERT_NOISY: descent to the slot mouth and the first alignment; clean from slide-pre on)

Run (a venv that imports sim/rollout.py; the worker gets its own Isaac process, which sees no GPU):
  .venv/bin/python sim/expert_rollout.py <demo.hdf5> <out.json> [--seconds 210] [--k 4] [...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # before numpy: see sim/expert_dagger.py
    os.environ.setdefault(_v, "1")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from physics_client import Physics  # noqa: E402
from expert import ARMS, HZ, Expert  # noqa: E402
from score import reach, score  # noqa: E402

INSERT_NOISY = ("insert-descend", "insert-align")  # --noise-insert
TRANSIT = ("approach", "park", "park-up", "push-approach", "carry", "clear-up", "insert-carry", "unfold")


def kick_rows(ex: Expert, q16, arm: str, delta, start_cmd) -> np.ndarray:
    """1 s smooth move of `arm`'s tool point by delta (world), then 0.5 s hold: the off-target state a policy leaves."""
    k = ARMS.index(arm)
    s16 = start_cmd.copy()
    s16[8 * k:8 * k + 7] = q16[8 * k:8 * k + 7]
    R, _ = ex.eef(arm, s16[8 * k:8 * k + 7])
    tp = ex.tool(arm, s16[8 * k:8 * k + 7])
    q7, _ = ex.cart_path(arm, s16, [(tp + np.asarray(delta), R)], [np.linalg.norm(delta) + 1e-3], ramp_s=0.3)
    rows, _ = ex.rows_from(s16, arm, q7, None)
    return np.vstack([rows, np.repeat(rows[-1:], HZ // 2, 0)])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=210.0)
    ap.add_argument("--k", type=int, default=4, help="rows per expert call")
    ap.add_argument("--offset", type=float, nargs=3)
    ap.add_argument("--grasp-offset", type=float, nargs=3)
    ap.add_argument("--kick", type=float, nargs=3)
    ap.add_argument("--init", nargs=2, metavar=("NPZ", "ROW"))
    ap.add_argument("--noise", type=float, nargs="+", metavar=("SIGMA", "SEED"))
    ap.add_argument("--noise-place", type=float, default=0.0)
    ap.add_argument("--noise-insert", type=float, default=0.0)
    return ap


def main():
    args = parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.demo, "r") as f:
        a = f["sim"].attrs
        tm, config = json.loads(a["target_mapping"]), json.loads(a["config_json"])
    phys = Physics(args.demo, args.offset, args.out.with_suffix(".physics.log"), args.init)
    try:
        header, state0 = phys.connect()
        result, npz = run(args, phys, header, state0, tm, config)
    finally:
        phys.close()
    np.savez_compressed(args.out.with_suffix(".npz"), **npz)
    args.out.write_text(json.dumps(result, indent=1))
    print("EXPERT_DONE", json.dumps({k: v for k, v in result.items() if k != "segments"}), flush=True)


def run(args, phys, header, state0, tm, config):
    n_blocks = len(header["block_names"])
    setup = dict(initial_block_poses=np.array(header["initial_block_poses"]), block_fixed=np.array(header["block_fixed"]))
    ex = Expert.from_hdf5(args.demo)
    ex.initial = setup["initial_block_poses"]  # pushdown is judged against the (possibly offset) start
    if args.grasp_offset:
        ex.grasp_offset = np.array(args.grasp_offset)
    states = [state0]
    actions, executed, events = [], [], []
    sigma = args.noise[0] if args.noise else 0.0
    rng = np.random.default_rng(int(args.noise[1]) if args.noise and len(args.noise) > 1 else 0)
    ou, theta, gain = np.zeros(16), 1.0 / HZ, 0.0  # OU state on the arm joints; grippers never get noise
    arm_cols = np.r_[0:7, 8:15]
    steps, t0, kicked, plan_s = int(args.seconds * HZ), time.perf_counter(), False, 0.0
    done_at = None
    while len(actions) < steps:
        s = states[-1]
        q, blocks = s[:16], s[16:].reshape(n_blocks, 7)
        rows = None
        if args.kick is not None and not kicked:
            for op in tm:
                arm = ARMS[op["arm"]]
                if "goal" in op and ex.held(arm, q, blocks[op["block_index"]]) and \
                        np.linalg.norm(blocks[op["block_index"], :3] - op["goal"][:3]) < 0.06:
                    rows = kick_rows(ex, q, arm, args.kick, ex.cmd if ex.cmd is not None else q)
                    kicked = True
                    events.append(dict(tick=len(actions), event="kick", op=op["block_id"], delta=args.kick))
                    ex.segment, ex.cmd = None, rows[-1].copy()
                    break
        if rows is None:
            tp = time.perf_counter()
            rows = ex.act(q, blocks, args.k)
            plan_s += time.perf_counter() - tp
            if ex.log and (not events or events[-1].get("segment") != ex.log[-1] or events[-1]["tick"] < len(actions) - 1):
                if not events or events[-1].get("n_log") != len(ex.log):
                    events.append(dict(tick=len(actions), segment=ex.log[-1], n_log=len(ex.log)))
            if ex.log and ex.log[-1] == "done" and done_at is None:
                done_at = len(actions)
            if done_at is not None and len(actions) > done_at + 2 * HZ:
                break
        rows = rows[: steps - len(actions)]
        sent = rows.copy()
        if sigma > 0:
            seg = ex.log[-1].split(":")[0] if ex.log else ""
            transit = seg in TRANSIT and not ex.log[-1].startswith("park-open")
            target = 1.0 if transit else args.noise_place / sigma if seg == "place" else \
                args.noise_insert / sigma if seg in INSERT_NOISY else 0.0
            for i in range(len(sent)):
                ou[arm_cols] += -theta * ou[arm_cols] + sigma * np.sqrt(2 * theta) * rng.standard_normal(len(arm_cols))
                gain += (target - gain) / 10  # ~0.3 s fade
                sent[i] += gain * ou
        states += list(phys.step(np.ascontiguousarray(sent, dtype=np.float64)))
        actions += list(rows)
        executed += list(sent)

    states, actions, executed = np.array(states), np.array(actions), np.array(executed)
    blocks = states[:, 16:].reshape(len(states), n_blocks, 7)
    active = np.full(len(states), -1, np.int16)  # target block held by either arm (sim/active_block of the demos)
    for t, (q, b) in enumerate(zip(states[:, :16], blocks)):
        for op in tm:
            if ex.held(ARMS[op["arm"]], q, b[op["block_index"]]):
                active[t] = op["block_index"]
                break
    variant = Path(args.demo).parent.name
    result = dict(demo=args.demo, variant=variant, offset=args.offset, grasp_offset=args.grasp_offset, kick=args.kick,
                  init=args.init, noise=sigma, ticks=len(actions), done_s=None if done_at is None else done_at / HZ,
                  wall_s=time.perf_counter() - t0, plan_s=plan_s,
                  **score(setup, blocks[-1], tm, config), reach_m=reach(blocks, tm),
                  final_err_mm=[float(np.linalg.norm(blocks[-1, m["block_index"], :3] - m["goal"][:3]) * 1000)
                                if "goal" in m else None for m in tm],
                  segments=events)
    return result, dict(qpos=states[:, :16], blocks=blocks, actions=actions, executed=executed, active=active)


if __name__ == "__main__":
    main()
