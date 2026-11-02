"""Fit the station geometry used by jenga.kinematics / jenga.offset from the local TOWER release.

Writes
  configs/nero_kinematics.json   per arm: FK chain (8x6), arm-base -> world transform, tool point, joint range
  configs/cameras.json           section "layouts": nominal frame-0 block poses per task variant
                                 (the camera section of that file is static and kept as is)

Question each fit answers:
  FK      does a generic 7-revolute chain reproduce observations/eef from observations/qpos (held-out error)?
  base    which rigid transform maps eef (arm-base frame) to the world frame of sim/block_poses?  Evidence:
          the block held during 'lift' (centre = base * (p_eef + R_eef * tool)), all training episodes.
  layouts mean frame-0 block poses per variant; the template of the offset estimator.
  photometric  per variant and external camera, the mean per-channel grey-level quantiles of frame 0
          (layouts[variant]["photometric_reference"]): the nominal tone the estimator maps each image back to
          before the colour mask (offset.normalise), so gain, white balance and gamma changes keep the wood hue.
  bias    systematic difference (estimate - truth) of jenga.offset on frame 0 of training episodes, per
          variant, stored as layouts[variant]["estimator_bias"] and subtracted by the estimator.
  gate    how large must a (bias-corrected) estimate be before the scene is distinguishable from a training layout?  Raw estimates
          on nominal frame 0 and on pixel-level synthetic rule-box shifts (offset.synthesize_shift) of the
          training episodes (gate-sample, rows in runs/spatial/gate/), then per-variant thresholds stored as
          layouts[variant]["offset_gate"] and applied by jenga.warp.gate_offset (fit_gate).

    python scripts/fit_kinematics.py [--workers 16] [--stage all|photometric|bias|gate-sample|gate]
  (all = FK, base, layouts, photometric, bias; after a new bias run gate-sample then gate)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import astuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from jenga.kinematics import Chain, quat_xyzw_to_matrix
from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT  # env JENGA_DATA_ROOT
OUT_KIN = ROOT / "configs/nero_kinematics.json"
OUT_CAM = ROOT / "configs/cameras.json"
GATE_ROWS = ROOT / "runs/spatial/gate/train_rows.json"
SPLIT = ROOT / "runs/prep/split.json"
RULE_D, RULE_PSI = 0.015, np.radians(10)  # robustness release: movable assembly up to 15 mm per axis, 10 deg yaw


def first_frame(f) -> dict:
    """Frame 0 of the external cameras, RGB."""
    from jenga.offset import EXTERNAL
    return {c: cv2.cvtColor(cv2.imdecode(np.frombuffer(f["observations/images/" + c][0], np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB) for c in EXTERNAL}


def compose(t0, d1) -> np.ndarray:
    """Offset (about the same centre) of applying d1 = (dx, dy, dpsi) after the Offset t0."""
    c, s = np.cos(d1[2]), np.sin(d1[2])
    return np.array([c * t0.dx - s * t0.dy + d1[0], s * t0.dx + c * t0.dy + d1[1], t0.dpsi + d1[2]])


def first_per_variant(train: list[str], per_variant: int) -> list[str]:
    """The first `per_variant` training episodes of every variant, variants in sorted order."""
    by_var = defaultdict(list)
    for e in train:
        by_var[Path(e).parent.name].append(e)
    return [e for v in sorted(by_var) for e in by_var[v][:per_variant]]


def phase_segments(f) -> list[tuple[int, int, str, int]]:
    ph = f["sim/task_phase"][:]
    ab = f["sim/active_block"][:]
    names = json.loads(f["sim"].attrs["phase_names"])
    ch = np.r_[0, np.flatnonzero(np.diff(ph) | np.diff(ab)) + 1, len(ph)]
    return [(int(ch[i]), int(ch[i + 1]), names[ph[ch[i]]], int(ab[ch[i]])) for i in range(len(ch) - 1)]


def read_episode(rel: str) -> dict:
    """Everything the fits need from one episode (runs in a worker process)."""
    with h5py.File(DATA / rel, "r") as f:
        q = f["observations/qpos"][:]
        e = f["observations/eef"][:]
        lq = f["leader/qpos"][:]
        ids = json.loads(f["sim"].attrs["block_ids"])
        tm = {m["block_index"]: m["arm"] for m in json.loads(f["sim"].attrs["target_mapping"])}
        held = []
        B = f["sim/block_poses"]
        for s, end, name, b in phase_segments(f):
            if name == "lift" and b in tm:
                for k in range(s + 5, min(end, s + 30), 5):  # block lifted clear, gripper closed
                    held.append((tm[b], ids[b], e[k, tm[b] * 7:tm[b] * 7 + 7], B[k, b]))
        fk_rows = np.arange(0, len(q), 25)
        return dict(rel=rel, variant=Path(rel).parent.name, q=q[fk_rows], eef=e[fk_rows], held=held,
                    qmin=np.minimum(q.min(0), lq.min(0)), qmax=np.maximum(q.max(0), lq.max(0)),
                    ids=ids, pose0=B[0])


def fk_residual(params, q, R_m, p_m, w_rot=0.1):
    R, p = Chain(params).fk(q)
    return np.concatenate([(p - p_m).ravel(), w_rot * (R - R_m).ravel()])


def fit_chain(q, R_m, p_m, restarts=3, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(q), min(600, len(q)), replace=False)
    best = None
    for _ in range(restarts):
        p0 = np.zeros((8, 6))
        p0[:, 2] = 0.08
        p0[:, 3:] = rng.normal(0, 1.0, (8, 3))
        sol = least_squares(fk_residual, p0.ravel(), args=(q[sub], R_m[sub], p_m[sub]), method="lm", max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol
    return least_squares(fk_residual, best.x, args=(q, R_m, p_m), method="lm", max_nfev=4000).x


def fk_errors(params, q, R_m, p_m):
    R, p = Chain(params).fk(q)
    ang = np.linalg.norm(Rotation.from_matrix(np.swapaxes(R, 1, 2) @ R_m).as_rotvec(), axis=1)
    return np.linalg.norm(p - p_m, axis=1), ang


def fit_base(held, x0_rot, loss="linear"):
    """block_world = R_wb (p_e + R_e tool) + t; unknowns R_wb (rotvec), t, tool (9)."""
    p_e = np.array([h[2][:3] for h in held], float)
    R_e = quat_xyzw_to_matrix(np.array([h[2][3:] for h in held], float))
    b_w = np.array([h[3][:3] for h in held], float)

    def res(x):
        Rw = Rotation.from_rotvec(x[:3]).as_matrix()
        return ((p_e + R_e @ x[6:9]) @ Rw.T + x[3:6] - b_w).ravel()

    x0 = np.r_[x0_rot, b_w.mean(0) - Rotation.from_rotvec(x0_rot).apply(p_e.mean(0)), 0, 0, 0]
    sol = least_squares(res, x0, loss=loss, f_scale=0.002)
    r = np.linalg.norm(res(sol.x).reshape(-1, 3), axis=1)
    return sol.x, r


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


_EST = None


def bias_sample(rel: str):
    """(variant, estimate - truth [3]) on frame 0 of one episode, bias correction disabled."""
    global _EST
    from jenga.offset import OffsetEstimator, true_offset
    if _EST is None:
        _EST = OffsetEstimator(OUT_CAM, correct_bias=False)
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        imgs = first_frame(f)
        ids = json.loads(f["sim"].attrs["block_ids"])
        pose0 = f["sim/block_poses"][0]
    r = _EST.estimate(v, imgs)
    t = true_offset(_EST.layouts[v]["blocks"], ids, pose0)
    return v, np.array([r.offset.dx - t.dx, r.offset.dy - t.dy, r.offset.dpsi - t.dpsi]), r.confidence


def calibrate_bias(train: list[str], workers: int, per_variant: int = 32):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(bias_sample, first_per_variant(train, per_variant), chunksize=2))
    cam = json.loads(OUT_CAM.read_text())
    for v in sorted({r[0] for r in res}):
        D = np.array([r[1] for r in res if r[0] == v])
        bias = np.median(D, 0)
        rem = D - bias
        cam["layouts"]["variants"][v]["estimator_bias"] = dict(
            dx=float(bias[0]), dy=float(bias[1]), dpsi=float(bias[2]), episodes=len(D),
            residual_xy_mm_rms=float(1e3 * np.sqrt((rem[:, :2] ** 2).sum(1).mean())),
            residual_psi_deg_rms=float(np.degrees(np.sqrt((rem[:, 2] ** 2).mean()))))
        print(v, "estimator bias", json.dumps(cam["layouts"]["variants"][v]["estimator_bias"]), flush=True)
    OUT_CAM.write_text(json.dumps(cam, indent=1))


def photometric_sample(rel: str):
    from jenga.offset import EXTERNAL, channel_quantiles
    levels = json.loads(OUT_CAM.read_text())["estimator"]["photometric_levels"]
    with h5py.File(DATA / rel, "r") as f:
        return Path(rel).parent.name, {c: channel_quantiles(cv2.imdecode(np.frombuffer(f["observations/images/" + c][0], np.uint8),
                                                                         cv2.IMREAD_COLOR)[..., ::-1], levels)
                                       for c in EXTERNAL}


def calibrate_photometric(train: list[str], workers: int, per_variant: int = 32):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(photometric_sample, first_per_variant(train, per_variant), chunksize=2))
    cam = json.loads(OUT_CAM.read_text())
    for v in sorted({name for name, _ in res}):
        qs = [q for name, q in res if name == v]
        cam["layouts"]["variants"][v]["photometric_reference"] = {
            c: np.round(np.mean([q[c] for q in qs], 0), 2).tolist() for c in qs[0]}
        print(v, "photometric reference from", len(qs), "frames", flush=True)
    OUT_CAM.write_text(json.dumps(cam, indent=1))


# ---------------------------------------------------------------- offset gate
def gate_sample(args):
    """One frame-0 estimate (bias-corrected, ungated) of episode `rel`; k < 0: nominal frame, else a pixel-level
    synthetic shift (k < 4: corner of the rule box, else uniform in it). Returns truth and raw estimate."""
    rel, k, seed = args
    global _EST
    from jenga.offset import EXTERNAL, OffsetEstimator, is_movable, synthesize_shift, true_offset
    from jenga.warp import Offset
    if _EST is None:
        cv2.setNumThreads(1)
        _EST = OffsetEstimator(OUT_CAM)
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        imgs = first_frame(f)
        ids = json.loads(f["sim"].attrs["block_ids"])
        P = f["sim/block_poses"][0].astype(float)
    t0 = true_offset(_EST.layouts[v]["blocks"], ids, P)
    truth, d1 = np.array([t0.dx, t0.dy, t0.dpsi]), np.zeros(3)
    if k >= 0:
        rng = np.random.default_rng(seed)
        if k < 4:
            d1 = np.array([RULE_D * (1 if k & 1 else -1), RULE_D * (1 if k & 2 else -1), RULE_PSI * rng.choice([-1, 1])])
        else:
            d1 = rng.uniform([-RULE_D, -RULE_D, -RULE_PSI], [RULE_D, RULE_D, RULE_PSI])
        _, _, c = _EST.template(v)
        mv = np.array([is_movable(b) for b in ids])
        imgs = {cam: synthesize_shift(imgs[cam], _EST.cams[cam], P, mv, c, Offset(*d1), _EST.dims) for cam in EXTERNAL}
        truth = compose(t0, d1)
    r = _EST.estimate(v, imgs, yaw=True)
    rt = _EST.estimate(v, imgs, yaw=False)
    return dict(variant=v, rel=rel, k=k, shift=d1.tolist(), truth=truth.tolist(),
                raw=[r.offset.dx, r.offset.dy, r.offset.dpsi], confidence=r.confidence, ok=r.ok,
                raw_translation_only=[rt.offset.dx, rt.offset.dy, 0.0], ok_translation_only=rt.ok)


def sample_gate(eps: list[str], workers: int, shifts_per_episode: int, out: Path):
    """Raw estimates on nominal frame 0 of `eps` and on `shifts_per_episode` synthetic shifts of each."""
    jobs = [(rel, k, 1000 * i + k) for i, rel in enumerate(eps) for k in range(-1, shifts_per_episode)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(gate_sample, jobs, chunksize=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=0))
    return rows


def q3(x) -> dict:
    return dict(mean=round(float(np.mean(x)), 3), p95=round(float(np.percentile(x, 95)), 3), max=round(float(np.max(x)), 3))


def fit_gate(rows_path: Path = GATE_ROWS):
    """Per-variant gate of the raw offset estimate (jenga.warp.gate_offset) from training-episode rows.

    translation_mm  largest nominal estimator error |d_est - d_true| on frame 0: a nominal scene whose estimate
                    exceeds it is (partly) moved toward its true layout jitter, which the edge-term estimator
                    measures to ~1-3 mm. On the held-out val rows this moves 22/56 nominal scenes and lowers their
                    mean footprint error from 2.02 to 1.19 mm with none ending > 0.5 mm worse than the identity
                    (also across the training lighting augmentation range, gamma and shading ramps after the
                    photometric normalisation; runs/spatial/validate/l.json). Gating on the largest |d_est| instead (jitter plus
                    error) leaves the jitter uncorrected (2 of 56 moved, mean 2.00 mm).
    yaw_deg         largest |dpsi_est| on nominal frame 0; every demonstration starts at yaw 0, so this is pure
                    estimator error and a nominal scene is never rotated
    yaw             keep the yaw search only if it lowers the p95 block-footprint error on the synthetic
                    rule-box shifts against the translation-only estimate (with its own nominal threshold)"""
    from jenga.offset import OffsetEstimator
    from jenga.warp import Offset, gate_offset
    est = OffsetEstimator(OUT_CAM)
    rows = json.loads(rows_path.read_text())
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    cam = json.loads(OUT_CAM.read_text())
    for v, rs in sorted(by_var.items()):
        nom = [r for r in rs if r["k"] < 0]
        shift = [r for r in rs if r["k"] >= 0]
        T = np.array([r["truth"] for r in nom])
        mag_d = {m: 1e3 * np.hypot(*np.array([r[key] for r in nom])[:, :2].T)
                 for m, key in (("full", "raw"), ("translation_only", "raw_translation_only"))}
        err_d = {m: 1e3 * np.hypot(*(np.array([r[key] for r in nom]) - T)[:, :2].T)
                 for m, key in (("full", "raw"), ("translation_only", "raw_translation_only"))}
        psi = np.degrees(np.abs([r["raw"][2] for r in nom]))
        gates = {"full": dict(translation_mm=float(err_d["full"].max()), yaw_deg=float(psi.max()), yaw=True),
                 "translation_only": dict(translation_mm=float(err_d["translation_only"].max()), yaw_deg=0.0, yaw=False)}
        St = np.array([r["truth"] for r in shift])
        fp = {}
        for m, key in (("full", "raw"), ("translation_only", "raw_translation_only")):
            A = np.array([astuple(gate_offset(Offset(*r[key]), gates[m])) for r in shift])
            fp[m] = est.footprint_error_mm(v, A, St)
        none = est.footprint_error_mm(v, np.zeros_like(St), St)
        mode = "full" if np.percentile(fp["full"], 95) <= np.percentile(fp["translation_only"], 95) else "translation_only"
        cam["layouts"]["variants"][v]["offset_gate"] = dict(
            **gates[mode],
            provenance=dict(
                generated_by="scripts/fit_kinematics.py --stage gate", rows=str(rows_path.relative_to(ROOT)),
                rule="translation_mm = max nominal |d_est - d_true|, yaw_deg = max nominal |dpsi_est| (truth 0), "
                     "train split frame 0, firm threshold [tau, 2 tau]; yaw kept iff it lowers the p95 footprint error "
                     "on synthetic rule-box shifts",
                nominal_episodes=len(nom), nominal_true_yaw_max_deg=float(np.degrees(np.abs(T[:, 2]).max())),
                nominal_true_translation_mm=q3(1e3 * np.hypot(*T[:, :2].T)),
                nominal_translation_estimate_mm=q3(mag_d[mode]), nominal_translation_error_mm=q3(err_d[mode]),
                nominal_yaw_abs_deg=q3(psi),
                shift_samples=len(shift), shift_footprint_error_mm={k: q3(x) for k, x in fp.items()} | {"none": q3(none)}))
        print(v, mode, json.dumps(gates[mode]), "footprint p95 full/translation_only/none",
              [round(float(np.percentile(x, 95)), 2) for x in (fp["full"], fp["translation_only"], none)], flush=True)
    OUT_CAM.write_text(json.dumps(cam, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stage", choices=("all", "photometric", "bias", "gate-sample", "gate"), default="all")
    args = ap.parse_args()
    t0 = time.time()
    split = json.loads(SPLIT.read_text())
    train = [e for t in split["tasks"].values() for e in t["train"]]
    val = [e for t in split["tasks"].values() for e in t["val"]]
    if args.stage == "gate":
        fit_gate()
        print(f"wrote offset_gate in {OUT_CAM.relative_to(ROOT)}")
        return
    if args.stage == "gate-sample":
        for eps, n, out in ((train, 8, GATE_ROWS), (val, 6, GATE_ROWS.with_name("val_rows.json"))):
            rows = sample_gate(eps, args.workers, n, out)
            print(f"wrote {len(rows)} rows to {out.relative_to(ROOT)} in {time.time() - t0:.0f}s", flush=True)
        print("GATE_SAMPLE_DONE")
        return
    if args.stage == "photometric":
        calibrate_photometric(train, args.workers)
        return
    if args.stage == "bias":
        calibrate_bias(train, args.workers)
        print(f"wrote estimator_bias in {OUT_CAM.relative_to(ROOT)} in {time.time() - t0:.0f}s")
        return
    with ProcessPoolExecutor(args.workers) as ex:
        tr = list(ex.map(read_episode, train, chunksize=4))
        va = list(ex.map(read_episode, val, chunksize=4))
    print(f"read {len(tr)} train / {len(va)} val episodes in {time.time() - t0:.0f}s", flush=True)

    report = {"split": str(SPLIT.relative_to(ROOT)), "train_episodes": len(tr), "val_episodes": len(va)}
    arms_out = {}
    # x0 for the base rotations: left base is yawed ~180 deg, back ~-90 deg (sign found by Kabsch below)
    for a, name in enumerate(("left", "back")):
        sl, se = slice(a * 8, a * 8 + 7), slice(a * 7, a * 7 + 7)
        q = np.concatenate([d["q"][:, sl] for d in tr]).astype(float)
        E = np.concatenate([d["eef"][:, se] for d in tr]).astype(float)
        params = fit_chain(q, quat_xyzw_to_matrix(E[:, 3:]), E[:, :3])
        qv = np.concatenate([d["q"][:, sl] for d in va]).astype(float)
        Ev = np.concatenate([d["eef"][:, se] for d in va]).astype(float)
        ep, er = fk_errors(params, qv, quat_xyzw_to_matrix(Ev[:, 3:]), Ev[:, :3])
        fk_rep = dict(val_samples=len(qv), pos_err_mm_mean=1e3 * ep.mean(), pos_err_mm_max=1e3 * ep.max(),
                      rot_err_deg_mean=np.degrees(er.mean()), rot_err_deg_max=np.degrees(er.max()))
        print(name, "FK", fk_rep, flush=True)

        held = [(d["variant"], *h[1:]) for d in tr for h in d["held"] if h[0] == a]
        P_e = np.array([h[2][:3] for h in held], float)
        B_w = np.array([h[3][:3] for h in held], float)
        Rk, tk = kabsch(P_e, B_w)
        rk = np.linalg.norm(P_e @ Rk.T + tk - B_w, axis=1)
        x, r = fit_base(held, Rotation.from_matrix(Rk).as_rotvec())
        xr, rr = fit_base(held, x[:3], loss="soft_l1")
        by_variant = defaultdict(list)
        by_block = defaultdict(list)
        for h, v in zip(held, rr):
            by_variant[h[0]].append(v)
            by_block[(h[0][:7], h[1])].append(v)
        R_wb = Rotation.from_rotvec(xr[:3])
        base_rep = dict(
            samples=len(held),
            kabsch_position_only_mm=dict(mean=1e3 * rk.mean(), p95=1e3 * np.percentile(rk, 95)),
            rigid_plus_tool_mm=dict(mean=1e3 * r.mean(), p95=1e3 * np.percentile(r, 95)),
            rigid_plus_tool_robust_mm=dict(mean=1e3 * rr.mean(), p50=1e3 * np.median(rr), p95=1e3 * np.percentile(rr, 95)),
            residual_by_variant_mm={k: round(1e3 * float(np.mean(v)), 3) for k, v in sorted(by_variant.items())},
            worst_blocks_mm=dict(sorted(((f"{k[0]}:{k[1]}", round(1e3 * float(np.mean(v)), 3))
                                         for k, v in by_block.items()), key=lambda kv: -kv[1])[:6]),
            base_yaw_deg=float(np.degrees(R_wb.as_euler("zyx")[0])),
            base_tilt_deg=[float(v) for v in np.degrees(R_wb.as_euler("zyx")[1:])],
        )
        print(name, "base", json.dumps(base_rep), flush=True)
        qlo = np.min([d["qmin"][sl] for d in tr], 0)
        qhi = np.max([d["qmax"][sl] for d in tr], 0)
        arms_out[name] = dict(
            chain_params=params.reshape(8, 6).tolist(),
            base_to_world=dict(R=R_wb.as_matrix().tolist(), t=xr[3:6].tolist()),
            tool_point_in_eef=xr[6:9].tolist(),
            observed_joint_range=dict(lo=qlo.tolist(), hi=qhi.tolist(), note="observations+leader qpos, train split; "
                                      "real joint limits are unknown (no URDF)"),
            fit=dict(fk=fk_rep, base=base_rep),
        )

    kin = dict(
        schema="jenga-nero-kinematics-v1",
        generated_by="scripts/fit_kinematics.py",
        frames=dict(eef="observations/eef frame (xyz + quaternion xyzw), arm-base frame",
                    world="sim/block_poses frame (pose_frame=environment_world_meters)",
                    chain="per joint i<7: F_i then Rz(q_i); F_7 = eef. F = (t xyz [m], rotation vector [rad])",
                    tool_point="centre of a block held during 'lift', in the eef frame"),
        robot="AgileX Nero (provenance prim NeroRobotWithWristCamera; TOWER-Assets THIRD_PARTY_NOTICES)",
        **report, arms=arms_out)
    OUT_KIN.write_text(json.dumps(kin, indent=1))

    # nominal layouts: mean frame-0 pose per block id and variant
    layouts = {}
    by_var = defaultdict(list)
    for d in tr:
        by_var[d["variant"]].append(d)
    for v, ds in sorted(by_var.items()):
        ids = ds[0]["ids"]
        P = np.mean([d["pose0"] for d in ds if d["ids"] == ids], 0)
        q = P[:, 3:] / np.linalg.norm(P[:, 3:], axis=1, keepdims=True)
        layouts[v] = dict(episodes=len(ds), blocks={k: [*map(float, P[i, :3]), *map(float, q[i])] for i, k in enumerate(ids)})
    cam = json.loads(OUT_CAM.read_text())
    cam["layouts"] = dict(note="mean sim/block_poses[0] over the train split, pose = xyz + quaternion xyzw (world). "
                          "Generated by scripts/fit_kinematics.py", variants=layouts)
    OUT_CAM.write_text(json.dumps(cam, indent=1))
    calibrate_photometric(train, args.workers)
    calibrate_bias(train, args.workers)
    print(f"wrote {OUT_KIN.relative_to(ROOT)} and layouts in {OUT_CAM.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
