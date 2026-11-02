"""Offline validation of the test-time reference-frame transform (jenga.offset + jenga.warp).

Questions, one per part (outputs in runs/spatial/validate/):
  a  How accurately does the image estimator recover rule-sized offsets (15 mm / 10 deg), raw and after the
     per-variant gate (warp.gate_offset)?  Val episodes' first frames with the movable blocks moved in pixel
     space (offset.synthesize_shift); truth = known composition of the frame's own offset and the synthetic
     one.  Footprint error = largest corner distance of the movable blocks (OffsetEstimator.footprint_error_mm).
  b  Is a nominal scene left alone, and does the virtual state remove the layout variation?  For the 56 val
     episodes: raw and gated estimate on the unshifted frame 0 (how often, and by how much, a nominal scene is
     moved), and the cross-episode spread of the tool point and of the joints at grasp ('close' start),
     release and push events, real vs virtual (FrameTransform.observe with each episode's true offset, the raw
     estimate, and the gated estimate as served).  Grasps should tighten, releases (destination, not warped)
     must stay identical.
  c  Are demonstrated commands still executable after W?  leader/qpos of val episodes as virtual commands,
     command() in H=50 chunks under corner offsets: step continuity, observed joint range, IK fallbacks,
     warp weight at home / release / Task4 push.
  d  Serving latency: one estimate, observe(1 row), command(H rows), single-arm single-row IK.
  r  Does the warp roughen the joint commands, and does it still move the arm exactly with the assembly where it
     manipulates it?  leader/qpos of val episodes served in k=8 chunks under the four corner offsets: real /
     virtual second-difference p99, tool-point error against the ideal at source and destination phases.
  l  Does the estimator hold under lighting changes?  Val frame 0, nominal and two rule-box corner shifts, under
     the training lighting augmentation's extremes (gain 0.3-2.0, white balance 3000-8500 K, and their corners),
     gamma 0.7 / 1.5 and one-sided shading ramps: ok rate, spurious motion of nominal scenes, gated footprint error.

    python scripts/validate_warp.py [--parts abcdrl] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple
from pathlib import Path

import cv2
import h5py
import numpy as np
from fit_kinematics import DATA, ROOT, RULE_D, RULE_PSI, SPLIT, compose, first_frame

from jenga.kinematics import ARMS, load_arms
from jenga.offset import EXTERNAL, OffsetEstimator, is_movable, synthesize_shift, true_offset
from jenga.warp import FrameTransform, Offset, Warp, WarpParams, family, gate_offset, load_layouts

OUT = ROOT / "runs/spatial/validate"
H = 50


def val_episodes() -> list[str]:
    split = json.loads(SPLIT.read_text())
    return [e for t in split["tasks"].values() for e in t["val"]]


def phase_starts(f) -> list[tuple[int, str, int]]:
    ph = f["sim/task_phase"][:]
    ab = f["sim/active_block"][:]
    names = json.loads(f["sim"].attrs["phase_names"])
    ch = np.r_[0, np.flatnonzero(np.diff(ph) | np.diff(ab)) + 1]
    return [(int(i), names[ph[i]], int(ab[i])) for i in ch]


# ---------------------------------------------------------------- a
_EST = None


def _est():
    global _EST
    if _EST is None:
        cv2.setNumThreads(1)
        _EST = OffsetEstimator()
    return _EST


def part_a_job(args):
    rel, k, seed = args
    est = _est()
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        imgs = first_frame(f)
        ids = json.loads(f["sim"].attrs["block_ids"])
        P = f["sim/block_poses"][0].astype(float)
    t0 = true_offset(est.layouts[v]["blocks"], ids, P)
    rng = np.random.default_rng(seed)
    if k < 4:  # corners of the rule box
        d1 = np.array([RULE_D * (1 if k & 1 else -1), RULE_D * (1 if k & 2 else -1), RULE_PSI * rng.choice([-1, 1])])
    else:
        d1 = rng.uniform([-RULE_D, -RULE_D, -RULE_PSI], [RULE_D, RULE_D, RULE_PSI])
    _, _, c = est.template(v)
    mv = np.array([is_movable(b) for b in ids])
    syn = {cam: synthesize_shift(imgs[cam], est.cams[cam], P, mv, c, Offset(*d1), est.dims) for cam in EXTERNAL}
    r = est.estimate(v, syn)
    truth = compose(t0, d1)
    raw = np.array(astuple(r.offset))
    gated = np.array(astuple(gate_offset(r.offset, est.gate_config(v, r) or None))) if r.ok else np.zeros(3)
    e, eg = raw - truth, gated - truth
    fp = est.footprint_error_mm(v, np.stack([raw, gated, np.zeros(3)]), truth[None])
    return dict(variant=v, rel=rel, truth=truth.tolist(), raw=raw.tolist(), gated=gated.tolist(),
                err_xy_mm=float(1e3 * np.hypot(*e[:2])), err_psi_deg=float(np.degrees(e[2])),
                gated_err_xy_mm=float(1e3 * np.hypot(*eg[:2])), gated_err_psi_deg=float(np.degrees(eg[2])),
                footprint_mm=dict(raw=float(fp[0]), gated=float(fp[1]), none=float(fp[2])),
                confidence=r.confidence, ok=r.ok, seconds=r.seconds, corner=k < 4)


def summarize(x) -> dict:
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.median(x)), p95=float(np.percentile(x, 95)),
                max=float(x.max()))


def error_summary(rows) -> dict:
    return dict(xy_mm=summarize([r["err_xy_mm"] for r in rows]),
                psi_abs_deg=summarize([abs(r["err_psi_deg"]) for r in rows]),
                gated_xy_mm=summarize([r["gated_err_xy_mm"] for r in rows]),
                gated_psi_abs_deg=summarize([abs(r["gated_err_psi_deg"]) for r in rows]),
                footprint_mm={k: summarize([r["footprint_mm"][k] for r in rows]) for k in ("raw", "gated", "none")})


def part_a(eps, workers, per_episode=6):
    jobs = [(rel, k, 1000 * i + k) for i, rel in enumerate(eps) for k in range(per_episode)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(part_a_job, jobs, chunksize=2))
    res = dict(overall=error_summary(rows) | dict(ok_rate=float(np.mean([r["ok"] for r in rows])),
                                                  confidence_min=float(min(r["confidence"] for r in rows)),
                                                  seconds_in_pool=summarize([r["seconds"] for r in rows])),
               corners=error_summary([r for r in rows if r["corner"]]),
               by_variant={})
    by = defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    for v, rs in sorted(by.items()):
        res["by_variant"][v] = error_summary(rs)
    (OUT / "a_rows.json").write_text(json.dumps(rows, indent=0))
    return res


# ---------------------------------------------------------------- b
def part_b_job(rel):
    est = _est()
    arms = load_arms()
    layouts = est.layouts
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        ids = json.loads(f["sim"].attrs["block_ids"])
        t0 = true_offset(layouts[v]["blocks"], ids, f["sim/block_poses"][0].astype(float))
        r = est.estimate(v, first_frame(f))
        g = gate_offset(r.offset, est.gate_config(v, r) or None) if r.ok else Offset()
        q = f["observations/qpos"][:].astype(float)
        tm = {m["block_index"]: m["arm"] for m in json.loads(f["sim"].attrs["target_mapping"])}
        events = []
        for i, name, b in phase_starts(f):
            if name in ("close", "release", "push") and b in tm:
                events.append((i, name, tm[b], ids[b]))
    out = dict(variant=v, rel=rel, true=list(astuple(t0)), est=list(astuple(r.offset)) if r.ok else [0, 0, 0],
               gated=list(astuple(g)), confidence=r.confidence, ok=r.ok, events=[])
    out["footprint_mm"] = dict(zip(("est", "gated", "none"), est.footprint_error_mm(
        v, np.array([out["est"], out["gated"], [0, 0, 0]]), np.array(out["true"])[None]).tolist()))
    rows = q[[e[0] for e in events]]
    virt = {}
    for kind, off in (("true", t0), ("est", Offset(*out["est"])), ("gated", g)):
        ft = FrameTransform(v, off, layouts=layouts, arms=arms, deadband=False)
        virt[kind] = np.array([ft.observe(row) for row in rows])  # one row at a time, as served
    for j, (i, name, a, bid) in enumerate(events):
        A = arms[ARMS[a]]
        sl = slice(8 * a, 8 * a + 7)
        rec = dict(frame=i, phase=name, arm=a, block=bid)
        for kind, qq in (("real", rows[j]), *((k, virt[k][j]) for k in ("true", "est", "gated"))):
            R, p = A.fk_world(qq[sl][None])
            rec[kind] = dict(tool=A.tool_world(R, p)[0].tolist(), q=qq[sl].tolist())
        out["events"].append(rec)
    return out


def part_b(eps, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(part_b_job, eps))
    (OUT / "b_rows.json").write_text(json.dumps(res, indent=0))
    groups = defaultdict(list)  # (variant, arm, block, phase) -> per-episode records (first occurrence)
    for ep in res:
        seen = set()
        for e in ep["events"]:
            key = (ep["variant"], e["arm"], e["block"], e["phase"])
            if key not in seen:
                seen.add(key)
                groups[key].append(e)
    agg = defaultdict(lambda: defaultdict(list))
    for (v, _a, _b, ph), es in groups.items():
        if len(es) < 3:
            continue
        for kind in ("real", "true", "est", "gated"):
            X = np.array([e[kind]["tool"][:2] for e in es])
            Q = np.array([e[kind]["q"] for e in es])
            agg[(family(v), ph)][kind + "_tool_xy_std_mm"].append(1e3 * np.sqrt(X.var(0).sum()))
            agg[(family(v), ph)][kind + "_q_std_deg"].append(np.degrees(np.sqrt(Q.var(0).mean())))
    table = {f"{f}:{ph}": {k: float(np.mean(x)) for k, x in d.items()} | {"groups": len(d["real_tool_xy_std_mm"])}
             for (f, ph), d in sorted(agg.items())}
    nominal = {}
    by = defaultdict(list)
    for ep in res:
        by[ep["variant"]].append(ep)
    for v, eps_v in sorted(by.items()):
        E, G = np.array([e["est"] for e in eps_v]), np.array([e["gated"] for e in eps_v])
        nominal[v] = dict(
            episodes=len(eps_v), warped=int(sum(np.any(G != 0, 1))), yaw_warped=int(sum(G[:, 2] != 0)),
            raw_yaw_abs_deg_max=float(np.degrees(np.abs(E[:, 2]).max())),
            applied_yaw_abs_deg_max=float(np.degrees(np.abs(G[:, 2]).max())),
            applied_translation_mm_max=float(1e3 * np.hypot(*G[:, :2].T).max()),
            footprint_mm={k: summarize([e["footprint_mm"][k] for e in eps_v]) for k in ("est", "gated", "none")})
    tot = dict(episodes=len(res), warped=sum(n["warped"] for n in nominal.values()),
               yaw_warped=sum(n["yaw_warped"] for n in nominal.values()),
               footprint_mm={k: summarize([e["footprint_mm"][k] for e in res]) for k in ("est", "gated", "none")})
    return dict(note="cross-episode std at the first occurrence of each (variant, arm, block, phase) event, averaged "
                     "over groups; tool_xy = sqrt(var x + var y), q = rms joint std. Kinds: real state, virtual state "
                     "with the true offset, the raw estimate, and the gated estimate (as served). nominal: frame-0 "
                     "estimate of the unshifted val episodes; footprint = largest movable-block corner error (mm) "
                     "left after applying the raw / gated / no offset", table=table,
                nominal=dict(overall=tot, by_variant=nominal))


# ---------------------------------------------------------------- c
CORNERS = [(RULE_D, RULE_D, RULE_PSI), (-RULE_D, RULE_D, -RULE_PSI), (RULE_D, -RULE_D, -RULE_PSI),
           (-RULE_D, -RULE_D, RULE_PSI)]


def part_c_job(args):
    rel, k = args
    arms = load_arms()
    layouts = load_layouts()
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        act = f["leader/qpos"][:].astype(float)
        starts = phase_starts(f)
        ph = f["sim/task_phase"][:]
        names = json.loads(f["sim"].attrs["phase_names"])
    ft = FrameTransform(v, Offset(*CORNERS[k]), layouts=layouts, arms=arms)
    real = np.concatenate([ft.command(act[i:i + H]) for i in range(0, len(act), H)])
    out = dict(variant=v, rel=rel, corner=k, ik_inexact=ft.joint.stats["ik_inexact"],
               max_ik_err_mm=1e3 * ft.joint.stats["max_ik_err_m"])
    w_all = {}
    for a, arm in enumerate(ARMS):
        sl = slice(8 * a, 8 * a + 7)
        A = arms[arm]
        R, p = A.fk_world(act[:, sl])
        w_all[arm] = ft.warp.weight(A.tool_world(R, p)[:, :2])
        dv = np.abs(np.diff(act[:, sl], axis=0)).max(1)
        dr = np.abs(np.diff(real[:, sl], axis=0)).max(1)
        out[arm] = dict(
            max_step_virtual_deg=float(np.degrees(dv.max())), max_step_real_deg=float(np.degrees(dr.max())),
            max_step_increase_deg=float(np.degrees((dr - dv).max())),
            max_joint_shift_deg=float(np.degrees(np.abs(real[:, sl] - act[:, sl]).max())),
            rows_outside_observed_range=int(((real[:, sl] < A.q_lo - 1e-6) | (real[:, sl] > A.q_hi + 1e-6)).any(1).sum()),
            rows_outside_range_virtual=int(((act[:, sl] < A.q_lo - 1e-6) | (act[:, sl] > A.q_hi + 1e-6)).any(1).sum()),
            max_range_excess_deg=float(np.degrees(np.maximum(np.maximum(A.q_lo - real[:, sl], real[:, sl] - A.q_hi), 0).max())),
        )
    out["grippers_identical"] = bool(np.array_equal(real[:, [7, 15]], act[:, [7, 15]]))
    out["w_home_max"] = float(max(w_all[a][0] for a in ARMS))
    rel_w, push_w = [], []
    for i, name, _b in starts:
        if name == "release":
            rel_w.append(max(w_all[a][i] for a in ARMS))
    for arm in ARMS:
        m = np.array([names[x] == "push" for x in ph])
        if m.any():
            push_w.append(w_all[arm][m])
    out["w_release_max"] = float(max(rel_w)) if rel_w else None
    if push_w:  # the pushing arm is the one with the larger weight during push
        out["w_push_min_best_arm"] = float(max(np.min(x) for x in push_w))
    return out


def part_c(eps, workers):
    jobs = [(rel, k) for rel in eps for k in range(len(CORNERS))]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(part_c_job, jobs))
    (OUT / "c_rows.json").write_text(json.dumps(rows, indent=0))
    res = dict(runs=len(rows), ik_inexact_rows=int(sum(r["ik_inexact"] for r in rows)),
               max_ik_err_mm=float(max(r["max_ik_err_mm"] for r in rows)),
               grippers_identical=all(r["grippers_identical"] for r in rows),
               w_home_max=float(max(r["w_home_max"] for r in rows)),
               w_release_max=float(max(r["w_release_max"] for r in rows if r["w_release_max"] is not None)))
    pw = [r["w_push_min_best_arm"] for r in rows if "w_push_min_best_arm" in r]
    if pw:
        res["task4_w_push_min"] = float(min(pw))
    for arm in ARMS:
        res[arm] = dict(max_step_increase_deg=summarize([r[arm]["max_step_increase_deg"] for r in rows]),
                        max_step_real_deg=float(max(r[arm]["max_step_real_deg"] for r in rows)),
                        max_step_virtual_deg=float(max(r[arm]["max_step_virtual_deg"] for r in rows)),
                        max_joint_shift_deg=float(max(r[arm]["max_joint_shift_deg"] for r in rows)),
                        rows_outside_observed_range=int(sum(r[arm]["rows_outside_observed_range"] for r in rows)),
                        rows_outside_range_virtual=int(sum(r[arm]["rows_outside_range_virtual"] for r in rows)),
                        max_range_excess_deg=float(max(r[arm]["max_range_excess_deg"] for r in rows)))
    by = defaultdict(list)
    for r in rows:
        by[family(r["variant"])].append(max(r[a]["max_range_excess_deg"] for a in ARMS))
    res["max_range_excess_deg_by_family"] = {k: float(max(v)) for k, v in sorted(by.items())}
    fam = defaultdict(list)
    for r in rows:
        fam[family(r["variant"])].append(r)
    res["by_family"] = {k: dict(ik_inexact_rows=int(sum(r["ik_inexact"] for r in rs)),
                                max_ik_err_mm=float(max(r["max_ik_err_mm"] for r in rs)),
                                max_step_increase_deg=float(max(r[a]["max_step_increase_deg"] for r in rs for a in ARMS)),
                                max_joint_shift_deg=float(max(r[a]["max_joint_shift_deg"] for r in rs for a in ARMS)))
                        for k, rs in sorted(fam.items())}
    return res


# ---------------------------------------------------------------- r
K_SERVE = 8  # rows commanded per request (server.handler_factory k)
SOURCE_PHASES = {"approach", "close", "lift"}  # the active arm is at the movable assembly: full warp expected
DEST_PHASES = {"place", "release"}  # at the fixed destination: no warp expected
SOURCE_EXTRA = {"task4": {"push", "retreat", "settle"}}  # Task4 pushes the movable tower itself
DEST_EXTRA = {"task5": {"push"}}  # Task5 pushes the inserted block at the destination


def serve_replay(ft: FrameTransform, v: np.ndarray, k: int = K_SERVE) -> np.ndarray:
    """Real command stream as served: observe() of the last executed real row, then command() of the next k rows."""
    real, last = [], v[0]
    for s in range(0, len(v), k):
        ft.observe(last)
        real.append(ft.command(v[s:s + k]))
        last = real[-1][-1]
    return np.concatenate(real)


def part_r_job(args):
    """Roughness and manipulation-point accuracy of one episode under one corner offset, served in k=8 chunks."""
    rel, k = args
    arms = load_arms()
    layouts = load_layouts()
    v = Path(rel).parent.name
    fam = family(v)
    with h5py.File(DATA / rel, "r") as f:
        act = f["leader/qpos"][:].astype(np.float32)
        n = int(f["sim/action_valid"][:].sum())
        ph = f["sim/task_phase"][:n]
        ab = f["sim/active_block"][:n]
        names = json.loads(f["sim"].attrs["phase_names"])
        tm = {m["block_index"]: m["arm"] for m in json.loads(f["sim"].attrs["target_mapping"])}
    n -= n % K_SERVE
    act, ph, ab = act[:n], ph[:n], ab[:n]
    off = Offset(*CORNERS[k])
    ft = FrameTransform(v, off, layouts=layouts, arms=arms)
    t0 = time.perf_counter()
    real = serve_replay(ft, act)
    secs = time.perf_counter() - t0
    joints = [*range(7), *range(8, 15)]
    moved = (np.abs(real - act)[:, joints].max(1) > 1e-6)[1:-1]
    d2v = np.abs(np.diff(act[:, joints], 2, axis=0)).max(1)[moved]
    d2r = np.abs(np.diff(real[:, joints].astype(float), 2, axis=0)).max(1)[moved]
    src = SOURCE_PHASES | SOURCE_EXTRA.get(fam, set())
    dst = DEST_PHASES | DEST_EXTRA.get(fam, set())
    phase = np.array([names[x] for x in ph])
    rigid = Warp(off, ft.warp.c, WarpParams(1e3, 2e3), deadband=False)  # the true rigid motion everywhere
    err = dict(source=[], dest=[], source_w=[], dest_w=[])
    home_w = 0.0
    for a, arm in enumerate(ARMS):
        A, sl = arms[arm], slice(8 * a, 8 * a + 7)
        active = np.array([tm.get(int(b), -1) == a for b in ab])
        xv = A.tool_world(*A.fk_world(act[:, sl].astype(float)))[:, :2]
        xr = A.tool_world(*A.fk_world(real[:, sl].astype(float)))[:, :2]
        w = ft.warp.weight(xv)
        home_w = max(home_w, float(w[0]))
        for kind, ph_set, ideal in (("source", src, rigid.forward(xv)), ("dest", dst, xv)):
            m = active & np.isin(phase, list(ph_set))
            err[kind] += (1e3 * np.linalg.norm(xr[m] - ideal[m], axis=1)).tolist()
            err[kind + "_w"] += w[m].tolist()
    return dict(variant=v, rel=rel, corner=k, rows=n, seconds=secs, ik_inexact=ft.joint.stats["ik_inexact"],
                d2_virtual_p99=float(np.percentile(d2v, 99)), d2_real_p99=float(np.percentile(d2r, 99)),
                d2_virtual_max=float(d2v.max()), d2_real_max=float(d2r.max()),
                source_err_mm=dict(max=max(err["source"], default=0.0),
                                   p99=float(np.percentile(err["source"], 99)) if err["source"] else 0.0),
                dest_err_mm=dict(max=max(err["dest"], default=0.0),
                                 p99=float(np.percentile(err["dest"], 99)) if err["dest"] else 0.0),
                source_w_min=min(err["source_w"], default=1.0), dest_w_max=max(err["dest_w"], default=0.0),
                source_rows=len(err["source"]), dest_rows=len(err["dest"]), home_w=home_w,
                home_moved=bool(np.abs(real[0] - act[0]).max() > 0))


def roughness_summary(rows) -> dict:
    out = {}
    fam = defaultdict(list)
    for r in rows:
        fam[family(r["variant"])].append(r)
    for f, rs in sorted(fam.items()):
        ratio = [r["d2_real_p99"] / r["d2_virtual_p99"] for r in rs]
        out[f] = dict(runs=len(rs), d2_p99_ratio_mean=float(np.mean(ratio)), d2_p99_ratio_max=float(np.max(ratio)),
                      d2_max_ratio_mean=float(np.mean([r["d2_real_max"] / r["d2_virtual_max"] for r in rs])),
                      source_err_mm_max=float(max(r["source_err_mm"]["max"] for r in rs)),
                      source_err_mm_p99_mean=float(np.mean([r["source_err_mm"]["p99"] for r in rs])),
                      dest_err_mm_max=float(max(r["dest_err_mm"]["max"] for r in rs)),
                      source_w_min=float(min(r["source_w_min"] for r in rs)),
                      dest_w_max=float(max(r["dest_w_max"] for r in rs)),
                      home_w_max=float(max(r["home_w"] for r in rs)), home_moved=int(sum(r["home_moved"] for r in rs)),
                      ik_inexact_rows=int(sum(r["ik_inexact"] for r in rs)),
                      ms_per_request=float(1e3 * sum(r["seconds"] for r in rs) / sum(r["rows"] / K_SERVE for r in rs)))
    return out


def part_r(eps, workers):
    jobs = [(rel, k) for rel in eps for k in range(len(CORNERS))]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(part_r_job, jobs))
    (OUT / "r_rows.json").write_text(json.dumps(rows, indent=0))
    return dict(note="val episodes x 4 corner offsets (15 mm, 10 deg), leader/qpos replayed as virtual commands in "
                     f"k={K_SERVE} chunks with observe() of the last real row in between (as served). d2 = per-row max "
                     "over joints of |second difference| on rows whose command is changed by the warp; ratio = "
                     "real p99 / virtual p99 per run. source/dest err = horizontal tool-point distance (mm) of the "
                     "executed command from the ideal (rigidly moved demo at the assembly; untouched demo at the "
                     "destination) on the active arm's rows in the source / destination phases; w = warp weight there.",
                by_family=roughness_summary(rows))


# ---------------------------------------------------------------- l
def _scale(gain=1.0, kelvin=6500.0):
    from jenga.dataset import kelvin_rgb
    return lambda x: x * (gain * kelvin_rgb(kelvin))


# The training lighting augmentation (jenga.dataset.lighting: gain log-uniform 0.3-2.0 x white balance 3000-8500 K,
# one draw for all views) at its extremes and corners, plus gamma and one-sided shading the augmentation lacks.
LIGHTING = {
    "none": lambda x: x,
    "gain0.3": _scale(0.3), "gain0.5": _scale(0.5), "gain2.0": _scale(2.0),
    "k3000": _scale(kelvin=3000), "k8500": _scale(kelvin=8500),
    "gain0.3_k3000": _scale(0.3, 3000), "gain0.3_k8500": _scale(0.3, 8500),
    "gain2.0_k3000": _scale(2.0, 3000), "gain2.0_k8500": _scale(2.0, 8500),
    "gamma0.7": lambda x: x ** 0.7, "gamma1.5": lambda x: x ** 1.5,
    "ramp_x": lambda x: x * np.linspace(0.55, 1.15, x.shape[1])[None, :, None],
    "ramp_y": lambda x: x * np.linspace(1.15, 0.55, x.shape[0])[:, None, None],
}


def relight(img: np.ndarray, kind: str) -> np.ndarray:
    x = LIGHTING[kind](img.astype(np.float32) / 255.0)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def part_l_job(args):
    """One val frame 0 (k < 0) or one rule-box corner shift of it (k = CORNERS index) under every LIGHTING kind."""
    rel, k = args
    est = _est()
    v = Path(rel).parent.name
    with h5py.File(DATA / rel, "r") as f:
        imgs = first_frame(f)
        ids = json.loads(f["sim"].attrs["block_ids"])
        P = f["sim/block_poses"][0].astype(float)
    t0 = true_offset(est.layouts[v]["blocks"], ids, P)
    truth = np.array(astuple(t0))
    if k >= 0:
        d1 = np.array(CORNERS[k])
        _, _, c = est.template(v)
        mv = np.array([is_movable(b) for b in ids])
        imgs = {cam: synthesize_shift(imgs[cam], est.cams[cam], P, mv, c, Offset(*d1), est.dims) for cam in EXTERNAL}
        truth = compose(t0, d1)
    rows = []
    for kind in LIGHTING:
        r = est.estimate(v, {cam: relight(imgs[cam], kind) for cam in EXTERNAL})
        raw = np.array(astuple(r.offset))
        gated = np.array(astuple(gate_offset(r.offset, est.gate_config(v, r) or None))) if r.ok else np.zeros(3)
        fp = est.footprint_error_mm(v, np.stack([raw, gated, np.zeros(3)]), truth[None])
        rows.append(dict(variant=v, rel=rel, k=k, lighting=kind, truth=truth.tolist(), raw=raw.tolist(),
                         gated=gated.tolist(), ok=r.ok, confidence=r.confidence, saturated=r.saturated, seconds=r.seconds,
                         err_xy_mm=float(1e3 * np.hypot(*(raw - truth)[:2])),
                         err_psi_deg=float(np.degrees(raw[2] - truth[2])),
                         footprint_mm=dict(raw=float(fp[0]), gated=float(fp[1]), none=float(fp[2]))))
    return rows


def part_l(eps, workers):
    jobs = [(rel, k) for rel in eps for k in (-1, 0, 3)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(part_l_job, jobs) for r in rs]
    (OUT / "l_rows.json").write_text(json.dumps(rows, indent=0))
    ref = {(r["rel"], r["k"]): np.array(r["raw"]) for r in rows if r["lighting"] == "none"}
    res = {}
    for kind in LIGHTING:
        nom = [r for r in rows if r["lighting"] == kind and r["k"] < 0]
        sh = [r for r in rows if r["lighting"] == kind and r["k"] >= 0]
        G = np.array([r["gated"] for r in nom])
        fpn = np.array([[r["footprint_mm"]["gated"], r["footprint_mm"]["none"]] for r in nom])
        dev = [np.subtract(r["raw"], ref[(r["rel"], r["k"])]) for r in nom]
        res[kind] = dict(
            nominal=dict(n=len(nom), ok=int(sum(r["ok"] for r in nom)), moved=int(np.any(G != 0, 1).sum()),
                         rotated=int((G[:, 2] != 0).sum()),
                         raw_err_xy_mm=summarize([r["err_xy_mm"] for r in nom]),
                         raw_psi_abs_deg=summarize([abs(r["err_psi_deg"]) for r in nom]),
                         raw_shift_vs_none_xy_mm=summarize([1e3 * np.hypot(*d[:2]) for d in dev]),
                         gated_footprint_mm=summarize(fpn[:, 0]),
                         worse_than_identity_0p5mm=int((fpn[:, 0] > fpn[:, 1] + 0.5).sum())),
            shifted=dict(n=len(sh), ok=int(sum(r["ok"] for r in sh)),
                         raw_err_xy_mm=summarize([r["err_xy_mm"] for r in sh]),
                         raw_psi_abs_deg=summarize([abs(r["err_psi_deg"]) for r in sh]),
                         gated_footprint_mm=summarize([r["footprint_mm"]["gated"] for r in sh]),
                         none_footprint_mm=summarize([r["footprint_mm"]["none"] for r in sh])),
            seconds_in_pool=summarize([r["seconds"] for r in nom + sh]))
    return dict(note="val frame 0, nominal and two rule-box corner shifts (CORNERS[0], CORNERS[3]; 15 mm, 10 deg) "
                     "synthesized before relighting, under each LIGHTING kind applied to both external cameras. "
                     "gated = served offset (identity when ok is False); footprint = largest movable-block corner "
                     "error (mm); shift_vs_none = raw estimate change against the unperturbed image.",
                by_lighting=res)


# ---------------------------------------------------------------- d
def timeit(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    t = 1e3 * np.array(t)
    return dict(median_ms=float(np.median(t)), p99_ms=float(np.percentile(t, 99)), max_ms=float(t.max()))


def part_d(eps):
    est = OffsetEstimator()
    arms = load_arms()
    res = {}
    est_t = []
    for rel in eps[::3]:
        with h5py.File(DATA / rel, "r") as f:
            imgs = first_frame(f)
        v = Path(rel).parent.name
        t0 = time.perf_counter()
        ft = FrameTransform.from_observation({"task_variant": v, "images": imgs}, estimator=est, arms=arms)
        est_t.append(time.perf_counter() - t0)
    res["from_observation_ms"] = summarize(1e3 * np.array(est_t))
    est_cold = time.perf_counter()
    OffsetEstimator()
    res["estimator_construct_ms"] = 1e3 * (time.perf_counter() - est_cold)
    # per-step serving cost near the source (w = 1), task1 val episode, grasp frame
    rel = next(e for e in eps if Path(e).parent.name.startswith("task1-3"))
    with h5py.File(DATA / rel, "r") as f:
        q = f["observations/qpos"][:].astype(np.float32)
        act = f["leader/qpos"][:].astype(np.float32)
        i = next(s for s, name, _ in phase_starts(f) if name == "close")
    ft = FrameTransform(Path(rel).parent.name, Offset(0.015, -0.015, np.radians(10)), arms=arms)
    res["observe_1row"] = timeit(lambda: ft.observe(q[i]), 300)
    res[f"command_H{H}"] = timeit(lambda: ft.command(act[i:i + H]), 100)
    res["command_H10"] = timeit(lambda: ft.command(act[i:i + 10]), 100)
    A = arms["left"]
    R, p = A.fk_world(q[i:i + 1, :7].astype(float))
    seed = q[i:i + 1, :7].astype(float) + 0.01
    res["ik_single_arm_single_row"] = timeit(lambda: A.ik_world(R, p, seed), 500)
    ident = FrameTransform(Path(rel).parent.name, Offset(), arms=arms)
    res["identity_observe_1row"] = timeit(lambda: ident.observe(q[i]), 300)
    res["budget_note"] = "30 Hz step = 33.3 ms; observe runs every step, command once per chunk"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abcdrl")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    eps = val_episodes()
    for part, fn in (("a", lambda: part_a(eps, args.workers)), ("b", lambda: part_b(eps, args.workers)),
                     ("c", lambda: part_c(eps, args.workers)), ("d", lambda: part_d(eps)),
                     ("r", lambda: part_r(eps, args.workers)), ("l", lambda: part_l(eps, args.workers))):
        if part not in args.parts:
            continue
        t0 = time.time()
        res = fn()
        (OUT / f"{part}.json").write_text(json.dumps(res, indent=1))
        print(f"part {part} done in {time.time() - t0:.0f}s:", json.dumps(res, indent=1), flush=True)
    print("VALIDATE_DONE")


if __name__ == "__main__":
    main()
