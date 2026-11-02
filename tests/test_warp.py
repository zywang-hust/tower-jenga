"""Workspace warp W: bijectivity and inverse, gating, and the joint-space FrameTransform as served."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jenga.kinematics import ARMS, DEFAULT_CONFIG, load_arms
from jenga.offset import OffsetResult
from jenga.warp import (VARIANT_PARAMS, FrameTransform, Offset, Warp, WarpParams, family, firm_threshold, gate_offset,
                        load_layouts, scene_geometry)
from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT  # env JENGA_DATA_ROOT
SPLIT = ROOT / "runs/prep/split.json"
CAMERAS = ROOT / "configs/cameras.json"
needs_config = pytest.mark.skipif(not DEFAULT_CONFIG.exists() or not CAMERAS.exists(), reason="configs missing")
needs_data = pytest.mark.skipif(not (DATA / "MANIFEST.jsonl").exists() or not SPLIT.exists(),
                                reason="needs the local TOWER release and runs/prep")
CORNERS = [Offset(sx * 0.015, sy * 0.015, sx * sy * np.radians(10)) for sx in (-1, 1) for sy in (-1, 1)]
JOINTS = [*range(7), *range(8, 15)]


def val_episode(prefix: str) -> tuple[str, str]:
    split = json.loads(SPLIT.read_text())
    return next((k, t["val"][0]) for k, t in split["tasks"].items() if k.startswith(prefix))


def test_warp_is_a_diffeomorphism_rigid_inside_identity_outside():
    """Every shipped family band, the optional destination gate and a C2 profile, at the rule-box corners:
    analytic Jacobian = finite differences, det J > 0, exact inverse, rigid motion inside r_in, identity beyond."""
    c, a = np.array([0.1, -0.2]), np.array([0.1 + 0.353, -0.2])
    params = [*VARIANT_PARAMS.values(), WarpParams(0.12, 0.22, 0.10, 0.20), WarpParams(0.05, 0.25, profile="smootherstep")]
    rng = np.random.default_rng(1)
    for P in params:
        for off in CORNERS:
            W = Warp(off, c, P, a)
            assert W.min_jacobian_det(half_width=0.5, step=0.005) > 0.25
            x = c + rng.uniform(-0.5, 0.5, (2000, 2))
            assert np.abs(W.inverse(W.forward(x)) - x).max() < 1e-9
            assert np.abs(W.forward(W.inverse(x)) - x).max() < 1e-9
            eps = 1e-7
            fd = np.stack([(W.forward(x[:50] + eps * e) - W.forward(x[:50] - eps * e)) / (2 * eps) for e in np.eye(2)], -1)
            assert np.allclose(W.jacobian(x[:50]), fd, atol=1e-5)
            R = np.array([[np.cos(off.dpsi), -np.sin(off.dpsi)], [np.sin(off.dpsi), np.cos(off.dpsi)]])
            inside = c + np.array([[0.0, 0.0], [0.7 * P.r_in, 0.0], [0.0, -0.7 * P.r_in]])
            assert np.allclose(W.forward(inside), c + (inside - c) @ R.T + [off.dx, off.dy])
            far = c + np.array([[P.r_out + 0.01, 0.0], [0.0, -P.r_out - 0.05]])
            assert np.array_equal(W.forward(far), far)
            if P.dest_in is not None:
                assert np.array_equal(W.forward(a[None]), a[None])


def test_gate_and_deadband():
    x = np.linspace(0, 10, 2001)
    y = np.array([firm_threshold(v, 2.0) for v in x])
    assert np.all(y[x <= 2.0] == 0) and np.array_equal(y[x >= 4.0], x[x >= 4.0])
    assert np.abs(np.diff(y)).max() <= 2 * (x[1] - x[0]) + 1e-12  # continuous, slope <= 2: no jump at the threshold
    gate = dict(translation_mm=4.0, yaw_deg=2.0, yaw=True)
    assert gate_offset(Offset(0.003, -0.002, np.radians(-1.9)), gate).is_zero()  # |d| = 3.6 mm, 1.9 deg: inside
    big = Offset(0.015, -0.015, np.radians(-10))
    assert gate_offset(big, gate) == big  # beyond 2 tau: applied unchanged
    g = gate_offset(Offset(0.005, 0.0, np.radians(3)), gate)  # inside the ramp: 2 (x - tau), direction kept
    assert np.isclose(g.dx, 0.002) and g.dy == 0 and np.isclose(g.dpsi, np.radians(2))
    assert gate_offset(Offset(0.0, 0.0, np.radians(-3)), gate).dpsi < 0
    assert gate_offset(big, gate | dict(yaw=False)).dpsi == 0
    assert gate_offset(big, None) == big
    # legacy hard deadband (uncalibrated layouts) and the exact-zero identity
    P = WarpParams(0.12, 0.22)
    for off in (Offset(0.002, -0.002, np.radians(0.4)), Offset()):
        W = Warp(off, [0.0, 0.0], P)
        xs = np.random.default_rng(2).normal(0, 0.1, (10, 2))
        assert W.identity and np.array_equal(W.forward(xs), xs) and np.array_equal(W.inverse(xs), xs)
    assert not Warp(Offset(0.0031, 0, 0), [0, 0], P).identity
    assert not Warp(Offset(0, 0, np.radians(0.6)), [0, 0], P).identity
    assert Warp(Offset(), [0, 0], P, deadband=False).identity
    assert not Warp(Offset(1e-4, 0, 0), [0, 0], P, deadband=False).identity


class _FixedEstimator:
    def __init__(self, raw: Offset, gate: dict | None, ok: bool = True):
        self.raw, self.gate, self.ok = raw, gate, ok

    def estimate(self, task_variant, images):
        return OffsetResult(self.raw, 0.95, self.ok, 0.0, {})

    def gate_config(self, task_variant, result=None):
        return self.gate or {}


@needs_config
def test_from_observation_applies_the_gate():
    obs = {"task_variant": "task5-3_tower_insertion_3", "images": {}}
    gate = dict(translation_mm=3.8, yaw_deg=4.2, yaw=True)
    nominal = Offset(0.0012, -0.001, np.radians(-3.1))  # the task5-3 nominal yaw the hard 0.5 deg deadband passed
    ft = FrameTransform.from_observation(obs, estimator=_FixedEstimator(nominal, gate))
    assert ft.identity and ft.raw_offset == nominal and ft.offset.is_zero()
    shifted = Offset(0.015, -0.015, np.radians(10))
    ft = FrameTransform.from_observation(obs, estimator=_FixedEstimator(shifted, gate))
    assert not ft.identity and ft.offset == shifted
    ramp = Offset(0.005, 0.0, 0.0)  # 5 mm: inside the ramp, applied 2 (5 - 3.8) = 2.4 mm, no hard deadband
    ft = FrameTransform.from_observation(obs, estimator=_FixedEstimator(ramp, gate))
    assert not ft.identity and np.isclose(ft.offset.dx, 0.0024)
    assert FrameTransform.from_observation(obs, estimator=_FixedEstimator(shifted, gate, ok=False)).identity
    # uncalibrated layouts keep the legacy hard deadband
    assert FrameTransform.from_observation(obs, estimator=_FixedEstimator(Offset(0.002, 0, 0), None)).identity
    assert not FrameTransform.from_observation(obs, estimator=_FixedEstimator(nominal, None)).identity


def test_pose_round_trip():
    rng = np.random.default_rng(3)
    W = Warp(Offset(0.015, 0.015, np.radians(10)), [0.0, 0.0], WarpParams(0.12, 0.22))
    n = 200
    ang = rng.uniform(-np.pi, np.pi, n)
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = np.cos(ang), -np.sin(ang), np.sin(ang), np.cos(ang), 1
    p = np.c_[rng.uniform(-0.3, 0.3, (n, 2)), rng.uniform(0, 0.3, n)]
    tool = np.array([0.2, 0.01, -0.02])
    R2, p2, w = W.warp_pose(R, p, tool)
    R3, p3, w3 = W.unwarp_pose(R2, p2, tool)
    assert np.allclose(R3, R, atol=1e-9) and np.allclose(p3, p, atol=1e-9) and np.allclose(w, w3)
    assert np.allclose(p2[:, 2], p[:, 2])  # height untouched


@needs_config
def test_config_covers_all_variants():
    layouts = load_layouts()
    for v, lay in layouts.items():
        c, a = scene_geometry(v, layouts)
        assert 0.30 < np.linalg.norm(c - a) < 0.40  # source and destination ~353 mm apart
        assert family(v) in VARIANT_PARAMS
        assert lay["offset_gate"]["provenance"]["nominal_episodes"] >= 40  # else the legacy deadband would apply


@needs_config
def test_frame_transform_identity_passthrough():
    ft = FrameTransform("task1-3_tower_transfer_3", Offset(0.001, 0.0, 0.0))
    assert ft.identity
    a = np.random.default_rng(4).normal(0, 1, (50, 16)).astype(np.float32)
    out = ft.command(a)
    assert out.dtype == np.float32 and np.array_equal(out, a)
    assert np.array_equal(ft.observe(a[0]), a[0])
    # wire short names resolve to the layout
    assert FrameTransform("task1-3", Offset()).task_variant == "task1-3_tower_transfer_3"


@needs_config
@needs_data
def test_frame_transform_on_demonstration():
    """Demo commands at a rule-box corner: W realised at the tool point where reachable, home and releases untouched,
    grippers passed through, and the closed loop (command, then observe the executed row) returns the virtual
    command in joint space. Task4 push strokes already end at full extension, so some rows are unreachable."""
    arms = load_arms()
    off = Offset(0.015, -0.015, np.radians(10))
    for prefix in ("task1-3", "task4-2"):
        variant, rel = val_episode(prefix)
        with h5py.File(DATA / rel, "r") as f:
            act = f["leader/qpos"][:].astype(np.float32)
            names = json.loads(f["sim"].attrs["phase_names"])
            ph = f["sim/task_phase"][:]
        ft = FrameTransform(variant, off, arms=arms)
        chunk = act[::10][:400]
        real = np.concatenate([ft.command(chunk[i:i + 50]) for i in range(0, len(chunk), 50)])
        assert real.shape == chunk.shape and real.dtype == np.float32
        assert np.array_equal(real[:, [7, 15]], chunk[:, [7, 15]])  # grippers pass through
        assert np.array_equal(real[0], chunk[0])  # home posture is outside the warp
        reachable_everywhere = not prefix.startswith("task4")
        if reachable_everywhere:
            assert ft.joint.stats["ik_inexact"] == 0
        ft2 = FrameTransform(variant, off, arms=arms)
        back = []
        for s in range(0, len(chunk), 50):
            r = ft2.command(chunk[s:s + 50])
            back.extend(ft2.observe(r[j - s]) for j in range(s, s + len(r)) if j % 8 == 0)
        back = np.array(back)
        for i, arm in enumerate(ARMS):
            A, sl = arms[arm], slice(8 * i, 8 * i + 7)
            xv = A.tool_world(*A.fk_world(chunk[:, sl].astype(float)))
            xr = A.tool_world(*A.fk_world(real[:, sl].astype(float)))
            err = np.linalg.norm(ft.warp.forward(xv[:, :2]) - xr[:, :2], axis=1)
            assert (err < 1e-4).mean() == 1.0 if reachable_everywhere else (err < 1e-4).mean() > 0.9
            ok = err[::8] < 1e-4
            assert np.abs(back[:, sl] - chunk[::8, sl])[ok].max() < 2e-4  # float32 wire rounding; a token is 3-7 mrad
        rel_rows = np.flatnonzero(np.r_[False, np.diff(ph) != 0] & (ph == names.index("release")))
        if len(rel_rows):
            assert np.abs(FrameTransform(variant, off, arms=arms).command(act[rel_rows]) - act[rel_rows]).max() == 0


@needs_config
@needs_data
def test_served_commands_stay_smooth():
    """Served in k=8 chunks at 30 Hz with observe() between them, the real joint stream stays about as smooth as the
    demonstration. Task4 (inside the warp almost throughout) guards the null-space polish: p99 second-difference
    ratio 0.82 with it, 4.7 without. Task1 crosses the transition band between the tower, home and destination:
    2.0 on this episode (11.5 without the polish); over 48 Task1 runs 1.5x mean, 2.2x max, against 6.0x with the
    former 12 -> 22 cm band (runs/spatial/validate/r.json, runs/spatial/profile_sweep.json)."""
    arms = load_arms()
    for prefix, bound in (("task4-2", 2.0), ("task1-3", 2.5)):
        variant, rel = val_episode(prefix)
        with h5py.File(DATA / rel, "r") as f:
            v = f["leader/qpos"][:].astype(np.float32)
            n = int(f["sim/action_valid"][:].sum())
        v = v[len(v) // 4:][:480] if prefix == "task4-2" else v[:n - n % 8]
        ft = FrameTransform(variant, Offset(0.015, -0.015, np.radians(10)), arms=arms)
        real, last = [], v[0]
        for s in range(0, len(v), 8):
            ft.observe(last)
            real.append(ft.command(v[s:s + 8]))
            last = real[-1][-1]
        real = np.concatenate(real)
        moved = (np.abs(real - v)[:, JOINTS].max(1) > 1e-6)[1:-1]
        assert moved.mean() > 0.2  # the warp is active
        d2v = np.percentile(np.abs(np.diff(v[:, JOINTS], 2, axis=0)).max(1)[moved], 99)
        d2r = np.percentile(np.abs(np.diff(real[:, JOINTS], 2, axis=0)).max(1)[moved], 99)
        assert d2r < bound * d2v, (prefix, d2r / d2v)
