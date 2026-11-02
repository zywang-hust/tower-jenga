"""Test-time reference-frame transform: run a policy trained on nominal layouts in a virtual nominal scene.

The robustness release moves only the movable assembly (XY up to 15 mm per axis, yaw up to 10 deg);
foundations, destination anchors and cameras stay fixed. Demonstrated end-effector motion near the source
assembly is rigidly attached to it, motion near the destination and the home posture are not
(runs/spatial/track.txt). The workspace warp W therefore moves end-effector poses rigidly with the assembly
near its centre and leaves them untouched far away:

    x_real = W(x_virtual) = c + Rz(w psi) (x_virtual - c) + w d,     w = s(|x_v - c|) * g(|x_v - a|)

on the horizontal coordinates of the tool point x (centre of a held block, kinematics.Arm.tool); z is kept.
s is 1 inside r_in and 0 beyond r_out, g is 0 inside dest_in of the destination anchor a and 1 beyond
dest_out (optional; the shipped VARIANT_PARAMS do not use it), both C1 smoothsteps. The whole end-effector pose
follows the local rigid motion (Rz(w psi), translation) evaluated at its tool point, so the gripper geometry is
preserved.

FrameTransform is the serving interface: built once per episode from the first observation, then
observe(q16_real) -> q16_virtual for the model input and command(actions_virtual [H,16]) -> actions_real.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jenga.kinematics import ARMS, Arm, load_arms, rot_z


@dataclass(frozen=True)
class Offset:
    """Rigid displacement of the movable assembly about its nominal centre: metres and radians."""

    dx: float = 0.0
    dy: float = 0.0
    dpsi: float = 0.0

    def within_deadband(self, d_m: float = 0.003, psi_rad: float = np.radians(0.5)) -> bool:
        return bool(np.hypot(self.dx, self.dy) < d_m and abs(self.dpsi) < psi_rad)

    def is_zero(self) -> bool:
        return self.dx == 0 and self.dy == 0 and self.dpsi == 0


def firm_threshold(x: float, tau: float) -> float:
    """Firm shrinkage of a magnitude: 0 up to tau, 2 (x - tau) on [tau, 2 tau], x beyond. Continuous, so an
    estimate just above the threshold moves the scene by a little instead of jumping to the full offset, and
    unbiased for offsets clearly above the noise (soft thresholding would shorten every correction by tau)."""
    x = float(x)
    return 0.0 if x <= tau else 2.0 * (x - tau) if x <= 2.0 * tau else x


def gate_offset(raw: Offset, gate: dict | None) -> Offset:
    """Applied offset from a raw image estimate and the variant's calibrated gate (configs/cameras.json
    layouts[variant]["offset_gate"], fitted by scripts/fit_kinematics.py --stage gate):
      translation  firm_threshold on |d| with tau = translation_mm (direction kept)
      yaw          firm_threshold on |dpsi| with tau = yaw_deg, or 0 when "yaw" is false
    The thresholds are the worst nominal estimator errors per variant, so a scene inside the training layout
    distribution is not moved by estimator noise. Without a gate the legacy hard deadband applies (Warp)."""
    if gate is None:
        return raw
    d = float(np.hypot(raw.dx, raw.dy))
    s = firm_threshold(d, 1e-3 * gate["translation_mm"]) / d if d > 0 else 0.0
    psi = 0.0
    if gate.get("yaw", True):
        psi = float(np.sign(raw.dpsi)) * firm_threshold(abs(raw.dpsi), np.radians(gate["yaw_deg"]))
    return Offset(s * raw.dx, s * raw.dy, psi)


@dataclass(frozen=True)
class WarpParams:
    r_in: float  # full warp inside this distance from the assembly centre (m)
    r_out: float  # no warp beyond
    dest_in: float | None = None  # no warp inside this distance from the destination anchor
    dest_out: float | None = None
    profile: str = "smoothstep"  # transition shape, a key of PROFILES


# Per task family, from the active arm's tool-point distance rc to the source centre over all 432 train+val
# demonstrations (validate_warp.py SOURCE/DEST_PHASES): the warp is full up to the largest rc in a source phase
# (approach/close/lift, Task4 push/retreat/settle) plus ~8 mm, and zero from the smallest rc in a destination phase
# (place/release, Task5 push) or at home minus ~11 mm:
#     source rc max   dest rc min   home rc min   ->  r_in   r_out (m)
#   task1   62            306           316            0.070  0.295
#   task2   24            350           355            0.035  0.340
#   task3   28            359           232            0.035  0.220
#   task4  166             -            355            0.175  0.345
#   task5   27            236           233            0.035  0.222
# Every destination row lies beyond r_out, so no destination gate is needed. The band spans the whole transit
# between the assembly and the destination/home, where the arm accelerates and decelerates, instead of a 100 mm ring
# crossed at peak speed: chunked demo replay at the four 15 mm / 10 deg corners, real/virtual joint second-difference
# p99 drops from 6.0/6.2/2.1/1.0/3.0x (12 -> 22 cm plus destination gate) to 1.5/1.1/1.1/1.0/1.1x for Task1..5
# (runs/spatial/validate/r.json), with source/destination rows still exactly w = 1 / w = 0. The C1 smoothstep beats
# the C2 smootherstep and a piecewise quadratic on that metric (runs/spatial/profile_sweep.json). With |d| <= 21 mm
# the widest slope 1.5 / 0.17 m keeps det J >= 0.8 (the yaw part is tangential and does not enter det J).
VARIANT_PARAMS: dict[str, WarpParams] = {
    "task1": WarpParams(0.070, 0.295),
    "task2": WarpParams(0.035, 0.340),
    "task3": WarpParams(0.035, 0.220),
    "task4": WarpParams(0.175, 0.345),
    "task5": WarpParams(0.035, 0.222),
}


def resolve_variant(task_variant: str, layouts: dict) -> str:
    """Layout key for a wire task_variant: exact name, else the one sharing its 'taskF-L' prefix."""
    if task_variant in layouts:
        return task_variant
    key = task_variant.split("_")[0]
    hits = [k for k in layouts if k.split("_")[0] == key]
    if len(hits) != 1:
        raise KeyError(f"no nominal layout for {task_variant}")
    return hits[0]


def family(task_variant: str) -> str:
    return task_variant.split("-")[0].split("_")[0]


PROFILES = {  # t in [0, 1] -> (value, d value / dt)
    "smoothstep": lambda t: (t * t * (3 - 2 * t), 6 * t * (1 - t)),  # C1, peak |f''| 6
    "smootherstep": lambda t: (t ** 3 * (t * (6 * t - 15) + 10), 30 * t * t * (1 - t) ** 2),  # C2, peak |f''| 5.8
    "quadratic": lambda t: (np.where(t < 0.5, 2 * t * t, 1 - 2 * (1 - t) ** 2), 4 * np.minimum(t, 1 - t)),  # C1, 4
}


def smoothstep(x: np.ndarray, lo: float, hi: float, profile: str = "smoothstep") -> tuple[np.ndarray, np.ndarray]:
    """Step 0 -> 1 between lo and hi (lo > hi flips it); returns value and derivative."""
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    f, df = PROFILES[profile](t)
    return f, np.where((t > 0) & (t < 1), df / (hi - lo), 0.0)


class Warp:
    """Planar warp W on tool-point xy plus the induced pose map. Identity when the offset is in the deadband."""

    def __init__(self, offset: Offset, center_xy, params: WarpParams, anchor_xy=None, *, deadband: bool = True):
        self.offset, self.params = offset, params
        self.c = np.asarray(center_xy, float)
        self.a = None if anchor_xy is None or params.dest_in is None else np.asarray(anchor_xy, float)
        self.d = np.array([offset.dx, offset.dy])
        self.psi = float(offset.dpsi)
        self.identity = offset.is_zero() or (deadband and offset.within_deadband())

    def weight(self, x: np.ndarray, *, grad: bool = False):
        """w(x) and optionally dw/dx for x [..., 2] in virtual coordinates."""
        v = x - self.c
        r = np.linalg.norm(v, axis=-1)
        s, ds = smoothstep(r, self.params.r_out, self.params.r_in, self.params.profile)  # 1 inside r_in
        if self.a is not None:
            u = x - self.a
            ra = np.linalg.norm(u, axis=-1)
            g, dg = smoothstep(ra, self.params.dest_in, self.params.dest_out, self.params.profile)
        else:
            u, ra, g, dg = None, None, np.ones_like(r), np.zeros_like(r)
        w = s * g
        if not grad:
            return w
        gw = (ds * g / np.maximum(r, 1e-12))[..., None] * v
        if u is not None:
            gw = gw + (s * dg / np.maximum(ra, 1e-12))[..., None] * u
        return w, gw

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x_virtual [..., 2] -> x_real."""
        if self.identity:
            return np.array(x, float)
        w = self.weight(x)
        ang = w * self.psi
        c, s = np.cos(ang), np.sin(ang)
        v = x - self.c
        rot = np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
        return self.c + rot + w[..., None] * self.d

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """dW/dx [..., 2, 2] (analytic)."""
        if self.identity:
            return np.broadcast_to(np.eye(2), np.shape(x)[:-1] + (2, 2)).copy()
        w, gw = self.weight(x, grad=True)
        ang = w * self.psi
        c, s = np.cos(ang), np.sin(ang)
        v = x - self.c
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        Rv_perp = np.stack([-s * v[..., 0] - c * v[..., 1], c * v[..., 0] - s * v[..., 1]], -1)  # dR/dang v
        return R + (self.psi * Rv_perp + self.d)[..., :, None] * gw[..., None, :]

    def inverse(self, y: np.ndarray, iters: int = 20, tol: float = 1e-10) -> np.ndarray:
        """x_virtual from x_real by Newton iterations (W is a C1 diffeomorphism, det J > 0)."""
        y = np.asarray(y, float)
        if self.identity:
            return y.copy()
        x = y.copy()
        for _ in range(iters):
            r = self.forward(x) - y
            if np.abs(r).max() < tol:
                break
            x = x - np.linalg.solve(self.jacobian(x), r[..., None])[..., 0]
        return x

    # --- poses: R [N,3,3], p [N,3] world, tool [3] in the eef frame
    def _rigid(self, w: np.ndarray, sign: float):
        return rot_z(sign * w * self.psi)

    def warp_pose(self, R, p, tool):
        """Virtual eef pose -> real eef pose. Returns (R', p', w)."""
        x = (p + R @ tool)[:, :2]
        if self.identity:
            return R, p, np.zeros(len(p))
        w = self.weight(x)
        Rz = self._rigid(w, 1.0)
        c3 = np.r_[self.c, 0.0]
        p2 = (Rz @ (p - c3)[..., None])[..., 0] + c3 + np.c_[w[:, None] * self.d, np.zeros(len(p))]
        return Rz @ R, p2, w

    def unwarp_pose(self, R, p, tool):
        """Real eef pose -> virtual eef pose. Returns (R, p, w)."""
        if self.identity:
            return R, p, np.zeros(len(p))
        y = (p + R @ tool)[:, :2]
        x = self.inverse(y)
        w = self.weight(x)
        Rz_inv = self._rigid(w, -1.0)
        c3 = np.r_[self.c, 0.0]
        p0 = p - np.c_[w[:, None] * self.d, np.zeros(len(p))] - c3
        return Rz_inv @ R, (Rz_inv @ p0[..., None])[..., 0] + c3, w

    def min_jacobian_det(self, half_width: float = 0.6, step: float = 0.002) -> float:
        """Numerical bijectivity check: min det(dW/dx) on a grid around the centre (local diffeo + proper)."""
        g = np.arange(-half_width, half_width + step / 2, step)
        X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + self.c
        return float(np.linalg.det(self.jacobian(X)).min())


def scene_geometry(task_variant: str, layouts: dict) -> tuple[np.ndarray, np.ndarray]:
    """(assembly centre, destination anchor) xy from the nominal layout of the variant."""
    blocks = layouts[task_variant]["blocks"]
    mov = [v for k, v in blocks.items() if k.startswith("source_") and not k.startswith("source_base")]
    dst = [v for k, v in blocks.items() if k.startswith("destination_base")]
    return np.mean(np.array(mov)[:, :2], 0), np.mean(np.array(dst)[:, :2], 0)


class JointWarp:
    """Joint-space wrapper of a Warp for both arms.

    command: IK(W(FK(q_v))), null-space reference q_v (the redundancy stays close to the demonstration).
    observe: IK(W^-1(FK(q_r))). Task space alone does not pin the 7-DoF redundancy, and the policy reads joint
    angles at 3-7 mrad per state token, so the inverse is anchored on the last commanded pairs: the reference
    is q_v[i] + (q_r - q_r[i]) for the commanded row i closest to the observation. With exact tracking that is
    q_v[i] itself, already on the target manifold, so the virtual state equals the virtual command exactly and
    the policy keeps seeing its nominal trajectory."""

    def __init__(self, warp: Warp, arms: dict[str, Arm] | None = None, *, max_ik_err_m: float = 2e-4,
                 polish_iters: int = 10, polish_gain: float = 0.5, polish_tol_dq: float = 1e-6):
        self.warp = warp
        self.arms = arms or load_arms()
        self.max_ik_err_m = max_ik_err_m
        # The IK stops as soon as the pose is reached, leaving a seed-dependent share of the null-space pull
        # undone; rows seeded differently then land on different points of the self-motion manifold, which shows
        # up as joint jitter. A fixed number of extra iterations without the early stop converges the redundancy
        # to the point closest to the reference, a smooth function of the target. Chunked (k=8) demo replay at
        # 15 mm / 10 deg, real/virtual second-difference p99 on Task4: 7.4x without, 1.02x with 10 or 20
        # iterations (runs/serve_ft/measure_polish{0,10,20}.json). The polish stops once no joint moves by more
        # than polish_tol_dq (1e-6 rad): served commands change by < 1e-6 rad and the polish cost drops ~30%.
        self.polish_iters, self.polish_gain, self.polish_tol_dq = polish_iters, polish_gain, polish_tol_dq
        self._cmd_delta = {a: np.zeros(7) for a in ARMS}
        self._obs_delta = {a: np.zeros(7) for a in ARMS}
        self._pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.stats = dict(ik_inexact=0, max_ik_err_m=0.0)

    def reset(self):
        for a in ARMS:
            self._cmd_delta[a][:] = 0
            self._obs_delta[a][:] = 0
        self._pairs.clear()

    def _map(self, q: np.ndarray, arm: str, forward: bool) -> tuple[np.ndarray, np.ndarray]:
        A = self.arms[arm]
        R, p = A.fk_world(q)
        R2, p2, w = (self.warp.warp_pose if forward else self.warp.unwarp_pose)(R, p, A.tool)
        out = q.copy()
        moving = w > 0
        if not moving.any():
            return out, w
        store = self._cmd_delta if forward else self._obs_delta
        ref = q[moving]
        seed = ref + store[arm]
        if not forward and arm in self._pairs:
            qv, qr = self._pairs[arm]
            i = np.argmin(np.linalg.norm(ref[:, None, :] - qr[None], axis=-1), axis=1)
            ref = seed = qv[i] + (ref - qr[i])
        qn, ep, _ = A.ik_world(R2[moving], p2[moving], seed, ref)
        bad = ep > self.max_ik_err_m
        if bad.any():  # second attempt from the input posture with more iterations
            idx = np.flatnonzero(moving)[bad]
            q2, ep2, _ = A.ik_world(R2[idx], p2[idx], q[idx], q[idx], iters=120)
            better = ep2 < ep[bad]
            sub = np.flatnonzero(bad)[better]
            qn[sub], ep[sub] = q2[better], ep2[better]
            bad = ep > self.max_ik_err_m
        if self.polish_iters:
            qp, epp, _ = A.ik_world(R2[moving], p2[moving], qn, ref, iters=self.polish_iters, tol_p=0.0, tol_r=0.0,
                                    null_gain=self.polish_gain, tol_dq=self.polish_tol_dq)
            keep = epp <= np.maximum(ep, self.max_ik_err_m)  # never trade task accuracy for the null space
            qn[keep], ep[keep] = qp[keep], epp[keep]
            bad = ep > self.max_ik_err_m
        # Unreachable targets (e.g. a push stroke that already ends at full extension, moved further away) keep
        # the damped least-squares solution: the closest reachable pose, continuous along the trajectory.
        # Falling back to the unwarped row would jump by the full offset.
        self.stats["ik_inexact"] += int(bad.sum())
        self.stats["max_ik_err_m"] = max(self.stats["max_ik_err_m"], float(ep.max()))
        out[moving] = qn
        store[arm] = (qn - q[moving])[-1]
        return out, w

    def real_command(self, q_virtual: np.ndarray, arm: str) -> np.ndarray:
        """q_virtual [N,7] -> q_real [N,7]: IK(W(FK(q)))."""
        qv = np.atleast_2d(np.asarray(q_virtual, float))
        qr = self._map(qv, arm, True)[0]
        self._pairs[arm] = (qv.copy(), qr.copy())
        return qr

    def virtual_state(self, q_real: np.ndarray, arm: str) -> np.ndarray:
        """q_real [N,7] -> q_virtual [N,7]: IK(W^-1(FK(q)))."""
        return self._map(np.atleast_2d(np.asarray(q_real, float)), arm, False)[0]


def load_layouts() -> dict:
    import json

    from jenga.kinematics import ROOT
    return json.loads((ROOT / "configs/cameras.json").read_text())["layouts"]["variants"]


class FrameTransform:
    """Per-episode transform between the real scene and the nominal scene the policy was trained on.

    Build once per session from the first observation (robot at home, assembly unoccluded):
        ft = FrameTransform.from_observation(obs)          # estimates the offset from the two external RGBs
        q16_virtual = ft.observe(q16_real)                  # model state input
        actions_real = ft.command(actions_virtual)          # [H,16] absolute joints + gripper widths (m)
    Layout of q16/actions: left joints 1..7, left gripper, back joints 1..7, back gripper. Grippers pass
    through. Stateful (IK seeds): one instance per session, discard on reset. Deterministic given the
    sequence of calls. `identity` is True when the gated estimate is zero or the estimate is not confident.

    `offset` is applied as given (hard 3 mm / 0.5 deg deadband unless deadband=False); from_observation applies
    the variant's calibrated gate (gate_offset) to the raw estimate first and keeps it in `raw_offset`.
    """

    def __init__(self, task_variant: str, offset: Offset, *, confidence: float = 1.0, arms=None, layouts=None,
                 params: WarpParams | None = None, deadband: bool = True, raw_offset: Offset | None = None):
        layouts = layouts or load_layouts()
        task_variant = resolve_variant(task_variant, layouts)
        self.task_variant, self.offset, self.confidence = task_variant, offset, confidence
        self.raw_offset = offset if raw_offset is None else raw_offset
        c, a = scene_geometry(task_variant, layouts)
        self.warp = Warp(offset, c, params or VARIANT_PARAMS[family(task_variant)], a, deadband=deadband)
        self.joint = JointWarp(self.warp, arms)

    @classmethod
    def from_observation(cls, observation: dict, *, estimator=None, **kw) -> FrameTransform:
        from jenga.offset import OffsetEstimator
        est = estimator or OffsetEstimator()
        r = est.estimate(observation["task_variant"], observation["images"])
        if not r.ok:
            return cls(observation["task_variant"], Offset(), confidence=r.confidence, raw_offset=r.offset, **kw)
        gate = est.gate_config(observation["task_variant"], r) or None
        if gate is None:  # uncalibrated layouts: legacy hard deadband
            return cls(observation["task_variant"], r.offset, confidence=r.confidence, **kw)
        return cls(observation["task_variant"], gate_offset(r.offset, gate), confidence=r.confidence,
                   raw_offset=r.offset, deadband=False, **kw)

    @property
    def identity(self) -> bool:
        return self.warp.identity

    def _apply(self, x: np.ndarray, forward: bool) -> np.ndarray:
        x = np.asarray(x)
        if self.identity:
            return x.copy()
        y = np.atleast_2d(x).astype(float).copy()
        for i, arm in enumerate(ARMS):
            sl = slice(8 * i, 8 * i + 7)
            fn = self.joint.real_command if forward else self.joint.virtual_state
            y[:, sl] = fn(y[:, sl], arm)
        return y.reshape(x.shape).astype(x.dtype if x.dtype.kind == "f" else np.float32)

    def observe(self, q16: np.ndarray) -> np.ndarray:
        return self._apply(q16, False)

    def command(self, actions: np.ndarray) -> np.ndarray:
        return self._apply(actions, True)
