"""Kinematics of the two AgileX Nero arms, identified from the dataset (no URDF is available locally).

Each arm is a generic serial chain: for joint i the fixed transform F_i (translation + rotation vector, 6
numbers) is followed by a rotation Rz(q_i); a final fixed transform F_7 maps to the recorded end-effector
frame (`observations/eef`, a flange-like frame, not the fingertip). Parameters, the arm-base -> world
transform (world = frame of `sim/block_poses`) and the held-block offset (tool point in the eef frame) live
in configs/nero_kinematics.json, produced by scripts/fit_kinematics.py.

All functions are batched over a leading axis. Units: metres and radians.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/nero_kinematics.json"
ARMS = ("left", "back")


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rodrigues; v [..., 3] -> [..., 3, 3]."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1)[..., None, None]
    k = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Inverse of rotvec_to_matrix for rotations away from pi (enough for small IK errors)."""
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.where(th < 1e-9, 0.5, th / np.maximum(2 * np.sin(th), 1e-12))
    return w * s[..., None]


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot_z(angle: np.ndarray) -> np.ndarray:
    a = np.asarray(angle, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1
    return R


@dataclass
class Chain:
    """Serial chain with 7 revolute joints about the local z axis. params: [8, 6] = (t xyz, rotvec)."""

    params: np.ndarray
    _R: np.ndarray = field(init=False, repr=False)
    _t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.params = np.asarray(self.params, float).reshape(8, 6)
        self._R = rotvec_to_matrix(self.params[:, 3:])
        self._t = self.params[:, :3].copy()

    def fk(self, q: np.ndarray, *, frames: bool = False):
        """q [N, 7] -> (R [N,3,3], p [N,3]) of the eef in the arm-base frame.
        With frames=True also returns joint axes z [N,7,3] and origins o [N,7,3] (for the Jacobian)."""
        q = np.atleast_2d(np.asarray(q, float))
        n = len(q)
        A = self._R[:7] @ rot_z(q)  # [N,7,3,3]: fixed rotation of joint i, then its revolute Rz(q_i)
        z = np.empty((n, 7, 3))
        o = np.empty((n, 7, 3))
        R = A[:, 0]
        p = np.broadcast_to(self._t[0], (n, 3))
        z[:, 0], o[:, 0] = R[:, :, 2], p  # Rz(q) leaves the local z axis in place
        for i in range(1, 7):
            p = p + R @ self._t[i]
            R = R @ A[:, i]
            z[:, i], o[:, i] = R[:, :, 2], p
        p = p + R @ self._t[7]
        R = R @ self._R[7]
        return (R, p, z, o) if frames else (R, p)

    def jacobian(self, q: np.ndarray, point: np.ndarray | None = None):
        """Geometric Jacobian [N, 6, 7] (linear rows first) of `point` (eef-frame offset, default origin)."""
        R, p, z, o = self.fk(q, frames=True)
        if point is not None:
            p = p + R @ np.asarray(point, float)
        r = p[:, None] - o
        J = np.empty((len(p), 6, 7))
        J[:, 0] = z[..., 1] * r[..., 2] - z[..., 2] * r[..., 1]  # z x (p - o), written out (np.cross is slow)
        J[:, 1] = z[..., 2] * r[..., 0] - z[..., 0] * r[..., 2]
        J[:, 2] = z[..., 0] * r[..., 1] - z[..., 1] * r[..., 0]
        J[:, 3:] = z.transpose(0, 2, 1)
        return R, p, J

    def ik(self, R_target: np.ndarray, p_target: np.ndarray, q_seed: np.ndarray, q_ref: np.ndarray | None = None,
           *, iters: int = 30, tol_p: float = 1e-6, tol_r: float = 1e-5, damping: float = 1e-6,
           null_gain: float = 0.2, bounds: tuple[np.ndarray, np.ndarray] | None = None, tol_dq: float = 0.0):
        """Damped least squares with a null-space pull toward q_ref (7-DoF redundancy). Batched.
        Stops early when the pose is within (tol_p, tol_r), or when no joint moved by more than tol_dq in the last
        iteration (converged null space; 0 disables that test).
        `bounds` (lo, hi) projects every iterate into the box, which keeps unreachable targets from driving the
        arm through a singularity (e.g. flipping the elbow) on its way to the closest reachable pose.
        Returns (q [N,7], position error [N] m, rotation error [N] rad)."""
        R_t = np.asarray(R_target, float).reshape(-1, 3, 3)
        p_t = np.asarray(p_target, float).reshape(-1, 3)
        q = np.array(q_seed, float).reshape(-1, 7)
        q_ref = q.copy() if q_ref is None else np.asarray(q_ref, float).reshape(-1, 7)
        eye6, eye7 = damping * np.eye(6), np.eye(7)
        for _ in range(iters):
            R, p, J = self.jacobian(q)
            e = np.concatenate([p_t - p, matrix_to_rotvec(R_t @ np.swapaxes(R, 1, 2))], 1)
            ep, er = np.linalg.norm(e[:, :3], axis=1), np.linalg.norm(e[:, 3:], axis=1)
            if ep.max() < tol_p and er.max() < tol_r:
                break
            Jt = np.swapaxes(J, 1, 2)
            Jp = Jt @ np.linalg.inv(J @ Jt + eye6)  # [N,7,6]
            q_prev = q
            q = q + (Jp @ e[:, :, None])[:, :, 0] + null_gain * ((eye7 - Jp @ J) @ (q_ref - q)[:, :, None])[:, :, 0]
            if bounds is not None:
                q = np.clip(q, bounds[0], bounds[1])
            if tol_dq > 0 and np.abs(q - q_prev).max() < tol_dq:
                break
        R, p = self.fk(q)
        return q, np.linalg.norm(p_t - p, axis=1), np.linalg.norm(matrix_to_rotvec(R_t @ np.swapaxes(R, 1, 2)), axis=1)


@dataclass
class Arm:
    """One arm: chain in its base frame, base -> world rigid transform, tool point (held-block centre) in eef."""

    chain: Chain
    base_R: np.ndarray
    base_t: np.ndarray
    tool: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray

    def fk_world(self, q: np.ndarray):
        R, p = self.chain.fk(q)
        return self.base_R @ R, p @ self.base_R.T + self.base_t

    def ik_world(self, R_world, p_world, q_seed, q_ref=None, *, limit_margin: float | None = 0.5, **kw):
        """IK for a world-frame eef pose. Iterates are kept within the observed joint range widened by
        `limit_margin` rad (the real limits are not published; None disables the box)."""
        R_b = self.base_R.T @ np.asarray(R_world, float)
        p_b = (np.asarray(p_world, float).reshape(-1, 3) - self.base_t) @ self.base_R
        if limit_margin is not None:
            kw["bounds"] = (self.q_lo - limit_margin, self.q_hi + limit_margin)
        return self.chain.ik(R_b, p_b, q_seed, q_ref, **kw)

    def tool_world(self, R_world: np.ndarray, p_world: np.ndarray) -> np.ndarray:
        return p_world + R_world @ self.tool


def load_arms(path: Path | str = DEFAULT_CONFIG) -> dict[str, Arm]:
    cfg = json.loads(Path(path).read_text())
    arms = {}
    for name in ARMS:
        a = cfg["arms"][name]
        arms[name] = Arm(Chain(np.array(a["chain_params"])), np.array(a["base_to_world"]["R"]),
                         np.array(a["base_to_world"]["t"]), np.array(a["tool_point_in_eef"]),
                         np.array(a["observed_joint_range"]["lo"]), np.array(a["observed_joint_range"]["hi"]))
    return arms
