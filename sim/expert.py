"""Privileged scripted expert for the local simulator: from any state (qpos16 + all block poses) it plans the
commands that finish the current operation, in the dataset action format (16 = per arm 7 joint targets in rad and
a gripper width in m, 30 Hz). Meant for DAgger labels and for checking that a state is recoverable.

Structure, reverse-engineered from the official generator's HDF5s:
- the arm per operation and the operation order come from target_mapping;
- grasp: the eef pose in the block frame is fixed per arm (rotation spread 0.005 rad over the training split), with the
  held block centre at the kinematics tool point; approach to 60 mm above it, vertical descent, close to 43.7 mm
  (the latched command; the evaluation controller follows the width directly), lift;
- place: carry above the goal, descend, open 5-9 mm above it so the block settles onto the layer, retreat upwards;
- an arm that has no operation parks at its start posture;
- pushdown (task4): closed gripper at the block end, push along the block axis away from the arm;
- insertion (task5): carry to 170 mm before the gap, slide 80 mm in, release, close the gripper, push home.

The expert has no clock. Each call works out the phase from geometry (is the block held, where is it relative to its
goal, where is the tool), plans one segment from the measured state and returns its rows. Placement targets come from
the measured held-block pose, so a block held off-centre or deeper still lands on its goal; before opening, the
block's measured goal error is corrected in a short extra move. IK is bounded to the demos' elbow branch; an arm
found folded onto its limits (policy states) unfolds first; a hand-off tolerance missed repeatedly widens.

Joint commands = IK solution + the static sag model (sim/expert_sag.py, sim/expert_fit.py): the PD settles short of
its command under gravity; the generator adds the same compensation.

Use:
    ex = Expert.from_hdf5(demo_path)            # or Expert(target_mapping, config, block_ids, home_q16)
    rows = ex.act(qpos16, blocks)               # closed loop: follows its current segment, replans when done
    seg = ex.plan(qpos16, blocks, fresh=True)   # DAgger label: stateless, plans from this state only
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from expert_sag import QPOS_RANGE, SAG  # noqa: E402
from jenga.kinematics import load_arms, matrix_to_rotvec, quat_xyzw_to_matrix, rot_z, rotvec_to_matrix  # noqa: E402

HZ = 30
ARMS = ("left", "back")
OPEN, HOLD = 0.1, 0.0437  # commanded widths: open, closed on a 50 mm block (demo latch value)
FIRM = 0.033  # held off-centre along its length: 2.4 N per finger lets the block pivot (45 deg measured), ~6 N holds
PUSH_W = 0.0  # task4 pushes with a closed gripper
# eef rotation in the block frame at grasp (rotation vector), medians over the training split
GRASP_ROT = {"left": (-1.300, 1.050, 1.050), "back": (1.177, 1.275, -1.268)}
CANON_YAW = {"left": 0.0, "back": -np.pi / 2}  # block yaw each arm handles in the demos
NOMINAL_Q = {  # demo grasp postures (motion_choices ik_continuity_seed_qpos), IK fallbacks
    "left": [-0.022, 0.906, -0.099, 0.871, 0.023, -0.075, 1.133],
    "back": [0.033, 0.824, -0.005, 0.912, -0.005, -0.054, 1.311]}
# joint limits of the station's Nero arms (USD physics:lower/upperLimit, both arms), less a margin
Q_LO = np.radians([-155.0, -99.69, -157.56, -57.87, -157.56, -41.83, -90.0]) + 0.03
Q_HI = np.radians([155.0, 99.69, 157.56, 122.61, 157.56, 54.43, 90.0]) - 0.03
# IK bounds per arm: j1, j3, j5 within the demos' range + 0.4 rad (+ 0.7 on a second try), j4 not below it - 0.7, so
# the IK stays on their elbow branch (it otherwise lands on the mirrored one, j1 + pi and j3 - pi, from offset states,
# where nothing near the tower is reachable). Postures beyond range + 0.8 count as folded (Expert.folded).
BRANCH = [0, 2, 4]
BOUNDS, WIDE = {}, {}
for _arm, (_lo, _hi) in QPOS_RANGE.items():
    for _out, _m in ((BOUNDS, 0.4), (WIDE, 0.7)):
        _l, _h = Q_LO.copy(), Q_HI.copy()
        _l[BRANCH] = np.maximum(_l[BRANCH], np.array(_lo)[BRANCH] - _m)
        _h[BRANCH] = np.minimum(_h[BRANCH], np.array(_hi)[BRANCH] + _m)
        _l[3] = max(_l[3], _lo[3] - 0.7)
        _out[_arm] = (_l, _h)
BLOCK = np.array([0.15, 0.05, 0.03])  # block size along its x y z
TIP = 0.009  # fingertips beyond the pinch point along the approach axis (centered_grasp_geometry: 0.187 - 0.178)
PRE_GRASP = 0.06  # source_approach_clearance
RELEASE_DZ = 0.006  # block centre above the goal when the gripper opens (demos 5-9 mm)
POS_TOL, YAW_TOL = 0.0025, np.radians(2.0)  # open only within this of the goal (score: 10 mm, 5 deg)
SPEED = dict(free=0.10, carry=0.08, lift=0.03, descend=0.03, place=0.02, push=0.03, joint=0.6)


def yaw_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rate_limit(q: np.ndarray, step: float = 0.02) -> np.ndarray:
    """Rows [n, 7] with any joint jump over `step` rad per tick spread over extra rows (where the IK changes its
    null-space solution between path pieces near a joint limit: a 0.06 rad jump in one tick jerks a held block)."""
    out = [q[:1]]
    for a, b in zip(q[:-1], q[1:]):
        m = int(np.ceil(np.abs(b - a).max() / step))
        out.append(a + np.linspace(0, 1, m + 1)[1:, None] * (b - a) if m > 1 else b[None])
    return np.concatenate(out)


def goal_like(Rg: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Goal rotation for a held block: a block turned upside down (knocked over) keeps that face down; of the two
    box-symmetric goals with z flipped, the nearer one. Turning it over in the hand is not attempted."""
    if Rb[2, 2] > 0:
        return Rg
    cands = [Rg @ np.diag(D) for D in ((1, -1, -1), (-1, 1, -1))]
    return min(cands, key=lambda R: np.linalg.norm(matrix_to_rotvec(Rb.T @ R)))


def lag_lead(arm: str, q0: np.ndarray, q7: np.ndarray) -> np.ndarray:
    """Joint lead [n, 7] that cancels the evaluation controller's tracking lag on rows q7 (row i reached after tick
    i+1, starting from q0): a joint driven by a position PD against its damping and the joint friction settles
    (kd + viscous) / kp * qdot + coulomb / kp behind a moving target, plus its command delay (LAG)."""
    tau, coul = LAG[arm]
    qd = np.diff(np.vstack([q0[None], q7]), axis=0) * HZ
    return tau * qd + coul * np.tanh(qd / 0.005)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def profile(n: int, ramp: int) -> np.ndarray:
    """Progress 0..1 over n rows (row i is the target after tick i+1): smooth ramps of `ramp` rows at both ends,
    constant speed between."""
    t = np.arange(1, n + 1, dtype=float)
    ramp = max(1, min(ramp, n // 2))
    v = np.minimum(1.0, np.minimum(t / ramp, (n + 1 - t) / ramp))
    v = 0.5 - 0.5 * np.cos(np.pi * np.clip(v, 0, 1))  # cosine ramps
    s = np.cumsum(v)
    return s / s[-1]


def slerp(R0: np.ndarray, R1: np.ndarray, s: np.ndarray) -> np.ndarray:
    w = matrix_to_rotvec(R0.T @ R1)
    return R0 @ rotvec_to_matrix(s[:, None] * w)


@dataclass
class Segment:
    name: str
    rows: np.ndarray  # [n, 16] commands
    reached: np.ndarray = field(default=None)  # [n, 16] predicted reached joints (for deviation checks)


class Expert:
    def __init__(self, target_mapping: list, config: dict, block_ids: list, home_q16, block_fixed=None,
                 initial_block_poses=None, arms=None):
        self.ops = target_mapping
        self.task = config.get("task", "")
        self.block_ids = block_ids
        self.home = np.asarray(home_q16, float)
        self.fixed = np.zeros(len(block_ids), bool) if block_fixed is None else np.asarray(block_fixed, bool)
        self.initial = None if initial_block_poses is None else np.asarray(initial_block_poses, float)
        self.arms = arms or load_arms()
        self.grasp_R = {a: rotvec_to_matrix(np.array(GRASP_ROT[a])) for a in ARMS}
        self.sag_W = {a: np.asarray(SAG[a]) for a in ARMS}
        self.tol = 2 * config.get("placement_m", 0.005), config.get("placement_degrees", 5.0)
        self.hold_w = INSERT_HOLD if self.task.startswith("tower_insertion") else HOLD
        # task5-3 under JENGA_PHYS_FIDELITY (sim/scene.py fidelity_setup): cuboid colliders at the authored heights
        cuboid = os.environ.get("JENGA_PHYS_FIDELITY", "0") == "1" and "cuboid_collision" in config.get("stage", "")
        self.cuboid = cuboid
        self.insert_dz, self.insert_seat = (INSERT_DZ_CUBOID, INSERT_SEAT_CUBOID) if cuboid else (INSERT_DZ, INSERT_SEAT)
        self.slide_to = INSERT_SLIDE_TO_CUBOID if cuboid else INSERT_SLIDE_TO
        self.segment: Segment | None = None
        self.cmd: np.ndarray | None = None  # last command row sent
        self.issued: np.ndarray | None = None  # predicted reached joints of the previous chunk's first row
        self.bias = np.zeros(3)  # closed-loop placement correction (world, m) for the current op
        self.bias_op = None
        self.grasp_offset = np.zeros(3)  # grasp point offset in the block frame (m): expert_rollout.py --grasp-offset
        self.log: list = []
        self.repeat, self.relax, self.lag = 0, 1, 0

    @classmethod
    def from_hdf5(cls, path, **kw):
        import h5py
        with h5py.File(path, "r") as f:
            a = f["sim"].attrs
            return cls(json.loads(a["target_mapping"]), json.loads(a["config_json"]), json.loads(a["block_ids"]),
                       f["observations/qpos"][0], json.loads(a["block_fixed_on_table"]),
                       json.loads(a["initial_block_poses"]), **kw)

    # ---------- kinematics helpers
    def sag(self, arm: str, q7: np.ndarray) -> np.ndarray:
        q7 = np.atleast_2d(q7)
        return np.c_[np.ones(len(q7)), np.sin(q7), np.cos(q7)] @ self.sag_W[arm]

    def eef(self, arm: str, q7) -> tuple[np.ndarray, np.ndarray]:
        R, p = self.arms[arm].fk_world(np.atleast_2d(q7))
        return R[0], p[0]

    def tool(self, arm: str, q7) -> np.ndarray:
        R, p = self.eef(arm, q7)
        return p + R @ self.arms[arm].tool

    def ik(self, arm: str, R: np.ndarray, p: np.ndarray, seed: np.ndarray, ref: np.ndarray | None = None):
        """IK for eef poses [n] (world), bounded to the demos' elbow branch; retries rows that miss."""
        A = self.arms[arm]
        R, p = R.reshape(-1, 3, 3), p.reshape(-1, 3)
        seed = np.broadcast_to(seed, (len(p), 7)).copy()
        lo, hi = BOUNDS[arm]
        lim = dict(limit_margin=None, bounds=(lo, hi))
        q, ep, er = A.ik_world(R, p, np.clip(seed, lo, hi), ref if ref is not None else seed, iters=60,
                               tol_p=1e-5, tol_r=1e-4, **lim)
        # real misses only (a few mrad of orientation is harmless): first wider bounds from the same seed, then from
        # the demo posture; the reference stays the seed so path rows keep their null-space continuity (a row solved
        # elsewhere jumps j1/j3/j5 and jerks a held block)
        nominal = np.broadcast_to(NOMINAL_Q[arm], (len(p), 7))
        for start, bounds in ((np.clip(seed, *WIDE[arm]), WIDE[arm]), (nominal, BOUNDS[arm]), (nominal, WIDE[arm])):
            miss = (ep > 2e-3) | (er > 3e-2)
            if not miss.any():
                break
            q2, ep2, er2 = A.ik_world(R, p, start, seed, iters=100, limit_margin=None, bounds=bounds)
            better = miss & (ep2 + 0.1 * er2 < ep + 0.1 * er)
            q[better], ep[better], er[better] = q2[better], ep2[better], er2[better]
        return q, ep, er

    def ik_tool(self, arm: str, R: np.ndarray, P: np.ndarray, seed: np.ndarray, ref: np.ndarray | None = None):
        """IK for eef rotations R [n] with the tool point at P [n]. Where R is out of reach (j7 on its limit, e.g. a
        grasp off the block centre), the tool point is kept at the best reachable rotation: missing the eef origin
        instead turns a 0.1 rad rotation miss into a 20 mm tool miss."""
        R, P = R.reshape(-1, 3, 3), P.reshape(-1, 3)
        q, ep, er = self.ik(arm, R, self.tool_pose_to_eef(arm, R, P), seed, ref)
        miss = np.flatnonzero(er > 1e-2)
        if len(miss):
            Ra = np.array([self.eef(arm, q[i])[0] for i in miss])
            r = q[miss] if ref is None else np.broadcast_to(ref, q.shape)[miss]
            q2, ep2, _ = self.ik(arm, Ra, self.tool_pose_to_eef(arm, Ra, P[miss]), q[miss], r)
            q[miss], ep[miss] = q2, ep2
        return q, ep, er

    def tool_pose_to_eef(self, arm: str, R_e: np.ndarray, tool_p: np.ndarray) -> np.ndarray:
        """eef position for an eef rotation and the world point where the tool point should be."""
        return tool_p - np.einsum("nij,j->ni", R_e.reshape(-1, 3, 3), self.arms[arm].tool)

    # ---------- state queries
    def held_relation(self, arm, q7, blk):
        """Block pose in the eef frame (R_be, p_be)."""
        R_e, p_e = self.eef(arm, q7)
        return R_e.T @ quat_xyzw_to_matrix(blk[3:]), R_e.T @ (blk[:3] - p_e)

    def held(self, arm: str, q16, blk) -> bool:
        k = ARMS.index(arm)
        w = q16[8 * k + 7]
        if not 0.035 < w < 0.056:  # same answer without the FK (2 s of every rollout's post-pass, docs/throughput.md 8.4)
            return False
        R_be, p_be = self.held_relation(arm, q16[8 * k:8 * k + 7], blk)
        tool_b = R_be.T @ (self.arms[arm].tool - p_be)  # tool point in the block frame: between its faces
        return 0.035 < w < 0.056 and bool(np.all(np.abs(tool_b) < BLOCK / 2 + 0.015))

    def at_goal(self, blk, op) -> bool:
        if "goal" not in op:
            return self.pushed_off(op, blk)
        g = np.asarray(op["goal"])
        # a block knocked upside down is placed so (goal_like) and counts as done: the scorer's quaternion test fails
        # it, but turning it over is beyond this expert and retrying loops release/close forever
        Rg = goal_like(quat_xyzw_to_matrix(g[3:]), quat_xyzw_to_matrix(blk[3:]))
        ang = np.degrees(np.linalg.norm(matrix_to_rotvec(Rg.T @ quat_xyzw_to_matrix(blk[3:]))))
        return np.linalg.norm(blk[:3] - g[:3]) <= self.tol[0] and ang <= self.tol[1]

    def pushed_off(self, op, blk) -> bool:
        start = self.initial[op["block_index"]] if self.initial is not None else op.get("_start")
        return bool(blk[2] < start[2] - 0.015 or np.linalg.norm(blk[:2] - start[:2]) > 0.025)

    def current_op(self, q16, blocks) -> int | None:
        for i, op in enumerate(self.ops):
            b = op["block_index"]
            done = self.at_goal(blocks[b], op)
            if done and "goal" in op and self.held(ARMS[op["arm"]], q16, blocks[b]):
                done = False  # still in the hand: release first
            if not done:
                return i
        return None

    def top_near(self, xy0, xy1, skip, blocks, radius=0.09) -> float:
        """Highest block top within `radius` of the segment xy0 -> xy1 (table: 0)."""
        d = xy1 - xy0
        L2 = max(float(d @ d), 1e-12)
        top = 0.0
        for i, b in enumerate(blocks):
            if i in skip:
                continue
            t = np.clip((b[:2] - xy0) @ d / L2, 0, 1)
            if np.linalg.norm(b[:2] - (xy0 + t * d)) < radius:
                top = max(top, b[2] + 0.5 * self.block_extent_z(b))
        return top

    @staticmethod
    def block_extent_z(b) -> float:
        R = quat_xyzw_to_matrix(b[3:])
        return float(np.abs(R[2]) @ BLOCK)

    # ---------- public API
    def act(self, qpos16, blocks, k: int = 4) -> np.ndarray:
        """Closed loop: k rows, continuing the current segment; replans when it is used up or the arm deviates."""
        q, blocks = np.asarray(qpos16, float), np.asarray(blocks, float)
        if self.segment is not None and len(self.segment.rows) and self.issued is not None:
            # the arm should be near the first row of the previous chunk by now; far off for 3 calls in a row means
            # blocked or pushed (a single call is tracking lag: replanning on it restarts the ramp-in from the
            # measured joints every chunk, and an arm unfolding from its limits never gets going)
            far = np.abs(self.issued - q)[[*range(7), *range(8, 15)]].max() > 0.1
            self.lag = self.lag + 1 if far else 0
            if self.lag >= 3:
                self.segment, self.lag = None, 0
        if self.segment is None or not len(self.segment.rows):
            self.segment = self.plan(q, blocks, fresh=self.cmd is None)
            # a hand-off tolerance missed again and again (the same segment recurring): widen it
            self.repeat = self.log[-8:].count(self.segment.name)
            self.relax = 1 + min(max(self.repeat - 1, 0), 3) * 0.5
            self.log.append(self.segment.name)
        rows = self.segment.rows[:k]
        self.issued = self.segment.reached[0].copy()
        self.segment.rows = self.segment.rows[len(rows):]
        self.segment.reached = self.segment.reached[len(rows):]
        if len(rows) < k:  # segment ended inside the chunk: hold, the next call sees the new state
            rows = np.vstack([rows, np.repeat(rows[-1:], k - len(rows), 0)])
        self.cmd = rows[-1].copy()
        return rows

    # ---------- planning
    def start(self, q16: np.ndarray, fresh: bool) -> np.ndarray:
        """Reached-joint start of a new plan: the last command minus sag when it is consistent with the measured
        state (continuity), else the measured joints; widths are the current commands."""
        s = q16.copy()
        if not fresh and self.cmd is not None:
            for k, arm in enumerate(ARMS):
                sl = slice(8 * k, 8 * k + 7)
                exp = self.cmd[sl] - self.sag(arm, self.cmd[sl])[0]
                if np.abs(exp - q16[sl]).max() < 0.02:
                    s[sl] = exp
            s[[7, 15]] = self.cmd[[7, 15]]
        else:
            for k in range(2):  # a width near the hold value is a closed command; else take it as commanded
                w = q16[8 * k + 7]
                s[8 * k + 7] = self.hold_w if 0.035 < w < 0.056 else (OPEN if w > 0.08 else w)
        return s

    def rows_from(self, s16: np.ndarray, arm: str, q7: np.ndarray | None, width: np.ndarray | None, lead=False):
        """Command rows: `arm` follows reached joints q7 [n,7] (+ sag) and widths [n]; the other arm holds.
        `lead`: the moving arm's rows also lead its joints by their tracking lag (lag_lead)."""
        n = len(q7) if q7 is not None else len(width)
        reached = np.repeat(s16[None], n, 0)
        k = ARMS.index(arm)
        if q7 is not None:
            reached[:, 8 * k:8 * k + 7] = q7
        if width is not None:
            reached[:, 8 * k + 7] = width
        rows = reached.copy()
        for j, a in enumerate(ARMS):
            rows[:, 8 * j:8 * j + 7] += self.sag(a, reached[:, 8 * j:8 * j + 7])
        if lead and q7 is not None:
            rows[:, 8 * k:8 * k + 7] += self.lead_no_horizontal(arm, q7, lag_lead(arm, s16[8 * k:8 * k + 7], q7))
        return rows, reached

    def lead_no_horizontal(self, arm: str, q7: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """The part of a joint lead dq [n,7] that moves the tool point vertically and rotates it, without moving it
        horizontally. The lead is there for the droop (vertical) of the slide; its horizontal part shifted the block
        sideways by about 1 mm when the slide started (t5_new3_28: 000015 op0 lat 1.0 -> 2.3 mm, 000005/10/43 op1
        -0.3 -> +0.5 mm), and a 50 mm block has about 1 mm per side in its 52 mm slot, so all four jammed at the
        tower face. Without it the horizontal path is the unled one that the regression passes."""
        A, eps = self.arms[arm], 1e-4

        def pose(q):
            R, p = A.fk_world(q)
            return R, p + np.einsum("nij,j->ni", R, A.tool)

        R0, p0 = pose(q7)
        J = np.zeros((len(q7), 6, 7))
        for j in range(7):
            R1, p1 = pose(q7 + eps * np.eye(7)[j])
            M = np.einsum("nij,nkj->nik", R1, R0)  # R1 R0^T
            J[:, :3, j] = (p1 - p0) / eps
            J[:, 3:, j] = 0.5 * np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0],
                                          M[:, 1, 0] - M[:, 0, 1]], 1) / eps
        dx = np.einsum("nij,nj->ni", J, dq)
        dx[:, :2] = 0.0
        return np.einsum("nij,nj->ni", np.linalg.pinv(J), dx)

    def cart_path(self, arm, s16, waypoints, speeds, width=None, ramp_s=0.3):
        """Tool-point path through world waypoints [(tool_p, R_eef)], each piece at its speed (m/s); IK per row
        seeded by joint interpolation. Returns (q7 rows [n,7], width rows or None)."""
        k = ARMS.index(arm)
        q = s16[8 * k:8 * k + 7]
        R0, p0 = self.eef(arm, q)
        cur = (p0 + R0 @ self.arms[arm].tool, R0)
        out = []
        for (p1, R1), v in zip(waypoints, speeds):
            dist = np.linalg.norm(p1 - cur[0])
            ang = np.linalg.norm(matrix_to_rotvec(cur[1].T @ R1))
            n = max(int(np.ceil(HZ * max(dist / v, ang / 0.5))) + int(2 * ramp_s * HZ * 0.5), 3)
            s = profile(n, int(ramp_s * HZ))
            P = cur[0] + s[:, None] * (p1 - cur[0])
            R = slerp(cur[1], R1, s)
            q_end, _, _ = self.ik_tool(arm, R[-1:], P[-1:], q)
            seed = q + s[:, None] * (q_end[0] - q)
            qs, ep, _ = self.ik_tool(arm, R, P, seed, seed)
            out.append(qs)
            q, cur = qs[-1], (p1, R1)
        q7 = rate_limit(np.concatenate(out))
        w = None
        if width is not None:
            w = np.full(len(q7), width[1])
            m = min(len(q7), int(0.5 * HZ))
            w[:m] = np.linspace(width[0], width[1], m + 1)[1:]
        return q7, w

    def joint_path(self, arm, s16, q_goal, speed=SPEED["joint"], width=None):
        k = ARMS.index(arm)
        q = s16[8 * k:8 * k + 7]
        n = max(int(np.ceil(HZ * np.abs(q_goal - q).max() / speed)) + 9, 6)
        s = profile(n, 9)
        q7 = q + s[:, None] * (q_goal - q)
        w = None
        if width is not None:
            w = np.full(n, width[1])
            m = min(n, int(0.5 * HZ))
            w[:m] = np.linspace(width[0], width[1], m + 1)[1:]
        return q7, w

    def plan(self, q16: np.ndarray, blocks: np.ndarray, fresh: bool) -> Segment:
        s16 = self.start(q16, fresh)
        i = self.current_op(q16, blocks)
        if i is not None and i != self.bias_op:
            self.bias, self.bias_op = np.zeros(3), i
        op = self.ops[i] if i is not None else None
        active = ARMS[op["arm"]] if op else None
        for arm in ARMS:  # folded onto its limits (seen in policy rollouts): unfold before anything else
            if self.folded(arm, q16):
                return self.unfold(arm, s16)
        # the arm without an operation goes home first (one arm moves at a time, as in the demos)
        for arm in ARMS:  # 0.15 rad: a parked arm nudged by the other one is left alone (re-parking it deadlocks)
            if arm != active and not self.is_home(arm, q16, 0.15 if active else 0.05):
                return self.park(arm, q16, s16, blocks)
        if op is None:
            return Segment("done", *self.rows_from(s16, "left", None, np.full(HZ, s16[7])))
        if "goal" not in op:
            return self.plan_push(op, q16, s16, blocks)
        if self.task.startswith("tower_insertion"):
            return self.plan_insert(op, q16, s16, blocks)
        return self.plan_pick_place(op, q16, s16, blocks)

    def is_home(self, arm, q16, tol=0.05) -> bool:
        k = ARMS.index(arm)
        return np.abs(q16[8 * k:8 * k + 7] - self.home[8 * k:8 * k + 7]).max() < tol

    def folded(self, arm, q16) -> bool:
        """j1, j3 or j5 more than 0.8 rad outside the demos' range, or j4 (elbow) more than 0.8 below it."""
        k = ARMS.index(arm)
        lo, hi = np.array(QPOS_RANGE[arm])
        q7 = q16[8 * k:8 * k + 7]
        out = (q7 < lo - 0.8) | (q7 > hi + 0.8)
        return bool(out[BRANCH].any() or q7[3] < lo[3] - 0.8)

    def unfold(self, arm, s16) -> Segment:
        """Joint move home, the elbow (j4) first and alone when it is out of range: moved together with the
        shoulder from a folded posture it jams (policy states with j1-j4 on their limits, measured)."""
        k = ARMS.index(arm)
        lo = np.array(QPOS_RANGE[arm][0])
        goal = self.home[8 * k:8 * k + 7].copy()
        j4 = s16[8 * k + 3]
        if j4 < lo[3] - 0.8:
            goal = s16[8 * k:8 * k + 7].copy()
            goal[3] = self.home[8 * k + 3]
        q7, _ = self.joint_path(arm, s16, goal)
        return Segment(f"unfold:{arm}", *self.rows_from(s16, arm, q7, None))

    def park(self, arm, q16, s16, blocks) -> Segment:
        k = ARMS.index(arm)
        w0 = s16[8 * k + 7]
        tp = self.tool(arm, q16[8 * k:8 * k + 7])
        top = self.top_near(tp[:2], tp[:2], set(), blocks, radius=0.12)
        if tp[2] < top + 0.08 and w0 < 0.08:  # holding or closed near blocks: open in place first
            return Segment(f"park-open:{arm}", *self.rows_from(s16, arm, None, self.ramp(w0, OPEN)))
        if tp[2] < top + 0.08:
            R, _ = self.eef(arm, s16[8 * k:8 * k + 7])
            q7, _ = self.cart_path(arm, s16, [(tp + [0, 0, top + 0.1 - tp[2]], R)], [SPEED["lift"] * 2])
            return Segment(f"park-up:{arm}", *self.rows_from(s16, arm, q7, None))
        q7, w = self.joint_path(arm, s16, self.home[8 * k:8 * k + 7], width=(w0, self.home[8 * k + 7]))
        return Segment(f"park:{arm}", *self.rows_from(s16, arm, q7, w))

    @staticmethod
    def ramp(w0, w1, seconds=0.6, hold=0.3):
        n, m = int(seconds * HZ), int(hold * HZ)
        return np.r_[np.linspace(w0, w1, n + 1)[1:], np.full(m, w1)]

    def grasp_pose(self, arm, blk):
        """eef rotation and tool target for grasping `blk` (of the two symmetric grasps, the one nearer the demos)."""
        yaw = yaw_of(quat_xyzw_to_matrix(blk[3:]))
        cands = [yaw, yaw + np.pi]
        yaw = min(cands, key=lambda y: abs(wrap(y - CANON_YAW[arm])))
        Rb = rot_z(yaw)
        return Rb @ self.grasp_R[arm], blk[:3] + Rb @ self.grasp_offset

    def approach_axis(self, arm) -> np.ndarray:
        """World-down at the demo grasp, in the eef frame (fingertips point along it)."""
        return -self.grasp_R[arm][2]

    def acquire(self, op, q16, s16, blocks, closed=HOLD, pre=PRE_GRASP) -> Segment | None:
        """Phases up to a closed grasp of op's block; None once it is held."""
        arm, b = ARMS[op["arm"]], op["block_index"]
        k = ARMS.index(arm)
        blk = blocks[b]
        if self.held(arm, q16, blk):
            return None
        q7 = q16[8 * k:8 * k + 7]
        w_meas, w_cmd = q16[8 * k + 7], s16[8 * k + 7]
        R_g, p_g = self.grasp_pose(arm, blk)
        tp = self.tool(arm, q7)
        R_e, _ = self.eef(arm, q7)
        line = np.linalg.norm(tp[:2] - p_g[:2])
        above = tp[2] - p_g[2]
        # 0.12 rad: a grasp off the block centre can put the demo rotation past the j7 limit (0.11 rad measured)
        aligned = np.linalg.norm(matrix_to_rotvec(R_e.T @ R_g)) < 0.12 * self.relax
        tol = 0.004 * self.relax
        if line < tol and abs(above) < tol and aligned:
            if w_cmd < 0.06 and w_meas < 0.035:  # closed on nothing: open again
                return Segment("regrip-open", *self.rows_from(s16, arm, None, self.ramp(w_cmd, OPEN)))
            return Segment("close", *self.rows_from(s16, arm, None, self.ramp(w_cmd, closed, 1.0, 0.5)))
        if line < 2 * tol and -tol < above < pre + 0.02 and aligned and w_meas > 0.08:
            q, _ = self.cart_path(arm, s16, [(p_g, R_g)], [SPEED["descend"]])
            return Segment("descend", *self.rows_from(s16, arm, q, None))
        top = self.top_near(tp[:2], tp[:2], {b}, blocks, radius=0.1)
        if tp[2] < max(top, blk[2]) + 0.05 and (line > 2 * tol or not aligned):  # low beside blocks: straight up
            up = max(top, blk[2]) + 0.07
            q, w = self.cart_path(arm, s16, [(np.r_[tp[:2], up], R_e)], [SPEED["lift"] * 2],
                                  width=(w_cmd, OPEN))
            return Segment("clear-up", *self.rows_from(s16, arm, q, w))
        pre_p = p_g + [0, 0, pre]
        q_goal, ep, _ = self.ik_tool(arm, R_g[None], pre_p[None], q7)
        q, w = self.joint_path(arm, s16, q_goal[0], width=(w_cmd, OPEN))
        return Segment("approach", *self.rows_from(s16, arm, q, w))

    def held_targets(self, arm, q7, blk, Rb_t, pb_t):
        """eef rotations/tool points that put the held block at poses (Rb_t [n,3,3], pb_t [n,3])."""
        R_be, p_be = self.held_relation(arm, q7, blk)
        R_e = Rb_t @ R_be.T
        tool_world = pb_t - np.einsum("nij,j->ni", R_e, p_be) + np.einsum("nij,j->ni", R_e, self.arms[arm].tool)
        return R_e, tool_world

    def plan_pick_place(self, op, q16, s16, blocks) -> Segment:
        seg = self.acquire(op, q16, s16, blocks)
        if seg is not None:
            return seg
        arm, b = ARMS[op["arm"]], op["block_index"]
        k = ARMS.index(arm)
        q7, blk = q16[8 * k:8 * k + 7], blocks[b]
        g = np.asarray(op["goal"], float)
        Rg = goal_like(quat_xyzw_to_matrix(g[3:]), quat_xyzw_to_matrix(blk[3:]))
        err = g[:3] - blk[:3]
        exy, dz = np.linalg.norm(err[:2]), -err[2]
        yaw_err = abs(wrap(yaw_of(Rg) - yaw_of(quat_xyzw_to_matrix(blk[3:]))))
        # fingertips below the block bottom (grasped deeper than the demos): open higher so they clear the layer
        _, p_be = self.held_relation(arm, q7, blk)
        a = self.approach_axis(arm)
        protrude = TIP - (p_be - self.arms[arm].tool) @ a - BLOCK[2] / 2
        rel_dz = max(RELEASE_DZ, protrude + 0.005)
        R_be, _ = self.held_relation(arm, q7, blk)
        lever = abs((p_be - self.arms[arm].tool) @ R_be[:, 0])  # grasp offset along the block's long axis
        w_cmd = s16[8 * k + 7]
        w_hold = min(w_cmd, FIRM if lever > 0.006 else self.hold_w)

        def to_block_path(points, speeds, rot_from=0):
            """Held-block waypoints (world centre; from waypoint `rot_from` the block turns to the goal yaw). A tilt in
            the hand is kept, it settles flat on release; straightening it costs wrist range the demos already use."""
            Rb0 = quat_xyzw_to_matrix(blk[3:])
            Rt = rot_z(wrap(yaw_of(Rg) - yaw_of(Rb0))) @ Rb0
            wps = []
            for j, p in enumerate(points):
                Rb = Rt if j >= rot_from else Rb0
                R_e, tw = self.held_targets(arm, q7, blk, Rb[None], np.asarray(p)[None])
                wps.append((tw[0], R_e[0]))
            return self.cart_path(arm, s16, wps, speeds)[0]

        if exy < POS_TOL and yaw_err < YAW_TOL and dz < rel_dz + 0.004:
            self.bias = np.zeros(3) if self.bias_op is None else self.bias
            return Segment("release", *self.rows_from(s16, arm, None, self.ramp(w_cmd, OPEN)))
        if exy < 0.006 and yaw_err < 0.05 and dz < rel_dz + 0.004:  # at height but off: correct and remember
            self.bias[:2] += 0.7 * err[:2]
        if exy < 0.006 and yaw_err < 0.05 and dz < 0.07:
            tgt = g[:3] + [0, 0, rel_dz] + self.bias
            q = to_block_path([tgt], [SPEED["place"]])
            return Segment("place", *self.rows_from(s16, arm, q, np.full(len(q), w_hold)))
        if exy < 0.04 and yaw_err < 0.2 and -0.005 < dz < 0.07:  # near but off (where policies stop): over, then down
            tgt = np.r_[g[:2] + self.bias[:2], max(blk[2], g[2] + rel_dz + 0.015)]
            q = to_block_path([tgt], [SPEED["descend"]])
            return Segment("align", *self.rows_from(s16, arm, q, np.full(len(q), w_hold)))
        # carry: up clear of everything near the route, over the goal, down to 50 mm above it
        route = self.top_near(blk[:2], g[:2], {b}, blocks)
        z_clear = route + self.block_extent_z(blk) / 2 + 0.02
        z_lift = max(blk[2], z_clear)
        z_over = max(z_clear, g[2] + 0.05)
        pts, spd, rot = [], [], 1
        if z_lift > blk[2] + 0.002:
            pts.append(np.r_[blk[:2], z_lift])
            spd.append(SPEED["lift"])
        else:
            rot = 0
        pts.append(np.r_[g[:2] + self.bias[:2], max(z_lift, z_over)])
        spd.append(SPEED["carry"])
        if max(z_lift, z_over) > g[2] + 0.051:
            pts.append(np.r_[g[:2] + self.bias[:2], g[2] + 0.05])
            spd.append(SPEED["descend"])
        q = to_block_path(pts, spd, rot_from=rot)
        return Segment("carry", *self.rows_from(s16, arm, q, np.full(len(q), w_hold)))

    # ---------- task4 pushdown, task5 push home
    def push_segment(self, arm, q16, s16, blocks, skip, start, end, R_push, close_w, name, speed=None) -> Segment:
        """Closed gripper on the line start -> end (tool point): close, go above start, descend, push."""
        k = ARMS.index(arm)
        q7, w_cmd = q16[8 * k:8 * k + 7], s16[8 * k + 7]
        tp = self.tool(arm, q7)
        R_cur, _ = self.eef(arm, q7)
        d = (end - start) / np.linalg.norm(end - start)
        R_push = self.reachable_pitch(arm, R_push, np.cross([0, 0, 1.0], d), [start + [0, 0, PUSH_PRE], start, end])
        aligned = np.linalg.norm(matrix_to_rotvec(R_cur.T @ R_push)) < 0.06 * self.relax
        rel = tp - start
        along, off = rel @ d, rel - (rel @ d) * d
        if w_cmd > close_w + 0.005:
            return Segment(f"{name}-close", *self.rows_from(s16, arm, None, self.ramp(w_cmd, close_w, 0.6, 0.2)))
        if aligned and np.linalg.norm(off) < 0.005 * self.relax and -0.01 < along < np.linalg.norm(end - start) - 0.003:
            q, _ = self.cart_path(arm, s16, [(end, R_push)], [speed or SPEED["push"]])
            return Segment(f"{name}", *self.rows_from(s16, arm, q, None))
        if aligned and np.linalg.norm(tp[:2] - start[:2]) < 0.006 * self.relax and \
                -0.005 * self.relax < tp[2] - start[2] < PUSH_PRE + 0.02:
            q, _ = self.cart_path(arm, s16, [(start, R_push)], [SPEED["descend"]])
            return Segment(f"{name}-descend", *self.rows_from(s16, arm, q, None))
        top = self.top_near(tp[:2], tp[:2], skip, blocks, radius=0.1)
        if tp[2] < top + 0.04 and np.linalg.norm(tp[:2] - start[:2]) > 0.02:  # low and must travel: up first
            q, _ = self.cart_path(arm, s16, [(np.r_[tp[:2], top + 0.07], R_cur)], [SPEED["lift"] * 2])
            return Segment(f"{name}-up", *self.rows_from(s16, arm, q, None))
        pre = start + [0, 0, PUSH_PRE]
        # a free joint move: of the solutions from the current posture and the demo posture take the one nearest the
        # demo posture (seeded from the post-lift posture alone the IK can land on j1 2.11 at the bounds' edge, from
        # where the descend stops 11 mm short forever: task4-2 000073)
        sols = [self.ik_tool(arm, R_push[None], pre[None], s) for s in (q7, NOMINAL_Q[arm])]
        qg = min([q[0] for q, ep, _ in sols if ep[0] < 2e-3] or [sols[0][0][0]], key=lambda q: np.linalg.norm(q - NOMINAL_Q[arm]))
        q, _ = self.joint_path(arm, s16, qg)
        return Segment(f"{name}-approach", *self.rows_from(s16, arm, q, None))

    def reachable_pitch(self, arm, R, axis, points):
        """R, or R pitched about `axis` (world) by up to 0.3 rad if R puts the tool points [m] beyond the joint
        limits (blocks pushed from offset towers: the elbow, j4, runs out first)."""
        P = np.array(points)
        for th in (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3):
            Rt = np.broadcast_to(rotvec_to_matrix(axis * th) @ R, (len(P), 3, 3))
            _, ep, er = self.ik(arm, Rt, self.tool_pose_to_eef(arm, Rt, P), NOMINAL_Q[arm])
            if ep.max() < 1e-3 and er.max() < 1e-2:
                return Rt[0]
        return R

    def plan_push(self, op, q16, s16, blocks) -> Segment:
        """task4: push the block along its axis, away from the arm, until it falls off the tower."""
        arm, b = ARMS[op["arm"]], op["block_index"]
        blk = blocks[b]
        Rb = quat_xyzw_to_matrix(blk[3:])
        d = np.r_[Rb[:2, 0], 0.0]
        d /= np.linalg.norm(d)
        if d[:2] @ (blk[:2] - self.arms[arm].base_t[:2]) < 0:
            d = -d
        F = np.c_[d, np.cross([0, 0, 1.0], d), [0, 0, 1.0]]
        start = blk[:3] + F @ np.array(PUSH_START[arm])
        end = blk[:3] + F @ np.array(PUSH_END[arm])
        R_push = F @ rotvec_to_matrix(np.array(PUSH_ROT[arm]))
        return self.push_segment(arm, q16, s16, blocks, {b}, start, end, R_push, PUSH_W, "push")

    # ---------- task5 insertion
    def plan_insert(self, op, q16, s16, blocks) -> Segment:
        """task5: carry to the entry before the gap, line up, slide in, release, push home with closed fingers."""
        arm, b = ARMS[op["arm"]], op["block_index"]
        k = ARMS.index(arm)
        q7, blk = q16[8 * k:8 * k + 7], blocks[b]
        g = np.asarray(op["goal"], float)
        if self.cuboid and INSERT_TRACK_CUBOID and self.initial is not None:
            # the dynamic tower moves (the previous insertion nudges it): aim at where its slot is now, the side
            # blocks' midpoint moved by as much as it has since the start
            sides = [i for i, n in enumerate(self.block_ids) if n.startswith(f"destination_l{op.get('destination_layer')}_side_")]
            if sides and INSERT_TRACK_CUBOID == "2" and len(sides) == 2:
                # the slot itself: centre line between the side blocks' inner faces, along their mean long axis
                g = g.copy()
                Rs = [quat_xyzw_to_matrix(blocks[i, 3:]) for i in sides]
                mid = blocks[sides, :3].mean(0)
                x = sum(R[:, 0] * np.sign(R[:, 0] @ Rs[0][:, 0]) for R in Rs)
                x = np.r_[x[:2], 0] / np.linalg.norm(x[:2])
                y = np.array([-x[1], x[0], 0.0])
                inner = [(blocks[i, :3] - mid) @ y - np.sign((blocks[i, :3] - mid) @ y) * abs(R[:, 1] @ y) * BLOCK[1] / 2
                         for i, R in zip(sides, Rs)]
                c = mid + y * np.mean(inner)
                g[:2] = (c + x * ((g[:3] - c) @ x))[:2]
                Rg0 = quat_xyzw_to_matrix(g[3:])
                if Rg0[:, 0] @ x < 0:
                    x = -x
                dyaw = np.arctan2(x[1], x[0]) - np.arctan2(Rg0[1, 0], Rg0[0, 0])
                (x1, y1, z1, w1), sz, cz = g[3:], np.sin(dyaw / 2), np.cos(dyaw / 2)  # rot_z(dyaw) applied on the left
                g[3:] = [cz * x1 - sz * y1, cz * y1 + sz * x1, cz * z1 + sz * w1, cz * w1 - sz * z1]
            elif sides:
                g = g.copy()
                g[:2] += blocks[sides, :2].mean(0) - self.initial[sides, :2].mean(0)
        Rg = goal_like(quat_xyzw_to_matrix(g[3:]), quat_xyzw_to_matrix(blk[3:]))
        d = np.r_[Rg[:2, 0], 0.0]
        d /= np.linalg.norm(d)
        if d[:2] @ (g[:2] - self.arms[arm].base_t[:2]) < 0:
            d = -d  # insertion direction: away from the arm
        F = np.c_[d, np.cross([0, 0, 1.0], d), [0, 0, 1.0]]
        a, lat, h = F.T @ (blk[:3] - g[:3])
        held = self.held(arm, q16, blk)
        if self.cuboid and self.log and self.log[-1] == "insert-release":
            held = False  # let go to INSERT_RELEASE_W_CUBOID: the fingers are still around the block
        w_cmd = s16[8 * k + 7]
        if not held and abs(lat) < 0.015 and -0.015 < h < 0.006 and a > -INSERT_ENTRY + 0.02:
            # in the gap: push it home with closed fingers behind its back face, at the goal's height (the line ends
            # on the goal: it takes out the few mm the release shifts it sideways, and levels a block whose outer
            # end tipped down on release); lifting it out again catches the roof
            pdz = INSERT_PUSH_DZ_CUBOID if self.cuboid else INSERT_PUSH_DZ
            start = np.r_[blk[:2], g[2]] + F @ np.array([-INSERT_PUSH_BACK - INSERT_PUSH_MARGIN, 0, pdz])
            end = g[:3] + F @ np.array([-INSERT_PUSH_BACK, 0, pdz])
            tp = self.tool(arm, q7)
            if w_cmd > INSERT_PUSH_W + 0.005 and np.linalg.norm(tp - blk[:3]) < 0.06:  # just let go: back out, up
                R_cur, _ = self.eef(arm, q7)
                out = tp + F @ np.array([-0.10, 0, 0.04])
                q, w = self.cart_path(arm, s16, [(out, R_cur)], [SPEED["descend"]], width=(w_cmd, w_cmd if self.cuboid else OPEN))
                return Segment("insert-backout", *self.rows_from(s16, arm, q, w))
            yaw_c = min((yaw_of(Rg), yaw_of(Rg) + np.pi), key=lambda y: abs(wrap(y - CANON_YAW[arm])))
            R_push = rot_z(yaw_c) @ self.grasp_R[arm]
            return self.push_segment(arm, q16, s16, blocks, {b}, start, end, R_push, INSERT_PUSH_W, "insert-push",
                                     INSERT_PUSH_SPEED_CUBOID if self.cuboid else None)
        if not held:
            seg = self.acquire(op, q16, s16, blocks, closed=INSERT_HOLD)
            if seg is not None:
                return seg
        Rb0 = quat_xyzw_to_matrix(blk[3:])

        def path(points, speeds, rot_from=0):
            wps = []
            for j, p in enumerate(points):
                R_e, tw = self.held_targets(arm, q7, blk, (Rg if j >= rot_from else Rb0)[None], np.asarray(p)[None])
                wps.append((tw[0], R_e[0]))
            return self.cart_path(arm, s16, wps, speeds)[0]

        entry = g[:3] - d * INSERT_ENTRY
        dz, seat = self.insert_dz, self.insert_seat
        if abs(lat) < 0.006 and -INSERT_H_LOW < h < 0.006 and a > -INSERT_ENTRY - 0.01:
            if a >= -self.slide_to - 0.004:
                # seat it on the floor before letting go, once: a block that does not follow the hand down (the
                # privileged expert sees the true pose) is let go after the press instead of pressed forever
                if h > dz - seat + 0.0005 and (not self.log or self.log[-1] != "insert-seat"):
                    q = path([np.r_[blk[:2], g[2] + dz - seat]], [SPEED["place"]])
                    return Segment("insert-seat", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd)))
                if self.cuboid:
                    widths = self.ramp(w_cmd, INSERT_RELEASE_W_CUBOID, seconds=INSERT_RELEASE_S_CUBOID)
                else:
                    widths = self.ramp(w_cmd, OPEN)
                return Segment("insert-release", *self.rows_from(s16, arm, None, widths))
            tol = INSERT_ALIGN_CUBOID if self.cuboid else INSERT_ALIGN
            lined_up = abs(lat) < tol[0] and abs(h - dz) < tol[1] or a > -INSERT_ENTRY + 0.01
            if self.cuboid and INSERT_PRE_CUBOID and lined_up and a < -INSERT_PRE_CUBOID + 0.002:
                # the hand drifts sideways along the slide (left arm: up to 1.8 mm over 20 mm) and the slot leaves
                # 1 mm a side: stop just short of the side blocks' ends and line up again
                pre = g[:3] - d * INSERT_PRE_CUBOID + [0, 0, dz]
                n_re = next((i for i, s in enumerate(reversed(self.log)) if s != "insert-realign"), len(self.log))
                if a < -INSERT_PRE_CUBOID - 0.003:
                    q = path([pre + self.bias], [SPEED["place"]])
                    return Segment("insert-slide-pre", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd), lead=True))
                if abs(lat) > tol[0] and n_re < 4:
                    self.bias -= 0.8 * (F @ np.array([0, lat, 0]))
                    q = path([pre + self.bias], [SPEED["place"]])
                    return Segment("insert-realign", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd)))
            if lined_up:
                tgt = g[:3] - d * self.slide_to + self.bias + [0, 0, dz]
                q = path([tgt], [SPEED["place"]])
                return Segment("insert-slide", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd), lead=True))
            self.bias -= 0.8 * (F @ np.array([0, lat, h - dz]))  # line up with the gap before entering it
            q = path([entry + self.bias + [0, 0, dz]], [SPEED["place"]])
            return Segment("insert-align", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd)))
        if np.linalg.norm(blk[:2] - entry[:2]) < 0.006 and -0.006 < blk[2] - g[2] < 0.13:
            q = path([entry + self.bias + [0, 0, dz]], [SPEED["place"]])
            return Segment("insert-descend", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd)))
        route = self.top_near(blk[:2], entry[:2], {b}, blocks)
        z = max(route + BLOCK[2] / 2 + 0.03, g[2] + INSERT_ABOVE, blk[2])
        pts, spd, rot = [np.r_[blk[:2], z], entry + [0, 0, INSERT_ABOVE]], [SPEED["lift"] * 1.5, SPEED["carry"]], 1
        if z > g[2] + INSERT_ABOVE + 0.001:
            pts[1] = np.r_[entry[:2], z]
            pts.append(entry + [0, 0, INSERT_ABOVE])
            spd.append(SPEED["descend"])
        q = path(pts, spd, rot_from=rot)
        return Segment("insert-carry", *self.rows_from(s16, arm, q, np.full(len(q), w_cmd)))


# pushdown (task4), measured on the demos: tool point relative to the block in the push frame
# F = [push dir, z x dir, z] at push start / end, and the eef rotation in F (rotation vector); medians over train
PUSH_START = {"left": (-0.099, -0.001, 0.008), "back": (-0.106, 0.0027, 0.009)}
PUSH_END = {"left": (0.136, -0.001, 0.007), "back": (0.125, 0.0027, 0.004)}
PUSH_ROT = {"left": (-1.501, 0.610, 0.649), "back": (-1.559, 0.826, 0.838)}
PUSH_PRE = 0.05  # descend onto the push line from this far above
# insertion (task5): the block is carried to INSERT_ENTRY before the goal along the insertion axis, INSERT_ABOVE
# over it, lowered, slid in to INSERT_SLIDE_TO before the goal and released; the closed fingers (20 mm) then push
# it home from INSERT_PUSH_BACK behind its centre.
# The fingers stop at the side blocks (centre -85 mm at most), so the block is let go with its centre 15-25 mm outside
# the base edge, pinched between the base edge and the roof; it tips on release unless its height is within +-0.3 mm
# (docs/research.md §6.1). Holding it 40 mm behind its centre (let go 25 mm inside the edge) does not work: it pivots
# 35-41 deg in the fingers on the lift, at the 30 mm and at the saturated 0 mm grip command alike.
INSERT_ENTRY, INSERT_ABOVE, INSERT_SLIDE_TO = 0.17, 0.10, 0.09
# grasp command 30 mm (config gripper_closed_width_m), not the demos' 0: under the evaluation controller a 0 m command
# saturates both finger drives, nothing centres the block and it slides along the jaw axis (18 mm measured)
INSERT_HOLD, INSERT_PUSH_W = 0.03, 0.02
INSERT_PUSH_BACK, INSERT_PUSH_DZ = 0.087, -0.001
# the push starts this much further behind the block, a margin for a back face known only to within a few mm; the
# line still ends on the goal
INSERT_PUSH_MARGIN = 0.01
# line up with the gap this closely (lateral, height) before sliding in: the slot leaves 1 mm sideways and the block
# bottom at most 1 mm below the base top (1.5/1.0 mm tolerances let a 1 mm position error jam)
INSERT_ALIGN = (0.00075, 0.0003)
# slide height vs the goal centre (goal = the layer's resting height, train mean), then seat: in task5-2 the slot has
# no vertical clearance, the roof is free and the base fixed, so the block slides in at the base top and is pressed
# down onto the base before it is let go. Released without seating, its true height at the slide decides everything
# (+0.1 mm held, -0.4 tipped over the base edge, -0.9 jammed on it). -0.2 was tuned locally and the left arm's
# 0.1 mm lag came on top, so with the lag led (LAG) -0.3 keeps task5-2 at the height it slid at before the lead. In task5-3 the same height applies:
# sliding at the side blocks' height (+1, where the demos slide) jammed the layer-1 block under the layer-2 block,
# which rests 1.9 mm low and tilted locally (3/13, runs/simdev/fleet/t5_new2_28).
INSERT_DZ, INSERT_SEAT = -0.0003, 0.002
# task5-3 under JENGA_PHYS_FIDELITY: every block is a full cuboid (no 0.85 mm bottom relief of the convex hull) and
# the base and layer-1 side tops sit exactly at the slot floors, so a block slid at -0.3 hits the floor's front edge.
# The 32 mm side blocks leave 2 mm over a 30 mm block: slide above the floor and let go there (no seat).
INSERT_DZ_CUBOID, INSERT_SEAT_CUBOID = 0.0005, 0.0
INSERT_SLIDE_TO_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_SLIDE_TO_CUBOID_MM", 90)) / 1000
# Measured on 8 train episodes of task5-3 on the dynamic tower (runs/simdev/phys/expert/<run>, 0/8 before):
# - release: the hand holds the block off-centre and a slow open lets one finger shove it ~5 mm sideways; open in one
#   step (0.034 s) and wide (100 mm) instead (st100: the tower no longer moves on release);
# - line-up: stop 158 mm before the goal, short of the side blocks' ends, line up again to 0.25 mm (the left arm drifts
#   up to 1.8 mm sideways over the slide; the slot leaves 1 mm a side) and slide from there (pre158t25: 4/8);
# - push home slowly, 1 mm above the goal centre: 30 mm/s drags the tower, 50 mm/s wrecks it (v10p1: 6/8);
# - aim at the slot as it is (INSERT_TRACK_CUBOID "2"): the centre line between the side blocks' inner faces and
#   their mean yaw, which the first insertion turns by ~0.1 deg; the layer-1 block entering 0.25 deg off caught the
#   ends of the layer-1 side blocks (t2: 8/8; "1", the side blocks' midpoint shift alone, v10p1: 6/8).
INSERT_RELEASE_W_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_RELEASE_W_CUBOID_MM", 100)) / 1000
INSERT_RELEASE_S_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_RELEASE_S_CUBOID", 0.034))
INSERT_ALIGN_CUBOID = (float(os.environ.get("JENGA_DIAG_INSERT_ALIGN_LAT_CUBOID_MM", 0.25)) / 1000, INSERT_ALIGN[1])
INSERT_PRE_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_PRE_CUBOID_MM", 158)) / 1000
INSERT_PUSH_SPEED_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_PUSH_SPEED_CUBOID", 0.01))
INSERT_PUSH_DZ_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_PUSH_DZ_CUBOID_MM", 1)) / 1000
INSERT_TRACK_CUBOID = os.environ.get("JENGA_DIAG_INSERT_TRACK_CUBOID", "2")
INSERT_TRACK_CUBOID = "" if INSERT_TRACK_CUBOID == "0" else INSERT_TRACK_CUBOID
INSERT_DZ += float(os.environ.get("JENGA_DIAG_INSERT_DZ_MM", 0)) / 1000  # diagnosis only: stress tests
INSERT_DZ_CUBOID += float(os.environ.get("JENGA_DIAG_INSERT_DZ_CUBOID_MM", 0)) / 1000
INSERT_SEAT_CUBOID = float(os.environ.get("JENGA_DIAG_INSERT_SEAT_CUBOID_MM", INSERT_SEAT_CUBOID * 1000)) / 1000
# Joint tracking lag of the evaluation controller per arm and joint (calibration runtime_readback, identical in every
# demo): (kd + viscous friction) / kp in s plus the arm's command delay (left 5 ms, back 10 ms), and dynamic Coulomb
# friction / kp in rad. Uncompensated, the back arm's j4 trails during a 20 mm/s slide and the hand runs 2-3 mm
# under its path (the model predicts the locally measured lag joint by joint, docs/research.md 6.1.7a), so the block
# reaches the far side of the layer below too low.
LAG = {"left": (np.array([0.0676, 0.0013, 0.0018, 0.0121, 0.0576, 0.0388, 0.0437]) + 0.005,
                np.array([3.05, 0.07, 0.10, 0.15, 0.35, 1.76, 1.89]) * 1e-3),
       "back": (np.array([0.0453, 0.0443, 0.0270, 0.1063, 0.0748, 0.0498, 0.1416]) + 0.010,
                np.array([3.85, 0.33, 0.01, 3.33, 0.08, 2.00, 0.28]) * 1e-3)}
# On a task5-3 tower that yields, the back arm's lag (LAG) can leave the held block several mm below the goal while it
# slides in; a tight lower bound would send it back to insert-carry, which lifts the block out through the tower.
# Below the goal is inside the slot, so the lower bound is wide.
INSERT_H_LOW = 0.015
