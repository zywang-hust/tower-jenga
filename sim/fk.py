"""Forward kinematics of the two Nero arms read from the station USD joints (UsdPhysics revolute, prismatic, fixed).

For a joint from body0 to body1 the joint frame is localPos0/localRot0 in body0 and localPos1/localRot1 in body1;
the motion is a rotation about (revolute) or a translation along (prismatic) the joint axis in that frame:
    world_body1 = world_body0 @ A @ motion(q) @ B,   A = body0 -> joint frame,   B = joint frame -> body1,
both taken from the rest pose of the USD (where all joints read zero). Units: metres, radians.

qpos layout (dataset): left 7 joints + gripper width, back 7 joints + gripper width. A width w opens the two
finger joints to +w/2 and -w/2; the `gripper` joint (a link without visuals) follows w.
"""
from __future__ import annotations

import numpy as np

from render import ROBOTS, _rigid

ARMS = ("left", "back")
FINGERS = {"gripper": 1.0, "gripper_joint1": 0.5, "gripper_joint2": -0.5}  # joint value per unit width


def _frame(pos, rot, scale: float) -> np.ndarray:
    """Joint-local position (body units) + quaternion -> rigid 4x4 in the body frame (metres)."""
    m = np.eye(4)
    w, (x, y, z) = rot.GetReal(), rot.GetImaginary()
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = np.asarray(pos, float) * scale
    return m


class ArmFK:
    def __init__(self, usdz: str):
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.Open(usdz)
        mpu, xc = UsdGeom.GetStageMetersPerUnit(stage), UsdGeom.XformCache()
        self.rest, self.joints = {}, []  # joints: (arm, name, parent, child, A, B, kind) in parent-first order
        for arm in ARMS:
            pending = []
            for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOTS[arm])):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    self.rest[str(p.GetPath())] = _rigid(xc.GetLocalToWorldTransform(p), mpu)
                j = UsdPhysics.Joint(p)
                if j and j.GetBody0Rel().GetTargets() and j.GetBody1Rel().GetTargets():
                    b0, b1 = (str(r.GetTargets()[0]) for r in (j.GetBody0Rel(), j.GetBody1Rel()))
                    assert p.GetAttribute("physics:axis").Get() in ("Z", None)
                    # joint positions are authored in the body's own (scaled) units
                    s0, s1 = (np.linalg.norm(np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b)))[0, :3]) * mpu
                              for b in (b0, b1))
                    A = _frame(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get(), s0)
                    B = np.linalg.inv(_frame(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get(), s1))
                    kind = ("revolute" if p.IsA(UsdPhysics.RevoluteJoint)
                            else "prismatic" if p.IsA(UsdPhysics.PrismaticJoint) else "fixed")
                    pending.append((arm, p.GetName(), b0, b1, A, B, kind))
            children = {j[3] for j in pending}
            while pending:  # topological order: a joint is ready once its parent body is not a pending child
                ready = [j for j in pending if j[2] not in children]
                for j in ready:
                    pending.remove(j)
                    children.discard(j[3])
                self.joints += ready

    def link_poses(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        """qpos [16] -> world pose of every rigid body of both arms."""
        poses = dict(self.rest)
        q = {arm: np.asarray(qpos, float)[8 * i:8 * i + 8] for i, arm in enumerate(ARMS)}
        for arm, name, parent, child, A, B, kind in self.joints:
            motion = np.eye(4)
            if kind == "revolute":
                v = q[arm][int(name[5:]) - 1]
                motion[:2, :2] = [[np.cos(v), -np.sin(v)], [np.sin(v), np.cos(v)]]
            elif kind == "prismatic":
                motion[2, 3] = q[arm][7] * FINGERS[name]
            poses[child] = poses[parent] @ A @ motion @ B
        return poses
