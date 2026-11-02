"""The official TOWER station in Isaac PhysX, physics only (this H800 cannot start Isaac's RTX renderer).

Everything that sets the dynamics comes from the episode's HDF5 `sim` attributes, nothing is tuned here:
120 Hz physics, the runtime joint readback (kp, kd, armature, friction, effort and velocity limits; stage units, mm),
per-arm command delays under a zero-order hold, the generator's gripper controller (rate limits, contact latch at
the hold load), initial block poses. Units at the interface are the dataset's:
qpos16 = [left j1..j7, left gripper width m, back j1..j7, back gripper width m]; block poses x y z qx qy qz qw in metres.
Must run inside the Isaac app (sim/isaac.sh); `Station` expects SimulationApp to be up with sim/physics_only.kit.
"""
import json
import os

import carb
import numpy as np
import omni.physics.tensors
import omni.usd
from isaacsim.core.api import SimulationContext
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

ROOT = "/PNP_DUALD1200_STATION"
ARMS = {"left": "StandA_Robot", "back": "StandB_Robot"}  # dataset arm order; calibration arm_mapping
ARM_DOFS = [f"joint{i}" for i in range(1, 8)]
FINGERS = ("gripper_joint1", "gripper_joint2")  # +-width/2 each, mm
MM = 1000.0
PHYSICS_HZ, ACTION_HZ = 120, 30


def robot_path(arm: str) -> str:
    return f"{ROOT}/StationModel/InstalledRobots/{ARMS[arm]}/NeroRobotWithWristCamera"


def fidelity_enabled() -> bool:
    """JENGA_PHYS_FIDELITY=1: build the blocks as the episode's attributes describe them (block_dimensions_m,
    block_mass_kg; the cuboid collider of the task5-3 stage overlay) and leave the task5-3 tower dynamic. Off by
    default: the pipeline then runs exactly as before (published 30 mm convex-hull blocks, frozen task5-3 tower)."""
    return os.environ.get("JENGA_PHYS_FIDELITY", "0") == "1"


def fidelity_setup(sim_attrs) -> dict:
    """The block geometry the official generator built, from the episode's own attributes. The task5 side blocks are
    32 mm tall ("taller solid side blocks", layout_metadata_json side_dimensions_m) and 0.11264 kg, against the
    published 30 mm, 0.1056 kg reusable block; the task5-3 stage overlay swaps the block collider for a full cuboid
    (provenance stage_compatibility), under which the official stacks rest at their authored heights to 0.01 mm. A
    UsdGeom.Cube of the mesh bounds does that here; the mesh's own boundingCube approximation sinks 0.4 mm per
    contact and its convex hull 0.6-1.2 mm (the latter as in the official baseline stage). Episodes start after the
    generator's reset settle (config reset_settle_seconds), as the recorded first frame does."""
    config = json.loads(sim_attrs["config_json"])
    n = len(json.loads(sim_attrs["block_ids"]))
    cuboid = "cuboid_collision" in config["stage"]
    # task1-4 record null dimensions and masses: the published block, left as authored
    return dict(block_dims=json.loads(sim_attrs["block_dimensions_m"]),
                block_mass=json.loads(sim_attrs["block_mass_kg"]),
                block_collider="cube_prim" if cuboid else None,
                # (contactOffset, restOffset) in mm on the cuboids: with restOffset 0 the inserted block's bottom runs
                # coplanar with the layer below and catches its edge, dragging the dynamic tower (replay of 8 train
                # demos: 0/8 within 7 mm); 0.05 mm per shape (0.1 mm per contact) 7/8, 0.02 5/8, 0.1 4/8
                block_offsets=(2.0, 0.05) if cuboid else None,
                frozen=[False] * n, reset_settle_s=float(config["reset_settle_seconds"]))


def episode_setup(sim_attrs, fidelity: bool | None = None) -> dict:
    """The parts of an HDF5 `sim.attrs` the station needs, as plain Python. `fidelity` (default: JENGA_PHYS_FIDELITY)
    adds fidelity_setup."""
    prov = json.loads(sim_attrs["provenance_json"])
    calib = prov["calibration"]
    assert calib["physics_hz"] == PHYSICS_HZ and calib["action_hz"] == ACTION_HZ, calib["physics_hz"]
    assert calib["command_interpolation"] == "zero_order_hold"
    setup = dict(
        readback={arm: {k: np.asarray(v)[0] for k, v in calib["runtime_readback"][arm]["readback"].items()}
                  for arm in ARMS},
        dof_names={arm: calib["runtime_readback"][arm]["dof_names"] for arm in ARMS},
        delays_s={arm: float(max(calib["command_delays_seconds"][arm])) for arm in ARMS},
        block_ids=json.loads(sim_attrs["block_ids"]),
        block_fixed=np.asarray(json.loads(sim_attrs["block_fixed_on_table"]), bool),
        initial_block_poses=np.asarray(json.loads(sim_attrs["initial_block_poses"]), float),
        gripper=gripper_setup(json.loads(sim_attrs["config_json"]), json.loads(sim_attrs["motion_choices_json"])),
        # task5-3 runs on the station under an unpublished overlay ("changes only the reusable block collider to a
        # full cuboid", provenance stage_compatibility) under which the tower rests at its authored heights with
        # 1-2 mm between the layers and never moves (46/78/109 mm to 0.01 mm over every demo); with the published
        # convex-hull colliders it settles 2-4 mm and the layer-2 slot shrinks below a block. The tower is held
        # kinematic in its place; the inserted blocks stay dynamic. JENGA_PHYS_FIDELITY=1 replaces this with the
        # overlay's geometry (fidelity_setup), under which the dynamic tower rests where the official one does.
        frozen=[b.startswith("destination_l") and "cuboid_collision" in json.loads(sim_attrs["config_json"])["stage"]
                for b in json.loads(sim_attrs["block_ids"])],
    )
    if fidelity_enabled() if fidelity is None else fidelity:
        setup.update(fidelity_setup(sim_attrs))
    return setup


def gripper_setup(config: dict, motion_choices: list) -> dict:
    """Gripper controller constants: the config's speeds, overridden by the episode's contact_settle_timing."""
    contact = next((m for m in motion_choices if m.get("kind") == "contact_settle_timing"), {})
    return dict(close=config["gripper_close_speed_m_s"], open=config["gripper_open_speed_m_s"],
                contact_width=contact.get("contact_zone_command_width_m", config["gripper_contact_width_m"]),
                contact_close=contact.get("contact_zone_close_speed_m_s", config["gripper_contact_close_speed_m_s"]),
                hold_n=contact.get("hold_load_per_finger_n", config["grasp_hold_force_n"]))


class Gripper:
    """The generator's width controller for one gripper, as its event log describes it: the commanded width is
    rate limited (slower inside the contact zone), and once both fingers stall on an object the drive target is
    latched at measured width - 2 * hold / kp, so each finger presses with the hold load instead of the full drive.
    Opening past the measured width releases the latch."""

    def __init__(self, close, open, contact_width, contact_close, hold_n, kp_n_per_m, width):
        self.speeds, self.contact_width = (close, open, contact_close), contact_width
        self.gap = 2 * hold_n / kp_n_per_m
        self.command, self.latched = width, None

    def step(self, desired: float, measured: float, speed: float, dt: float) -> float:
        """Drive target width (m) for the next physics step; `speed` is the measured width rate, m/s."""
        close, open_, contact_close = self.speeds
        rate = open_ if desired > self.command else contact_close if self.command <= self.contact_width else close
        self.command += float(np.clip(desired - self.command, -rate * dt, rate * dt))
        if self.command > measured:
            self.latched = None
        elif self.latched is None and measured - self.command > self.gap and abs(speed) < 5e-3:
            self.latched = measured - self.gap
        return self.command if self.latched is None else max(self.command, self.latched)


def _block_prims(stage) -> tuple[list, list]:
    """Station block prims split into kinematic foundations and free blocks, in authored order."""
    prims = [p for p in stage.GetPrimAtPath(f"{ROOT}/ReplayTaskBlocks").GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    kinematic = [p for p in prims if p.GetAttribute("physics:kinematicEnabled").Get()]
    return kinematic, [p for p in prims if p not in kinematic]


def _set_collider(prim, kind: str) -> None:
    """Replace the block's mesh collider. 'cuboid': the mesh's boundingCube approximation (a PhysX box of the mesh
    bounds); 'cube_hull': an 8-vertex cuboid mesh of the same bounds as a convex hull; 'cube_prim': a UsdGeom.Cube.
    The two added colliders take the mesh's physics material and the mesh stops colliding."""
    mesh = next(c for c in Usd.PrimRange(prim) if c.IsA(UsdGeom.Mesh) and c.HasAPI(UsdPhysics.CollisionAPI))
    if kind in ("cuboid", "boundingCube", "convexHull", "convexDecomposition"):
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set({"cuboid": "boundingCube"}.get(kind, kind))
        return
    rng = UsdGeom.Mesh(mesh).ComputeLocalBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    path = mesh.GetPath().GetParentPath().AppendChild("CuboidCollider")
    if kind == "cube_hull":
        geom = UsdGeom.Mesh.Define(prim.GetStage(), path)
        corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        geom.CreatePointsAttr([Gf.Vec3f(*map(float, c)) for c in corners])
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        geom.CreateFaceVertexCountsAttr([4] * 6)
        geom.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr().Set("convexHull")
    elif kind == "cube_prim":
        geom = UsdGeom.Cube.Define(prim.GetStage(), path)
        geom.CreateSizeAttr(1.0)
        geom.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2)))
        geom.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    else:
        raise ValueError(kind)
    UsdGeom.Imageable(geom).CreatePurposeAttr("guide")
    mat = UsdShade.MaterialBindingAPI(mesh).GetDirectBinding("physics").GetMaterial()
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Set(False)


def _set_attrs(stage, scene_attrs, body_attrs, blocks) -> None:
    """Diagnosis: author physxScene:* attributes on every physics scene and physxRigidBody:* on the blocks."""
    def put(prim, name, value):
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, {bool: Sdf.ValueTypeNames.Bool, int: Sdf.ValueTypeNames.Int,
                                               float: Sdf.ValueTypeNames.Float, str: Sdf.ValueTypeNames.Token}[type(value)])
        attr.Set(value)
    for prim in stage.Traverse() if scene_attrs else []:
        if prim.IsA(UsdPhysics.Scene):
            PhysxSchema.PhysxSceneAPI.Apply(prim)
            for k, v in (scene_attrs or {}).items():
                put(prim, k, v)
    for prim in blocks if body_attrs else []:
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        for k, v in body_attrs.items():
            put(prim, k, v)


def _place(prim, pose_m: np.ndarray, dims_m=None) -> None:
    """Move a block prim to `pose_m` (world, metres, xyzw), keeping its scale (the geometry is authored in metres
    under a x1000 scale), or rescaling its local axes to `dims_m` (block frame x y z, metres). The geometry is
    centred on the prim origin and the parents are identity, so the pose is the translate and orient op values."""
    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeUntransformedBound(prim).ComputeAlignedRange()
    center = bound.GetMidpoint()
    parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert center.GetLength() < 1e-6 and parent == Gf.Matrix4d(1), (prim.GetPath(), center, parent)
    xf = UsdGeom.Xformable(prim)
    scale = Gf.Transform(xf.GetLocalTransformation()).GetScale()
    if dims_m is not None:
        size = np.array(bound.GetSize())
        assert np.allclose(size, [0.15, 0.05, 0.03], atol=1e-4) or np.allclose(size, dims_m, atol=1e-4), (prim.GetPath(), size)
        scale = Gf.Vec3d(*(np.array(scale) * np.asarray(dims_m) / size))
    rotate = Gf.Matrix4d().SetRotate(Gf.Quatd(float(pose_m[6]), *map(float, pose_m[3:6])))
    xf.ClearXformOpOrder()  # USD row vectors: scale, then rotate, then translate
    xf.AddTransformOp().Set(Gf.Matrix4d().SetScale(scale) * rotate * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*(pose_m[:3] * MM))))


class _NumpyView:
    """A tensor-API view that takes and returns numpy whatever the frontend. On the GPU pipeline the frontend is
    torch: states live on the device, DOF and body properties on the host."""

    HOST = ("stiffness", "damping", "armature", "friction", "max_", "limit", "mass", "inertia", "material")

    def __init__(self, view, device: str):
        self.view, self.device = view, device

    def __getattr__(self, name):
        attr = getattr(self.view, name)
        if self.device == "cpu" or not callable(attr):
            return attr
        import torch
        device = "cpu" if any(h in name for h in self.HOST) else self.device

        def call(*args):
            out = attr(*(torch.as_tensor(np.asarray(a), device=device) for a in args))
            return out.cpu().numpy() if isinstance(out, torch.Tensor) else out
        return call


class Station:
    """One environment. reset() then tick(target16) per 30 Hz row; state() reads qpos16 and block poses."""

    def __init__(self, usdz: str, setup: dict, device: str = "cpu"):
        self.setup = setup
        omni.usd.get_context().open_stage(usdz)
        stage = omni.usd.get_context().get_stage()
        kinematic, free = _block_prims(stage)
        fixed = setup["block_fixed"]
        assert fixed.sum() <= len(kinematic) and (~fixed).sum() <= len(free), (fixed, len(kinematic), len(free))
        pools = {True: iter(kinematic), False: iter(free)}
        self.blocks = [next(pools[bool(f)]) for f in fixed]
        n = len(fixed)
        dims, masses = setup.get("block_dims") or [None] * n, setup.get("block_mass") or [None] * n
        for i, (prim, pose, frozen) in enumerate(zip(self.blocks, setup["initial_block_poses"], setup.get("frozen", [False] * n))):
            _place(prim, pose, dims[i])
            if masses[i] is not None:
                UsdPhysics.MassAPI(prim).GetMassAttr().Set(float(masses[i]))
            if setup.get("block_collider"):
                _set_collider(prim, setup["block_collider"])
            if setup.get("block_offsets"):  # (contactOffset, restOffset) in mm: diagnosis of resting penetration
                for c in Usd.PrimRange(prim):
                    if c.HasAPI(UsdPhysics.CollisionAPI):
                        api = PhysxSchema.PhysxCollisionAPI.Apply(c)
                        api.CreateContactOffsetAttr(float(setup["block_offsets"][0]))
                        api.CreateRestOffsetAttr(float(setup["block_offsets"][1]))
            if frozen:
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
            friction = setup.get("block_friction")  # (static, dynamic) override: the task layers' value is not published
            for prefix, st, dy in setup.get("role_friction") or []:  # diagnosis: per block-id prefix
                if setup["block_ids"][i].startswith(prefix):
                    friction = (st, dy)
            if friction:
                mat = prim.GetPrimAtPath("PhysicsMaterials/Wood")
                mat.GetAttribute("physics:staticFriction").Set(friction[0])
                mat.GetAttribute("physics:dynamicFriction").Set(friction[1])
            if setup.get("block_mass_scale"):  # robustness probe: the authored physics:mass (0.1056 kg) scaled
                m = UsdPhysics.MassAPI(prim).GetMassAttr()
                m.Set(float(m.Get()) * setup["block_mass_scale"])
        for prim in kinematic + free:
            if prim not in self.blocks:
                prim.SetActive(False)
        self.block_paths = [str(p.GetPath()) for p in self.blocks]

        settings = carb.settings.get_settings()  # state is read through the tensor API: skip the USD writeback
        settings.set("/physics/updateToUsd", False)
        settings.set("/physics/updateVelocitiesToUsd", False)
        self.sim = SimulationContext(physics_dt=1 / PHYSICS_HZ, rendering_dt=1 / ACTION_HZ, stage_units_in_meters=0.001,
                                     backend="numpy" if device == "cpu" else "torch", device=device)
        _set_attrs(stage, setup.get("scene_attrs"), setup.get("body_attrs"), self.blocks)
        self.sim.initialize_physics()
        self.sim.play()
        view = omni.physics.tensors.create_simulation_view("numpy" if device == "cpu" else "torch")
        self.view = view
        self.arms = {arm: _NumpyView(view.create_articulation_view(f"{robot_path(arm)}/root_joint"), device)
                     for arm in ARMS}
        free_paths = [p for p, f in zip(self.block_paths, fixed) if not f]
        self.free = _NumpyView(view.create_rigid_body_view(free_paths), device) if free_paths else None
        self.dof = {}
        for arm, art in self.arms.items():
            names = [p.rsplit("/", 1)[-1] for p in art.dof_paths[0]]
            assert names == setup["dof_names"][arm], (names, setup["dof_names"][arm])
            self.dof[arm] = names
            rb = setup["readback"][arm]
            idx = np.arange(1, dtype=np.int32)
            art.set_dof_stiffnesses(rb["kp"][None].astype(np.float32), idx)
            art.set_dof_dampings(rb["kd"][None].astype(np.float32), idx)
            art.set_dof_armatures(rb["armatures"][None].astype(np.float32), idx)
            art.set_dof_friction_properties(rb["friction"][None].astype(np.float32), idx)
            art.set_dof_max_forces(rb["max_efforts"][None].astype(np.float32), idx)
            art.set_dof_max_velocities(rb["max_velocities"][None].astype(np.float32), idx)
        self.delay_steps = {arm: int(np.ceil(d * PHYSICS_HZ - 1e-9)) for arm, d in setup["delays_s"].items()}
        self.finger_kp = {arm: float(setup["readback"][arm]["kp"][self.dof[arm].index(FINGERS[0])]) for arm in ARMS}
        self.joints, self.grippers = {}, {}

    def readback(self) -> dict:
        """What the engine now holds, in the calibration's layout, to verify against `setup['readback']`."""
        return {arm: dict(kp=a.get_dof_stiffnesses()[0], kd=a.get_dof_dampings()[0], armatures=a.get_dof_armatures()[0],
                          friction=a.get_dof_friction_properties()[0], max_efforts=a.get_dof_max_forces()[0],
                          max_velocities=a.get_dof_max_velocities()[0]) for arm, a in self.arms.items()}

    def _dof_vector(self, arm: str, joints: np.ndarray, width_m: float) -> np.ndarray:
        """Joints (rad) and gripper width (m) -> the articulation's DOF vector (gripper and fingers in mm)."""
        w = width_m * MM
        value = dict(zip(ARM_DOFS, joints), gripper=w, gripper_joint1=w / 2, gripper_joint2=-w / 2)
        return np.array([value[n] for n in self.dof[arm]], np.float32)

    def _width(self, arm: str, values: np.ndarray) -> float:
        i, j = (self.dof[arm].index(f) for f in FINGERS)
        return float(values[i] - values[j]) / MM

    def reset(self, qpos16: np.ndarray) -> None:
        idx = np.arange(1, dtype=np.int32)
        for k, (arm, art) in enumerate(self.arms.items()):
            q8 = qpos16[8 * k:8 * k + 8]
            q = self._dof_vector(arm, q8[:7], q8[7])
            art.set_dof_positions(q[None], idx)
            art.set_dof_velocities(np.zeros_like(q)[None], idx)
            art.set_dof_position_targets(q[None], idx)
            self.joints[arm] = q8[:7]
            g = self.setup["gripper"]  # None: the drive target is the commanded width
            self.grippers[arm] = g and Gripper(**g, kp_n_per_m=self.finger_kp[arm], width=float(q8[7]))

    def settle(self, qpos16: np.ndarray, seconds: float) -> None:
        """The generator's reset settle: arms held at `qpos16` while the blocks come to rest, then the arms are put
        back exactly (velocities zero); the blocks keep their settled state."""
        self.reset(qpos16)
        for _ in range(int(round(seconds * ACTION_HZ))):
            self.tick(qpos16)
        self.reset(qpos16)

    def tick(self, target16: np.ndarray) -> None:
        """Advance 1/30 s: each arm's joints switch to `target16` after its command delay; the grippers run their
        controller every physics step (gripper_delay_seconds is 0)."""
        idx, dt = np.arange(1, dtype=np.int32), 1 / PHYSICS_HZ
        for step in range(PHYSICS_HZ // ACTION_HZ):
            for k, (arm, art) in enumerate(self.arms.items()):
                q8 = target16[8 * k:8 * k + 8]
                if step == self.delay_steps[arm]:
                    self.joints[arm] = q8[:7]
                width = float(q8[7])
                if self.grippers[arm]:
                    pos, vel = art.get_dof_positions()[0], art.get_dof_velocities()[0]
                    width = self.grippers[arm].step(width, self._width(arm, pos), self._width(arm, vel), dt)
                art.set_dof_position_targets(self._dof_vector(arm, self.joints[arm], width)[None], idx)
            self.sim.step(render=False)

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        q = []
        for arm, art in self.arms.items():
            pos = art.get_dof_positions()[0]
            q += [pos[self.dof[arm].index(n)] for n in ARM_DOFS] + [self._width(arm, pos)]
        return np.array(q), self.block_poses()

    def block_poses(self) -> np.ndarray:
        """All episode blocks (x y z qx qy qz qw, metres); the kinematic foundations stay at their initial poses."""
        poses = self.setup["initial_block_poses"].copy()
        if self.free is None:
            return poses
        free = self.free.get_transforms()  # body origin = block centre, stage units, xyzw
        poses[~self.setup["block_fixed"]] = np.c_[free[:, :3] / MM, free[:, 3:]]
        return poses
