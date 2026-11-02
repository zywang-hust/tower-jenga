"""Raster renderer for the TOWER station: the four dataset cameras at 640x480 from link and block poses.

The station USD is read once (usd-core, no Kit): every visible, render-purpose mesh is triangulated into one
vertex buffer in the rest frame of its owner (the static world, a robot link or a block, i.e. the nearest
rigid-body ancestor). A frame then only needs one 4x4 per owner. Materials are the authored UsdPreviewSurface
values: base color (constant or texture, with its UsdTransform2d and wrap mode), roughness (constant or texture),
metallic, and the tangent-space normal map. Metallic textures are reduced to their mean. Lighting is the authored rect light (point
approximation with a soft shadow map) plus a hemispheric ambient term standing in for RTX indirect light.

Backend: nvdiffrast's CUDA rasterizer with deferred shading in torch. On the H800 only two TPCs run the
graphics pipeline, so OpenGL is geometry bound (28 ms per 4-camera set, 19 ms of it with 8x8 viewports);
the CUDA rasterizer uses every SM and batches environments x cameras in one call.

Units: metres; poses are world-from-local 4x4 in the env world frame (= station frame scaled to metres).
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# torch.compile's inductor pool defaults to one worker process per CPU (32 per renderer process on the pod, 1 GB RSS
# each, all booting at import); this renderer compiles three small graphs once per process.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

# The official station, as every process here loads it (renderer, FK, and the Isaac physics via sim/scene.py). This
# module imports neither torch nor pxr at load, so any of the three interpreters can take the path from it.
STATION = str(Path(__file__).resolve().parents[1] / "runs/simdev/assets/station_538d8b1/aliyun-rtx-training.usdz")
ROOT = "/PNP_DUALD1200_STATION"
ROBOTS = {"left": f"{ROOT}/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera",
          "back": f"{ROOT}/StationModel/InstalledRobots/StandB_Robot/NeroRobotWithWristCamera"}
WRIST = "gripper_base/WristCameraMount/camera_stand_link/d435_camera_link/camera_link/camera_color_frame/" \
        "camera_color_optical_frame/wrist_camera"
CAMERAS = {"left_back_camera": f"{ROOT}/CameraAlignment/low_camera",
           "right_front_camera": f"{ROOT}/CameraAlignment/high_camera",
           "left_wrist_camera": f"{ROBOTS['left']}/{WRIST}",
           "back_wrist_camera": f"{ROBOTS['back']}/{WRIST}"}
ROTATED = ("left_back_camera", "right_front_camera")  # dataset JPEGs of these are the raw render rotated 180 deg
BLOCKS = f"{ROOT}/ReplayTaskBlocks"
TEX = 2048  # texture atlas cell size (the block textures are up to 2000 px wide)
MIP_BIAS = -1.0  # texture LOD bias: RTX runs DLAA, whose integration guide sets log2(render / display) - 1 = -1
LIGHT_GRID = 2  # the rect light as LIGHT_GRID x LIGHT_GRID point samples, one shadow map each
DECIMATE = 5000, 0.25, 15000  # untextured meshes above 5000 triangles keep a quarter, at most 15000 (robot CAD, D435)
GL_FROM_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes (y down, z forward) -> GL (y up, z back)


@dataclass
class Scene:
    owners: list[str]       # prim paths; owner 0 is the static world
    rest: np.ndarray        # [O, 4, 4] rest pose of each owner
    verts: np.ndarray       # [V, 13] float32: position (owner frame), normal, uv, tangent + handedness, owner; by owner
    faces: np.ndarray       # [T, 3] int32
    face_material: np.ndarray  # [T] int32
    materials: np.ndarray   # [M, 12] float32: rgb, texture layer (-1 none), uv scale xy, offset xy, mirror, roughness, metallic,
                            # detail layer (-1 none)
    textures: np.ndarray    # [L, TEX, TEX, 4] uint8, sRGB base colour
    details: np.ndarray     # [D, TEX, TEX, 4] uint8, raw: tangent-space normal (rgb), roughness (a)
    cameras: dict           # name -> (owner index, owner-from-camera 4x4 in OpenCV axes, K 3x3)
    light: dict             # rect light as point samples: positions [K, 3], emission direction, rgb intensity per sample


def _rigid(m, mpu: float) -> np.ndarray:
    """Gf row-vector matrix in stage units -> column-vector rigid 4x4 in metres (scale removed)."""
    a = np.array(m, float).T
    a[:3, :3] /= np.linalg.norm(a[:3, :3], axis=0)
    a[:3, 3] *= mpu
    return a


def _owner(prim, index: dict) -> int:
    """Index of the nearest rigid-body ancestor (0 = static world)."""
    while prim and str(prim.GetPath()) not in index:
        prim = prim.GetParent()
    return index.get(str(prim.GetPath()), 0) if prim else 0


def _normals(mesh, pts, idx, counts, corner) -> np.ndarray:
    """Authored normals per triangle corner, else area-weighted smooth point normals."""
    from pxr import UsdGeom
    n = mesh.GetNormalsAttr().Get()
    interp = mesh.GetNormalsInterpolation()
    if n is not None and len(n):
        n = np.asarray(n, float)
        if interp == UsdGeom.Tokens.faceVarying:
            return n[corner]
        if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
            return n[idx[corner]]
        if interp == UsdGeom.Tokens.uniform:
            return np.repeat(n[_triangles(counts)[0]][:, None], 3, 1)
    v = pts[idx[corner]]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    acc = np.zeros_like(pts)
    for k in range(3):
        np.add.at(acc, idx[corner[:, k]], fn)
    return acc[idx[corner]]


def _triangles(counts: np.ndarray):
    """Fan triangulation: face index per triangle and the three corner indices into faceVertexIndices."""
    n = counts - 2
    face = np.repeat(np.arange(len(counts)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 1
    start = (np.cumsum(counts) - counts)[face]
    return face, np.stack([start, start + k, start + k + 1], 1)


def _crease_normals(tri: np.ndarray, angle: float = 40.0) -> np.ndarray:
    """Per-corner normals of a triangle soup [T, 3, 3]: area-weighted mean over faces sharing the vertex whose
    normal is within `angle` degrees of this face's normal (hard CAD edges stay hard, curved shells smooth)."""
    _, v = np.unique(tri.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    unit = fn / np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-20)
    face = np.repeat(np.arange(len(tri)), 3)
    order = np.argsort(v, kind="stable")
    count = np.bincount(v)
    deg = count[v]
    corner = np.repeat(np.arange(len(v)), deg)
    other = face[order][np.repeat(np.cumsum(count)[v] - deg, deg) + np.arange(deg.sum()) - np.repeat(np.cumsum(deg) - deg, deg)]
    keep = (unit[face[corner]] * unit[other]).sum(-1) > np.cos(np.radians(angle))
    n = np.stack([np.bincount(corner[keep], fn[other[keep], k], len(v)) for k in range(3)], -1)
    return n.reshape(-1, 3, 3)


def _decimate(tri: np.ndarray, mat: np.ndarray, keep: float):
    """Quadric decimation of a triangle soup [T, 3, 3] per material -> (soup, normals, material). With several
    materials the borders between them are kept so the pieces stay watertight against each other."""
    import fast_simplification
    parts, ids, groups = [], [], np.unique(mat)
    for m in groups:
        pts, idx = np.unique(tri[mat == m].reshape(-1, 3).round(7), axis=0, return_inverse=True)
        faces = idx.reshape(-1, 3)
        if len(faces) > 100:
            pts, faces = fast_simplification.simplify(pts, faces, 1 - keep, preserve_border=len(groups) > 1)
        parts.append(pts[faces])
        ids.append(np.full(len(faces), m))
    tri = np.concatenate(parts)
    return tri, _crease_normals(tri), np.concatenate(ids)


def _tangents(tri: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Per-triangle d(position)/du and the sign of d(position)/dv against n x t (mirrored uv), repeated per corner
    [T, 3, 4]; zero where the uv map is degenerate."""
    e1, e2, d1, d2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = d1[:, :1] * d2[:, 1:] - d2[:, :1] * d1[:, 1:]
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    t, b = (e1 * d2[:, 1:] - e2 * d1[:, 1:]) * inv, (e2 * d1[:, :1] - e1 * d2[:, :1]) * inv
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-20)
    w = np.where((np.cross(np.cross(e1, e2), t) * b).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    return np.repeat(np.concatenate([t, w], -1)[:, None], 3, 1)


def kelvin_rgb(t: float) -> np.ndarray:
    """Blackbody colour (Tanner Helland's fit), normalised to max 1, linearised from sRGB."""
    t = t / 100
    r = 255 if t <= 66 else 329.7 * (t - 60) ** -0.1332
    g = 99.47 * np.log(t) - 161.12 if t <= 66 else 288.12 * (t - 60) ** -0.0755
    b = 255 if t >= 66 else 0 if t <= 19 else 138.52 * np.log(t - 10) - 305.04
    c = np.clip([r, g, b], 0, 255) / 255
    return c ** 2.2 / (c ** 2.2).max()


class _Materials:
    """Deduplicated material table and texture array built while walking the stage."""

    def __init__(self, usdz: str):
        self.zip, self.rows, self.index, self.layers, self.layer_of = zipfile.ZipFile(usdz), [], {}, [], {}
        self.details, self.detail_of = [], {}

    def _image(self, shader) -> Image.Image:
        name = str(shader.GetInput("file").Get().path).lstrip("./")
        return Image.open(io.BytesIO(self.zip.read(name)))

    def _layer(self, shader) -> int:
        key = str(shader.GetInput("file").Get().path)
        if key not in self.layer_of:
            self.layer_of[key] = len(self.layers)
            self.layers.append(np.asarray(self._image(shader).convert("RGBA").resize((TEX, TEX), Image.BICUBIC)))
        return self.layer_of[key]

    def _detail(self, normal, rough, constant: float) -> int:
        """Detail layer: normal map in rgb (flat if none), roughness in alpha (the constant if no texture)."""
        key = tuple(str(t.GetInput("file").Get().path) if t else None for t in (normal, rough))
        if key not in self.detail_of:
            if normal:
                assert tuple(normal.GetInput("scale").Get())[:3] == (2, 2, 2) and tuple(normal.GetInput("bias").Get())[:3] == (-1, -1, -1)
            rgb = self._image(normal).convert("RGB").resize((TEX, TEX), Image.BICUBIC) if normal else Image.new("RGB", (TEX, TEX), (128, 128, 255))
            a = self._image(rough).convert("L").resize((TEX, TEX), Image.BICUBIC) if rough else Image.new("L", (TEX, TEX), round(constant * 255))
            self.detail_of[key] = len(self.details)
            self.details.append(np.dstack([np.asarray(rgb), np.asarray(a)]))
        return self.detail_of[key]

    def __call__(self, material) -> int:
        from pxr import UsdShade
        rgb, layer, uv, mirror, rough, metal, detail = (0.5, 0.5, 0.5), -1, (1, 1, 0, 0), 0.0, 0.5, 0.0, -1
        surface = material.ComputeSurfaceSource()[0] if material else None
        if surface:
            def value(name, default):
                i = surface.GetInput(name)
                if not i:
                    return default, None
                if i.HasConnectedSource():
                    return default, UsdShade.Shader(i.GetConnectedSources()[0][0].source.GetPrim())
                return i.Get(), None
            rgb, tex = value("diffuseColor", rgb)
            if tex is not None:
                layer = self._layer(tex)
                mirror = float(tex.GetInput("wrapS").Get() == "mirror") if tex.GetInput("wrapS") else 0.0
                st = tex.GetInput("st")
                if st and st.HasConnectedSource():
                    xf = UsdShade.Shader(st.GetConnectedSources()[0][0].source.GetPrim())
                    if xf.GetIdAttr().Get() == "UsdTransform2d":
                        s, t = xf.GetInput("scale").Get(), xf.GetInput("translation").Get()
                        uv = (*(s if s is not None else (1, 1)), *(t if t is not None else (0, 0)))
            rough, rough_tex = value("roughness", rough)
            normal_tex = value("normal", None)[1]
            if rough_tex is not None:
                rough = np.asarray(self._image(rough_tex).convert("L"), float).mean() / 255
            if rough_tex is not None or normal_tex is not None:
                detail = self._detail(normal_tex, rough_tex, rough)
            metal, tex = value("metallic", metal)
            if tex is not None:
                metal = np.asarray(self._image(tex).convert("L"), float).mean() / 255
        row = (*rgb, layer, *uv, mirror, rough, metal, detail)
        if row not in self.index:
            self.index[row] = len(self.rows)
            self.rows.append(row)
        return self.index[row]


def load_scene(usdz: str) -> Scene:
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(usdz)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    xc = UsdGeom.XformCache()

    def world(prim) -> np.ndarray:
        m = np.array(xc.GetLocalToWorldTransform(prim), float).T
        m[:3] *= mpu
        return m

    owners, rest = [""], [np.eye(4)]
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            owners.append(str(p.GetPath()))
            rest.append(_rigid(xc.GetLocalToWorldTransform(p), mpu))
    index = {o: i for i, o in enumerate(owners)}
    mats, chunks = _Materials(usdz), []
    it = iter(Usd.PrimRange(stage.GetPrimAtPath(ROOT)))
    for p in it:
        img = UsdGeom.Imageable(p)
        if img and (img.GetVisibilityAttr().Get() == "invisible" or img.GetPurposeAttr().Get() in ("guide", "proxy")):
            it.PruneChildren()
            continue
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        pts = np.asarray(mesh.GetPointsAttr().Get(), float)
        face, corner = _triangles(counts)
        owner = _owner(p, index)
        to_owner = np.linalg.inv(rest[owner]) @ world(p)
        pos = pts[idx[corner]] @ to_owner[:3, :3].T + to_owner[:3, 3]
        nrm = _normals(mesh, pts, idx, counts, corner)
        nrm = nrm @ np.linalg.inv(to_owner[:3, :3])  # inverse-transpose for column vectors
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        pv = UsdGeom.PrimvarsAPI(p).GetPrimvar("st")
        uv = np.zeros(corner.shape + (2,))
        if pv and pv.HasValue():
            vals, interp = np.asarray(pv.ComputeFlattened(), float), pv.GetInterpolation()
            uv = vals[corner] if interp == "faceVarying" else vals[idx[corner]] if interp in ("vertex", "varying") else uv
        mat = np.full(len(face), mats(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]))
        for s in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            sel = np.isin(face, np.asarray(s.GetIndicesAttr().Get()))
            bound = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
            if bound:
                mat[sel] = mats(bound)
        if len(face) > DECIMATE[0] and all(mats.rows[m][3] < 0 for m in np.unique(mat)):
            pos, nrm, mat = _decimate(pos.reshape(-1, 3, 3), mat, min(DECIMATE[1], DECIMATE[2] / len(face)))
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
            uv = np.zeros(nrm.shape[:2] + (2,))
        tri = np.concatenate([pos.reshape(-1, 3, 3), nrm.reshape(-1, 3, 3), uv, _tangents(pos.reshape(-1, 3, 3), uv)], -1).reshape(-1, 12)
        extra = np.repeat(np.stack([mat, np.full(len(mat), owner)], 1), 3, 0)
        chunks.append(np.concatenate([tri, extra], 1).astype(np.float32))

    cameras = {}
    for name, path in CAMERAS.items():
        cam = stage.GetPrimAtPath(path)
        owner = _owner(cam, index)
        pose = np.linalg.inv(rest[owner]) @ _rigid(xc.GetLocalToWorldTransform(cam), mpu) @ np.diag([1, -1, -1, 1])
        fx, fy, cx, cy = (cam.GetAttribute(f"omni:lensdistortion:opencvPinhole:{k}").Get() for k in ("fx", "fy", "cx", "cy"))
        cameras[name] = (owner, pose, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))

    lp = next(p for p in stage.Traverse() if p.IsA(UsdLux.RectLight))
    rl, m = UsdLux.RectLight(lp), _rigid(xc.GetLocalToWorldTransform(lp), mpu)
    rgb = np.asarray(rl.GetColorAttr().Get(), float) * rl.GetIntensityAttr().Get() * 2 ** rl.GetExposureAttr().Get()
    if rl.GetEnableColorTemperatureAttr().Get():
        rgb = rgb * kelvin_rgb(rl.GetColorTemperatureAttr().Get())
    size = np.array([rl.GetWidthAttr().Get(), rl.GetHeightAttr().Get()]) * mpu
    cells = (np.arange(LIGHT_GRID) + 0.5) / LIGHT_GRID - 0.5  # cell centres over the rectangle
    grid = np.stack(np.meshgrid(cells, cells, indexing="ij"), -1).reshape(-1, 2) * size
    light = {"samples": m[:3, 3] + grid @ m[:3, :2].T, "direction": -m[:3, 2], "rgb": rgb * size.prod() / len(grid)}
    soup = np.concatenate(chunks)  # weld identical corners, then order vertices by owner
    key = np.ascontiguousarray(np.delete(soup, 12, 1))
    uniq, inverse = np.unique(key.view(np.dtype((np.void, key.shape[1] * 4))).ravel(), return_inverse=True)
    verts = uniq.view(np.float32).reshape(-1, key.shape[1])
    order = np.argsort(verts[:, 12], kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return Scene(owners, np.stack(rest), verts[order], rank[inverse].reshape(-1, 3).astype(np.int32),
                 soup[::3, 12].astype(np.int32), np.array(mats.rows, np.float32), np.stack(mats.layers), np.stack(mats.details),
                 cameras, light)


def projection(K: np.ndarray, w: int, h: int, near: float = 0.01, far: float = 5.0) -> np.ndarray:
    """OpenCV intrinsics (pixel centres at integers) -> GL clip from GL camera (x right, y up, z back)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5
    return np.array([[2 * fx / w, 0, 1 - 2 * cx / w, 0], [0, 2 * fy / h, 2 * cy / h - 1, 0],
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])


def pose_matrix(p) -> np.ndarray:
    """4x4 passthrough, or xyz + quaternion xyzw (dataset block pose) -> 4x4."""
    p = np.asarray(p, float)
    if p.shape == (4, 4):
        return p
    x, y, z, w = p[3:] / np.linalg.norm(p[3:])
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = p[:3]
    return m


BLOCK_SIZE = np.array([0.15, 0.05, 0.03])  # the published block mesh along its x y z (metres)


def episode_dims(sim_attrs, names) -> dict:
    """{block name: [x, y, z] metres} of an episode's blocks whose size differs from the published block, from the
    HDF5 `sim` attrs block_dimensions_m (columns of sim/block_poses, null on task1-4); `names` from episode_blocks.
    The official sim scales those prims (task5 side blocks are 32 mm tall); sim/scene.py fidelity_setup reads the
    same attribute."""
    import json
    raw = sim_attrs.get("block_dimensions_m") if hasattr(sim_attrs, "get") else None
    dims = json.loads(raw) if raw is not None else None
    if not dims:
        return {}
    assert len(dims) == len(names), (len(dims), len(names))
    return {n: [float(x) for x in d] for n, d in zip(names, dims) if d is not None and not np.allclose(d, BLOCK_SIZE)}


def block_scale(dims) -> np.ndarray:
    """4x4 scale of the block's local axes taking the published block to `dims` (x y z metres): right-multiplies a
    world-from-block pose, as the official stage's prim scale does. _surface turns normals with the same matrix, not
    its inverse transpose: exact on the axis-aligned faces, about 4 degrees off on the mesh's few oblique corner
    normals at 32/30."""
    return np.diag([*(np.asarray(dims, float) / BLOCK_SIZE), 1.0])


def episode_blocks(usdz: str, fixed) -> list[str]:
    """Station block names of an episode's blocks, assigned as sim/scene.py Station does: fixed blocks take the
    kinematic prims and free blocks the others, each pool in authored order."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(usdz)  # keep the stage alive while its prims are read
    prims = [p for p in stage.GetPrimAtPath(BLOCKS).GetChildren() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    kinematic = [p.GetName() for p in prims if p.GetAttribute("physics:kinematicEnabled").Get()]
    pools = {True: iter(kinematic), False: iter(p.GetName() for p in prims if p.GetName() not in kinematic)}
    return [next(pools[bool(f)]) for f in fixed]


def _look_down(eye: np.ndarray, fov_deg: float, size: int) -> np.ndarray:
    """GL clip-from-world for a square camera at `eye` looking along -z (the rect light's emission axis)."""
    f = size / 2 / np.tan(np.radians(fov_deg) / 2)
    view = np.eye(4)
    view[:3, 3] = -eye  # camera axes = world axes: looking down -z with y up is already the GL convention
    return projection(np.array([[f, 0, size / 2 - 0.5], [0, f, size / 2 - 0.5], [0, 0, 1]]), size, size, 0.1, 3.0) @ view


@dataclass
class Look:
    """Global lighting constants, Nelder-Mead fit to official RTX frames of episodes not used by render_check.py
    (runs/simdev/render_dev/fit_look.py: PSNR 21.5 dB on its 6 frames x 4 cameras)."""
    exposure: float = 1.93e-3
    sky: tuple = (59.0, 59.0, 59.0)     # ambient radiance seen by upward normals
    ground: tuple = (43.0, 43.0, 43.0)  # and by downward normals
    softness: float = 2.9               # shadow PCF tap spacing in shadow-map texels
    background: tuple = (0.66, 0.66, 0.66)  # linear colour where no geometry is hit (RTX shows a light grey)
    shadow: float = 1.0                 # occluded share of the direct light actually removed (RTX fills the rest)
    relief: float = 1.0                 # normal-map xy gain (RTX's directional indirect light shows more relief)


HIDDEN = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 1]], float)  # clip (0,0,2,1): beyond far


class Renderer:
    """Renders the four dataset cameras for a batch of environments in one rasterizer call.

    render(link_poses, block_poses) -> {camera: HxWx3 uint8} for one env. render_batch(poses) takes [E, O, 4, 4]
    owner pose stacks from `poses()` and returns uint8 [E, 4, H, W, 3] in dataset orientation (CAMERAS order).
    """

    def __init__(self, usdz: str, device: str = "cuda", size=(640, 480), shadow: int = 512, look: Look = Look(),
                 refine: str | None = None):
        import nvdiffrast.torch as dr
        import torch

        from render_kernels import transform
        self._transform = transform
        self.torch, self.dr, self.look, (self.w, self.h), self.shadow = torch, dr, look, size, shadow
        s = self.scene = load_scene(usdz)
        self.dev = torch.device(device)
        f = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=self.dev)  # noqa: E731
        i = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.int64, device=self.dev)  # noqa: E731
        self.blocks = {o.rsplit("/", 1)[1]: k for k, o in enumerate(s.owners) if o.startswith(BLOCKS + "/")}
        for k in self.blocks.values():  # block_scale is relative to this size
            v = s.verts[s.verts[:, 12] == k, :3]
            assert np.allclose(np.ptp(v, 0), BLOCK_SIZE, atol=1e-5) and np.allclose(v.max(0) + v.min(0), 0, atol=1e-5), s.owners[k]
        self.links = {o: k for k, o in enumerate(s.owners) if k and not o.startswith(BLOCKS + "/")}
        owner = s.verts[:, 12].astype(int)
        self.owner = i(owner)
        self.pos = f(np.c_[s.verts[:, :3], np.ones(len(owner))])
        self.attr = f(s.verts[None, :, :12])
        self.tri = torch.as_tensor(s.faces, device=self.dev)
        self.face_material, self.face_owner = i(s.face_material), i(owner[s.faces[:, 0]])
        self.materials = f(s.materials)
        g = self.grid = int(np.ceil(np.sqrt(max(len(s.textures), len(s.details)))))

        def atlas(layers, gamma):  # layer k in cell (k % g, k // g); mip chain for the screen-space footprint
            a = np.zeros((g * TEX, g * TEX, layers.shape[-1]), np.uint8)
            for k, t in enumerate(layers):
                a[k // g * TEX:(k // g + 1) * TEX, k % g * TEX:(k % g + 1) * TEX] = t
            a = (f(a)[None] / 255) ** gamma
            return a, dr.texture_construct_mip(a, max_mip_level=6)
        self.atlas, self.mip = atlas(s.textures[..., :3], 2.2)  # sRGB -> linear
        self.detail, self.detail_mip = atlas(s.details, 1.0)
        # one rasterizer context per resolution: a shared one frees and reallocates its colour and depth buffers on
        # every switch between the camera and shadow-map sizes, and each cudaFree synchronises the device
        self.ctx, self.ctx_shadow = dr.RasterizeCudaContext(device=self.dev), dr.RasterizeCudaContext(device=self.dev)
        cams = list(s.cameras.values())
        self.cam_owner, self.cam_local = [c[0] for c in cams], f([c[1] for c in cams])
        self._cam_owner = i(self.cam_owner)  # a list index is uploaded (and synchronised) on every use
        # rows top-down for the raw views; the rotated ones get their 180 degrees from an x flip instead
        flip = [np.diag([-1, 1, 1, 1] if c in ROTATED else [1, -1, 1, 1]) for c in CAMERAS]
        self.cam_proj = f([fl @ projection(c[2], self.w, self.h) @ GL_FROM_CV for fl, c in zip(flip, cams)])
        L = s.light
        assert np.allclose(L["direction"], [0, 0, -1]), "shadow maps assume the authored downward rect light"
        self.light_vp = f([_look_down(x, 90.0, shadow) for x in L["samples"]])  # [K, 4, 4]
        self.light_pos, self.light_dir, self.light_rgb = f(L["samples"]), f(L["direction"]), f(L["rgb"])
        self.hidden = f(HIDDEN)
        self.casts = torch.ones(len(s.owners), dtype=torch.bool, device=self.dev)  # owners in the shadow maps
        if refine:
            from refine import Refine
            self.refine = Refine.load(refine, self.dev)
        else:
            self.refine = None

    def poses(self, link_poses: dict | None = None, block_poses: dict | None = None, dims: dict | None = None) -> np.ndarray:
        """Owner pose stack [O, 4, 4]: static = identity, links default to rest, unlisted blocks hidden (zero).
        `dims` {block name: [x, y, z] m} (episode_dims) scales those blocks' local axes from the published size."""
        out = self.scene.rest.copy()
        out[list(self.blocks.values())] = 0
        for path, m in (link_poses or {}).items():
            out[self.links[path]] = m
        for name, m in (block_poses or {}).items():
            out[self.blocks[name]] = pose_matrix(m) @ block_scale(dims[name]) if dims and name in dims else pose_matrix(m)
        return out

    def render(self, link_poses: dict | None = None, block_poses: dict | None = None, dims: dict | None = None) -> dict[str, np.ndarray]:
        return dict(zip(CAMERAS, self.render_batch(self.poses(link_poses, block_poses, dims)[None])[0]))

    def render_batch(self, poses: np.ndarray) -> np.ndarray:
        return self.shade(self.torch.as_tensor(np.asarray(poses, np.float32), device=self.dev)).cpu().numpy()

    def render_rgbx_async(self, poses: np.ndarray, out):
        """render_batch without waiting: the images as RGBX (4th byte 0) into the pinned uint8 host tensor `out`
        [>= E, 4, H, W, 4], copied on the current stream. -> a CUDA event that completes when `out` holds them."""
        torch = self.torch
        with torch.cuda.nvtx.range("h2d"):
            P = torch.from_numpy(np.asarray(poses, np.float32)).pin_memory().to(self.dev, non_blocking=True)
        rgb = self.shade(P)
        with torch.cuda.nvtx.range("d2h"):
            rgbx = torch.nn.functional.pad(rgb, (0, 1))
            out[:len(rgbx)].copy_(rgbx, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
        return done

    def shade(self, P):
        """Owner poses [E, O, 4, 4] (torch, device) -> uint8 [E, 4, H, W, 3] on the device, through the learned
        refinement when one is loaded."""
        nvtx = self.torch.cuda.nvtx  # stage ranges for the nsys timeline (docs/throughput.md §8.2)
        rgb, gbuf = self.raster(P, gbuffer=self.refine is not None)
        if self.refine is not None:
            with nvtx.range("refine"):
                rgb = self.refine(rgb, gbuf)
        with nvtx.range("uint8"):
            return (rgb.clamp(0, 1) * 255 + 0.5).to(self.torch.uint8)

    def _look_tensor(self, look: Look):
        """The Look constants on the device, uploaded once per distinct value (a per-call upload synchronises)."""
        key = (look.exposure, *look.sky, *look.ground, look.softness, *look.background, look.shadow, look.relief)
        if getattr(self, "_look_key", None) != key:
            self._look_key, self._look_dev = key, self.torch.tensor(key, dtype=self.torch.float32, device=self.dev)
        return self._look_dev

    def raster(self, P, gbuffer: bool = False):
        """Owner poses [E, O, 4, 4] -> sRGB float [E, 4, H, W, 3] and, if asked, the G-buffer [E, 4, H, W, 4]:
        camera-frame normal (OpenCV axes) and distance to the camera in metres, zero where nothing is hit."""
        torch, dr = self.torch, self.dr
        nvtx = torch.cuda.nvtx
        E, C, H, W = len(P), len(CAMERAS), self.h, self.w
        with nvtx.range("transform"):
            hidden = (P == 0).flatten(2).all(-1)  # [E, O]
            cam = P[:, self._cam_owner] @ self.cam_local  # [E, C, 4, 4] world-from-camera
            M = (self.cam_proj @ torch.linalg.inv_ex(cam)[0])[:, :, None] @ P[:, None]  # [E, C, O, 4, 4]; _ex: no sync
            clip = self._transform(self.pos, self.owner, torch.where(hidden[:, None, :, None, None], self.hidden, M).flatten(0, 1))
        with nvtx.range("raster"):
            rast, db = dr.rasterize(self.ctx, clip, self.tri, (H, W), grad_db=True)
            attr, da = dr.interpolate(self.attr, rast, self.tri, rast_db=db, diff_attrs=[6, 7])
        with nvtx.range("shadow"):
            Ml = torch.where((hidden | ~self.casts)[:, None, :, None, None], self.hidden, self.light_vp[None, :, None] @ P[:, None])
            rl, _ = dr.rasterize(self.ctx_shadow, self._transform(self.pos, self.owner, Ml.flatten(0, 1)), self.tri, (self.shadow, self.shadow))
            depth = torch.where(rl[..., 3] > 0, rl[..., 2], 1.0).view(E, -1, self.shadow, self.shadow)  # [E, K, S, S]

        with nvtx.range("surface"):
            face = rast[..., 3].long() - 1
            p, n, t, mat, uv, uv_detail, uv_da = _surface(attr, da, face, P[:, :, :3].flatten(0, 1), self.face_material,
                                                          self.face_owner, self.materials, C, self.grid)
            uv_da = uv_da * 2.0 ** MIP_BIAS
        with nvtx.range("texture"):
            sample = lambda a, m, x: dr.texture(a, x, uv_da, mip=m, filter_mode="linear-mipmap-linear", max_mip_level=6)  # noqa: E731
            tex, detail = sample(self.atlas, self.mip, uv), sample(self.detail, self.detail_mip, uv_detail)
        look = self.look
        with nvtx.range("lighting"):
            rgb = _lighting(p, n, t, mat, tex, detail, face, cam[..., :3, 3].flatten(0, 1), depth, self.light_vp, self.light_pos,
                            self.light_dir, self.light_rgb, self._look_tensor(look))
        with nvtx.range("antialias"):
            rgb = dr.antialias(rgb, rast, clip, self.tri).clamp(0, 1) ** (1 / 2.2)
        if not gbuffer:
            return rgb.view(E, C, H, W, 3), None
        with nvtx.range("gbuffer"):
            R = cam[..., :3, :3].flatten(0, 1)[:, None, None]  # world-from-camera rotation per image (OpenCV axes)
            eye = cam[..., :3, 3].flatten(0, 1)[:, None, None]
            nc = (R * n[..., :, None]).sum(-2)
            hit = (face >= 0)[..., None].float()
            gbuf = torch.cat([nc, (p - eye).norm(dim=-1, keepdim=True)], -1) * hit
        return rgb.view(E, C, H, W, 3), gbuf.view(E, C, H, W, 4)


def _compile(fn):
    """torch.compile on first call, so importing this module (sim/fk.py, sim/rollout.py and the Isaac workers do)
    loads no torch: compiling at import started torch's inductor worker pool, 32 processes, in every importer."""
    compiled = None

    def call(*args, **kw):
        nonlocal compiled
        if compiled is None:
            import torch
            compiled = torch.compile(fn, dynamic=True)
        return compiled(*args, **kw)
    return call


@_compile
def _surface(attr, da, face, P34, face_material, face_owner, materials, cams: int, grid: int):
    """Per pixel: world position, normal and tangent (+ handedness), material row, base-colour and detail atlas uv,
    their screen derivatives."""
    import torch
    f = face.clamp(min=0)
    mat = materials[face_material[f]]  # [B, H, W, 12]
    env = torch.arange(len(face), device=face.device)[:, None, None] // cams
    pose = P34[env * (len(P34) // (len(face) // cams)) + face_owner[f]]  # [B, H, W, 3, 4]
    p = (pose[..., :3] * attr[..., None, :3]).sum(-1) + pose[..., 3]
    n = torch.nn.functional.normalize((pose[..., :3] * attr[..., None, 3:6]).sum(-1), dim=-1)
    t = torch.cat([(pose[..., :3] * attr[..., None, 8:11]).sum(-1), attr[..., 11:12]], -1)
    uv = attr[..., 6:8] * mat[..., 4:6] + mat[..., 6:8]
    uv = torch.where(mat[..., 8:9] > 0.5, 1 - (uv.remainder(2) - 1).abs(), uv.remainder(1))
    uv = torch.stack([uv[..., 0], 1 - uv[..., 1]], -1).clamp(1 / TEX, 1 - 1 / TEX)  # st origin is bottom-left
    cell = lambda layer: torch.stack([layer.remainder(grid), (layer / grid).floor()], -1)  # noqa: E731
    return p, n, t, mat, (cell(mat[..., 3].clamp(min=0)) + uv) / grid, (cell(mat[..., 11].clamp(min=0)) + uv) / grid, \
        da * mat[..., [4, 4, 5, 5]] / grid


@_compile
def _lighting(p, n, t, mat, tex, detail, face, eye, depth, light_vp, light_pos, light_dir, light_rgb, look):
    """Linear radiance x exposure: rect light as K point samples (GGX + Lambert, PCF shadow map each) plus a
    hemispheric ambient term. Detail-mapped materials take roughness and a tangent-space normal from the map."""
    import torch
    norm = torch.nn.functional.normalize
    base = torch.where(mat[..., 3:4] >= 0, tex, mat[..., :3])
    mapped = mat[..., 11:12] >= 0
    tu = norm(t[..., :3] - n * (n * t[..., :3]).sum(-1, keepdim=True), dim=-1)  # Gram-Schmidt against the normal
    m = (detail[..., :3] * 2 - 1) * torch.cat([look[12:13], look[12:13], look[12:13] ** 0], -1)
    n = torch.where(mapped, norm(tu * m[..., :1] + torch.cross(n, tu, dim=-1) * t[..., 3:] * m[..., 1:2] + n * m[..., 2:], dim=-1), n)
    rough = torch.where(mapped, detail[..., 3:], mat[..., 9:10]).clamp(min=0.05)
    metal = mat[..., 10:11]
    v = torch.nn.functional.normalize(eye[:, None, None] - p, dim=-1)
    n = torch.where((n * v).sum(-1, keepdim=True) < 0, -n, n)
    nv = (n * v).sum(-1, keepdim=True).clamp(min=1e-3)
    a2, k = rough ** 4, (rough + 1) ** 2 / 8
    F0 = 0.04 + (base - 0.04) * metal
    direct = 0
    for i in range(len(light_pos)):
        lv = light_pos[i] - p
        d2 = (lv * lv).sum(-1, keepdim=True)
        l = lv * d2.rsqrt()  # noqa: E741
        nl = (n * l).sum(-1, keepdim=True).clamp(min=0)
        irradiance = light_rgb * (-(l * light_dir).sum(-1, keepdim=True)).clamp(min=0) * nl / d2 \
            * (1 - look[11] * (1 - _visible(p, depth[:, i], light_vp[i], look[7])))
        h = torch.nn.functional.normalize(l + v, dim=-1)
        nh = (n * h).sum(-1, keepdim=True).clamp(min=0)
        D = a2 / (torch.pi * (nh * nh * (a2 - 1) + 1) ** 2)
        Vis = 1 / (4 * (nl * (1 - k) + k) * (nv * (1 - k) + k))
        F = F0 + (1 - F0) * (1 - (h * v).sum(-1, keepdim=True).clamp(min=0)) ** 5
        direct = direct + ((1 - metal) * base / torch.pi + D * Vis * F) * irradiance
    ambient = (look[4:7] + (look[1:4] - look[4:7]) * (0.5 + 0.5 * n[..., 2:])) * (base + (F0 - base) * metal * 0.5)
    rgb = look[0] * (direct + ambient)
    return torch.where(face[..., None] >= 0, rgb, look[8:11]).contiguous()


def _visible(p, depth, light_vp, softness):
    """Fraction of a 3x3 PCF kernel (taps 2 x softness apart) around each pixel's light-space position that sees
    the light; each tap is a bilinear blend of the four nearest depth comparisons, so shadow edges do not step at
    shadow-map texels (a texel spans ~4 mm on the table, ~30 wrist-camera pixels at grasp range)."""
    import torch
    E, S = len(depth), depth.shape[-1]
    q = torch.cat([p, torch.ones_like(p[..., :1])], -1) @ light_vp.T
    q = q[..., :3] / q[..., 3:]
    xy = (q[..., :2] * 0.5 + 0.5) * S - 0.5
    env = torch.arange(len(p), device=p.device)[:, None, None] // (len(p) // E)
    lit = lambda ix, iy: (q[..., 2] - 2e-4 <= depth[env, iy.clamp(0, S - 1), ix.clamp(0, S - 1)]).float()  # noqa: E731
    vis = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            u, v = xy[..., 0] + 2 * dx * softness, xy[..., 1] + 2 * dy * softness
            x0, y0 = u.floor(), v.floor()
            fx, fy, x0, y0 = u - x0, v - y0, x0.long(), y0.long()
            vis = vis + (lit(x0, y0) * (1 - fx) + lit(x0 + 1, y0) * fx) * (1 - fy) \
                + (lit(x0, y0 + 1) * (1 - fx) + lit(x0 + 1, y0 + 1) * fx) * fy
    return vis[..., None] / 9
