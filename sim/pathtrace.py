"""Physically based reference renderer for the station: Mitsuba 3 (CUDA/OptiX, no RT cores needed) path tracing
of the same geometry, UsdPreviewSurface materials, textures and rect light that sim/render.py rasterises. It
answers how much of the gap to the official RTX frames is light transport (indirect light, soft shadows,
reflections) and how much is elsewhere (tone mapping, textures, DLAA, JPEG). Two venvs are involved:

  export   (render_venv, has pxr/nvdiffrast)  sim/pathtrace.py export <scene.npz>
  poses    (render_venv)                      sim/pathtrace.py poses <poses.npz> [--gap | --train N] [--seed S]
  render   (.venvs/mitsuba)                   sim/pathtrace.py render <scene.npz> <poses.npz> <out.npz> [--spp 64]
                                               [--depth 4] [--ambient A] [--exposure E] [--linear] [--denoise]
  fit      (any venv with numpy)              sim/pathtrace.py fit <linear.npz> <poses.npz>  -> exposure, ambient

`render --linear` writes the two linear components (rect light only, unit constant environment only) as float16
for `fit` (which needs the recorded frames that `poses --train` stores); without it, it writes uint8 sRGB
[N, 4, H, W, 3] in dataset orientation, the format sim/policy_gap.py takes with --images, plus the measured
seconds per state. Full meshes are used (no decimation); one launch renders the 4 cameras of a state.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BACKGROUND = 0.66  # linear, post exposure: what RTX shows where no geometry is hit (sim/render.py Look)
HIDE = np.array([0.0, 0.0, -50.0])
# camera order and dataset orientation as sim/render.py (not imported: the mitsuba venv has no torch)
CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")
ROTATED = ("left_back_camera", "right_front_camera")


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def export(out: str):
    """Scene arrays for the Mitsuba side, from sim/render.py's USD reader with decimation off."""
    import render
    from pxr import Usd, UsdGeom, UsdLux
    from render import STATION, _rigid, kelvin_rgb, load_scene
    render.DECIMATE = (10 ** 9, 1.0, 10 ** 9)
    s = load_scene(STATION)
    stage = Usd.Stage.Open(STATION)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lp = next(p for p in stage.Traverse() if p.IsA(UsdLux.RectLight))
    rl, m = UsdLux.RectLight(lp), _rigid(UsdGeom.XformCache().GetLocalToWorldTransform(lp), mpu)
    rgb = np.asarray(rl.GetColorAttr().Get(), float) * rl.GetIntensityAttr().Get() * 2 ** rl.GetExposureAttr().Get()
    if rl.GetEnableColorTemperatureAttr().Get():
        rgb = rgb * kelvin_rgb(rl.GetColorTemperatureAttr().Get())
    cams = {n: dict(owner=c[0], pose=c[1].tolist(), K=c[2].tolist()) for n, c in s.cameras.items()}
    np.savez(out, owners=np.array(s.owners), rest=s.rest, verts=s.verts, faces=s.faces, face_material=s.face_material,
             materials=s.materials, textures=s.textures, details=s.details, cameras=json.dumps(cams),
             light_pose=m, light_size=np.array([rl.GetWidthAttr().Get(), rl.GetHeightAttr().Get()]) * mpu, light_rgb=rgb)
    print(f"{len(s.owners)} owners, {len(s.faces)} triangles, {len(s.textures)} textures, {len(s.details)} detail maps -> {out}")


def poses(out: str, gap: bool, train: int, seed: int):
    """Owner pose stacks [N, O, 4, 4] of the policy-gap states (--gap) or of N random train-split states."""
    import h5py
    from fk import ArmFK
    from policy_gap import load_real, states
    from render import STATION, Renderer, episode_dims
    from train_refine import D, SPLIT, block_names
    if gap:
        st = states()
    else:
        rng = np.random.default_rng(seed)
        eps = sorted((e, task) for task, s in SPLIT.items() for e in s["train"] if not task.startswith("task5-3"))
        st = []
        for k in rng.choice(len(eps), train, replace=False):
            ep, task = eps[k]
            with h5py.File(f"{D}/{ep}") as f:
                st.append(dict(episode=ep, variant=task, t=int(rng.integers(len(f["observations/qpos"])))))
    R, fk, P, files = Renderer(STATION), ArmFK(STATION), [], {}
    for s in st:
        if s["episode"] not in files:
            f = h5py.File(f"{D}/{s['episode']}")
            files[s["episode"]] = f, block_names(f)
        f, names = files[s["episode"]]
        P.append(R.poses(fk.link_poses(f["observations/qpos"][s["t"]]), dict(zip(names, f["sim/block_poses"][s["t"]])),
                             episode_dims(f["sim"].attrs, names)))
    extra = {} if gap else {"images": load_real(st)[0]}  # recorded frames of the train states, for `fit`
    np.savez(out, poses=np.stack(P).astype(np.float32), states=json.dumps(st), **extra)
    print(f"{len(P)} pose stacks -> {out}")


class PathTracer:
    """Mitsuba scene of the exported station; render(P [O, 4, 4]) -> linear RGBA [4, H, W, 4] per camera."""

    def __init__(self, scene_npz: str, spp: int, depth: int, ambient: float, denoise: bool = False):
        import mitsuba as mi
        mi.set_variant("cuda_ad_rgb")
        self.mi, self.spp, (self.w, self.h) = mi, spp, (640, 480)
        self.denoiser = mi.OptixDenoiser((self.w, self.h), albedo=False, normals=False, temporal=False) if denoise else None
        z = np.load(scene_npz)
        self.owners, self.rest = list(z["owners"]), z["rest"]
        verts, faces, fmat, mats = z["verts"], z["faces"], z["face_material"], z["materials"]
        tex, det = z["textures"], z["details"]
        owner = verts[:, 12].astype(int)[faces[:, 0]]
        d: dict = {"type": "scene", "integrator": {"type": "path", "max_depth": depth, "hide_emitters": True}}
        for k in range(len(tex)):
            d[f"tex{k}"] = {"type": "bitmap", "bitmap": mi.Bitmap(np.ascontiguousarray(tex[k, ..., :3])), "raw": False, "id": f"tex{k}"}
        for k in range(len(det)):  # normal map (rgb) and roughness (alpha), both raw
            d[f"nrm{k}"] = {"type": "bitmap", "bitmap": mi.Bitmap(np.ascontiguousarray(det[k, ..., :3])), "raw": True, "id": f"nrm{k}"}
            d[f"rgh{k}"] = {"type": "bitmap", "bitmap": mi.Bitmap(np.ascontiguousarray(det[k, ..., 3:4])), "raw": True, "id": f"rgh{k}"}
        self.bsdf = {}
        for i, m in enumerate(mats):
            rgb, layer, (sx, sy, ox, oy), mirror, rough, metal, detail = m[:3], int(m[3]), m[4:8], m[8] > 0.5, float(m[9]), float(m[10]), int(m[11])
            to_uv = mi.ScalarTransform3f(np.array([[sx, 0, ox], [0, -sy, 1 - oy], [0, 0, 1]], np.float32))  # st origin bottom-left
            wrap = "mirror" if mirror else "repeat"
            base = {"type": "rgb", "value": [float(c) for c in rgb]}
            if layer >= 0:
                base = dict(d[f"tex{layer}"], to_uv=to_uv, wrap_mode=wrap)
                base.pop("id")
            r = {"type": "rgb", "value": max(rough, 0.05)} if detail < 0 else dict(d[f"rgh{detail}"], to_uv=to_uv, wrap_mode=wrap)
            r.pop("id", None)
            b = {"type": "principled", "base_color": base, "roughness": r, "metallic": metal, "specular": 0.5}
            if detail >= 0:
                n = dict(d[f"nrm{detail}"], to_uv=to_uv, wrap_mode=wrap)
                n.pop("id")
                b = {"type": "normalmap", "normalmap": n, "bsdf": b}
            self.bsdf[i] = b
        # One mesh per (owner, material), one BSDF object per material (distinct BSDF objects per mesh made every
        # render re-trace the Dr.Jit graph, ~9 s). Every mesh (and the emissive quad of the rect light) gets
        # world-frame vertex positions per state; the 4 cameras are one batch sensor, i.e. one launch per state.
        self.meshes, shared = [], {}  # (owner, name, rest positions Point3f, rest normals Normal3f)
        for o in range(len(self.owners)):
            sel_o = owner == o
            for mat in np.unique(fmat[sel_o]):
                f = faces[sel_o & (fmat == mat)]
                used, inv = np.unique(f.ravel(), return_inverse=True)
                v = verts[used]
                props = mi.Properties()
                props["bsdf"] = shared.setdefault(int(mat), mi.load_dict(self.bsdf[int(mat)]))
                name = f"o{o}m{mat}"
                d[name] = self._mesh(name, v[:, :3], v[:, 3:6], v[:, 6:8], inv, props)
                self.meshes.append((o, name, mi.Point3f(v[:, :3].T), mi.Normal3f(v[:, 3:6].T)))
        # the authored rect light as an emissive quad (owner index `light`): local xy plane, emitting along local -z
        L, (w, h) = z["light_pose"], z["light_size"]
        self.light_pose, light = L, len(self.owners)
        quad = np.array([[-w / 2, -h / 2, 0], [-w / 2, h / 2, 0], [w / 2, h / 2, 0], [w / 2, -h / 2, 0]], np.float32)
        props = mi.Properties()
        props["emitter"] = mi.load_dict({"type": "area", "radiance": {"type": "rgb", "value": [float(c) for c in z["light_rgb"]]}})
        d["light"] = self._mesh("light", quad, np.tile([0, 0, -1.0], (4, 1)), np.zeros((4, 2)), np.array([0, 2, 1, 0, 3, 2]), props)
        self.meshes.append((light, "light", mi.Point3f(quad.T), mi.Normal3f(np.tile([0, 0, -1.0], (4, 1)).T)))
        if ambient > 0:
            d["env"] = {"type": "constant", "radiance": {"type": "rgb", "value": ambient}}
        self.cams = json.loads(str(z["cameras"]))
        self.scene = mi.load_dict(d)
        self.params = mi.traverse(self.scene)
        self.sensor, self.crops = self._batch_sensor()
        self.sp = mi.traverse(self.sensor)

    def _mesh(self, name, pos, nrm, uv, faces, props):
        mi = self.mi
        m = mi.Mesh(name, len(pos), len(faces) // 3, has_vertex_normals=True, has_vertex_texcoords=True, props=props)
        q = mi.traverse(m)
        q["vertex_positions"] = mi.Float(np.asarray(pos, np.float32).ravel())
        q["vertex_normals"] = mi.Float(np.asarray(nrm, np.float32).ravel())
        q["vertex_texcoords"] = mi.Float(np.asarray(uv, np.float32).ravel())
        q["faces"] = mi.UInt32(np.asarray(faces, np.uint32).ravel())
        q.update()
        return m

    def _batch_sensor(self):
        """The 4 OpenCV pinholes (pixel centres at integers) as perspective sub-sensors (x left, y up, z forward) of one
        batch sensor on a common virtual film centred on each principal point; the W x H window is cut out per camera.
        Near clips as the official capture: 10 mm wrist (the optical centre sits inside the D435 housing), 1 mm external."""
        mi = self.mi
        W, H = self.w, self.h
        pp = {n: (c["K"][0][2] + 0.5, c["K"][1][2] + 0.5) for n, c in self.cams.items()}  # continuous principal point
        hw = int(np.ceil(max(max(cx, W - cx) for cx, _ in pp.values())))  # half size: window inside, centring error <= 0.5 px
        hh = int(np.ceil(max(max(cy, H - cy) for _, cy in pp.values())))
        subs, crops = {}, {}
        for k, n in enumerate(CAMERAS):
            fov = float(np.degrees(2 * np.arctan(hw / self.cams[n]["K"][0][0])))
            subs[f"s{k}"] = {"type": "perspective", "fov": fov, "fov_axis": "x", "near_clip": 0.01 if "wrist" in n else 0.001,
                             "far_clip": 5.0 + 1e-3 * k}  # distinct: identical sub-sensor dicts are merged by load_dict
            crops[n] = (k * 2 * hw + int(round(hw - pp[n][0])), int(round(hh - pp[n][1])))
        self.film = (2 * hw, 2 * hh)
        sensor = mi.load_dict({"type": "batch", **subs, "film": {"type": "hdrfilm", "width": len(CAMERAS) * 2 * hw, "height": 2 * hh,
                                                                  "pixel_format": "rgba", "rfilter": {"type": "box"}},
                               "sampler": {"type": "independent", "sample_count": self.spp}})
        return sensor, crops

    def render(self, P: np.ndarray) -> dict[str, np.ndarray]:
        """Owner poses [O, 4, 4] (zero = hidden) -> {camera: linear RGBA [H, W, 4]}."""
        import drjit as dr
        mi = self.mi
        W = np.concatenate([P.astype(np.float64), self.light_pose[None]])  # world-from-owner, light appended
        hidden = (W == 0).reshape(len(W), -1).all(-1)
        W[hidden] = np.eye(4)
        W[hidden, :3, 3] = HIDE
        for o, name, rest, nrm in self.meshes:
            M = mi.Transform4f(mi.Matrix4f(W[o].astype(np.float32)))
            self.params[f"{name}.vertex_positions"] = dr.ravel(M @ rest)
            self.params[f"{name}.vertex_normals"] = dr.ravel(M @ nrm)
        self.params.update()
        for k, n in enumerate(CAMERAS):
            cam = self.cams[n]
            M = P[cam["owner"]] @ np.array(cam["pose"]) @ np.diag([-1.0, -1.0, 1.0, 1.0])  # OpenCV -> Mitsuba camera axes
            self.sp[f"s{k}.to_world"] = mi.Transform4f(mi.Matrix4f(M.astype(np.float32)))
        self.sp.update()
        # no dr.freeze (replays rendered the wrist sensors black); a plain render re-traces in ~0.25 s
        img = np.asarray(mi.render(self.scene, sensor=self.sensor, spp=self.spp, seed=0))
        out = {}
        for n, (x, y) in self.crops.items():
            tile = np.ascontiguousarray(img[y:y + self.h, x:x + self.w])
            out[n] = np.asarray(self.denoiser(mi.TensorXf(tile))) if self.denoiser else tile
        return out


def render_states(scene_npz: str, poses_npz: str, out: str, spp: int, depth: int, ambient: float, exposure: float, linear: bool,
                  denoise: bool = False):
    P = np.load(poses_npz)["poses"]
    t0 = time.time()
    rot = [c in ROTATED for c in CAMERAS]  # the dataset JPEGs of the external cameras are the raw view turned 180 deg
    if linear:  # components for `fit`: rect light only, then unit constant environment only
        pt = PathTracer(scene_npz, spp, depth, 0.0)
        env = PathTracer(scene_npz, spp, depth, 1.0)
        env.params["light.emitter.radiance.value"] = env.mi.Color3f(0.0)
        env.params.update()
        A, B = [], []
        for i, p in enumerate(P):
            a, b = pt.render(p), env.render(p)
            a, b = np.stack([a[c] for c in CAMERAS]), np.stack([b[c] for c in CAMERAS])
            a[rot], b[rot] = a[rot][:, ::-1, ::-1], b[rot][:, ::-1, ::-1]
            A.append(a)
            B.append(b)
            print(f"{i + 1}/{len(P)} {(time.time() - t0) / (i + 1):.1f} s/state", flush=True)
        np.savez(out, light=np.stack(A).astype(np.float16), env=np.stack(B).astype(np.float16))
    else:
        pt, imgs = PathTracer(scene_npz, spp, depth, ambient, denoise), []
        t0 = time.time()
        for i, p in enumerate(P):
            r = pt.render(p)
            im = np.stack([r[c] for c in CAMERAS])  # [4, H, W, 4] linear
            rgb = exposure * im[..., :3] + BACKGROUND * (1 - im[..., 3:4])
            im8 = (srgb(rgb) * 255 + 0.5).astype(np.uint8)
            im8[rot] = im8[rot][:, ::-1, ::-1]
            imgs.append(im8)
            if i % 10 == 0:
                print(f"{i + 1}/{len(P)} {(time.time() - t0) / (i + 1):.2f} s/state", flush=True)
        sec = (time.time() - t0) / len(P)
        np.savez(out, images=np.stack(imgs), seconds_per_state=sec)
        print(f"render {sec:.3f} s/state (4 cameras, spp {spp}, denoise {denoise})", flush=True)
    print("PATHTRACE_DONE", flush=True)


def fit(linear_npz: str, real_npz: str):
    """Least-squares exposure e and ambient a: srgb^-1(real) ~ e * (light + a * env), on pixels that hit geometry;
    then the per-channel gains left over (white balance) and a linearity check (tone curve)."""
    z, real = np.load(linear_npz), np.load(real_npz)["images"].astype(np.float32) / 255
    light, env = z["light"].astype(np.float32), z["env"].astype(np.float32)
    lin = np.where(real <= 0.04045, real / 12.92, ((real + 0.055) / 1.055) ** 2.4)
    hit = light[..., 3] > 0.999
    A = np.stack([light[..., :3][hit].ravel(), env[..., :3][hit].ravel()], 1)
    y = lin[hit].ravel()
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    e, ea = sol
    pred = A @ sol
    print(json.dumps(dict(exposure=float(e), ambient=float(ea / e), rmse_linear=float(np.sqrt(((pred - y) ** 2).mean())),
                          psnr_srgb=float(10 * np.log10(1 / ((srgb(pred) - srgb(y)) ** 2).mean())), n=int(hit.sum()))))
    P3, Y3 = pred.reshape(-1, 3), y.reshape(-1, 3)
    print("per-channel gain left over (RGB):", np.round((P3 * Y3).sum(0) / (P3 * P3).sum(0), 3))
    # linearity check: binned median of real against prediction (a straight line means no extra tone curve)
    bins = np.quantile(pred, np.linspace(0, 1, 11))
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pred >= lo) & (pred < hi)
        print(f"pred {lo:.3f}-{hi:.3f}: real median {np.median(y[m]):.3f} ratio {np.median(y[m]) / max(np.median(pred[m]), 1e-6):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("out")
    p = sub.add_parser("poses")
    p.add_argument("out")
    p.add_argument("--gap", action="store_true")
    p.add_argument("--train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("render")
    r.add_argument("scene")
    r.add_argument("poses")
    r.add_argument("out")
    r.add_argument("--spp", type=int, default=64)
    r.add_argument("--depth", type=int, default=4)
    r.add_argument("--ambient", type=float, default=0.0)
    r.add_argument("--exposure", type=float, default=1.0)
    r.add_argument("--linear", action="store_true")
    r.add_argument("--denoise", action="store_true", help="OptiX denoiser on the linear frame")
    f = sub.add_parser("fit")
    f.add_argument("linear")
    f.add_argument("real")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.out)
    elif a.cmd == "poses":
        poses(a.out, a.gap, a.train, a.seed)
    elif a.cmd == "render":
        render_states(a.scene, a.poses, a.out, a.spp, a.depth, a.ambient, a.exposure, a.linear, a.denoise)
    else:
        fit(a.linear, a.real)


if __name__ == "__main__":
    main()
