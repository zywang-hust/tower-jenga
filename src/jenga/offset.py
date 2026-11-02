"""Estimate the rigid offset of the movable assembly from the first frame of the two external cameras.

Model-based: the nominal layout of the variant (configs/cameras.json "layouts") is projected with the
published camera model (configs/cameras.json "cameras"). Movable blocks (ids source_* except the source
foundation) are moved rigidly about their nominal centre c by (dx, dy, dpsi); every other block is drawn at
its fixed nominal pose and acts as the reference. The score is the correlation of the rendered silhouette
union with a blurred +-1 wood-colour mask, plus edge_weight x the grey-gradient map along the visible edges
of the movable blocks, summed over both cameras; search is a translation x yaw grid followed by pattern
search over (dx, dy, dpsi). The edge term carries the wood-on-wood structure the colour mask merges: a single
block or a two-block tower on a wooden foundation has little silhouette of its own, and the silhouette alone
traded yaw against translation (task5-2: 7% of synthetic rule-box shifts off by > 15 mm, up to 38 mm; with
the edge term none above 8 mm, runs/spatial/gate/). Confidence is the worst per-camera F1 between the rendered movable
silhouette and the wood mask in the search ROI; below `min_confidence` the caller should use the identity.
The raw estimate is gated per variant before use (jenga.warp.gate_offset, layouts[variant]["offset_gate"]).

Images: RGB uint8 [480, 640, 3], as delivered online and as decoded from the dataset JPEGs (the external
cameras are already rotated by 180 degrees; the camera model accounts for it). Before the mask, each image is
mapped per channel onto the variant's nominal tone (`normalise`, layouts[variant]["photometric_reference"]), so
gain, white balance and gamma changes do not move the wood hue out of the fixed HSV window. Without it a
3000/8500 K cast or gamma 0.7 sent every val nominal frame to the identity (runs/spatial/validate/l.json).

The rendered template has a systematic per-variant offset against the simulator truth (2-4 mm, about
1 deg; camera model and mask boundary effects). scripts/fit_kinematics.py measures it on training episodes
and stores it as layouts[variant]["estimator_bias"]; `estimate` subtracts it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from jenga.kinematics import quat_xyzw_to_matrix
from jenga.warp import Offset, resolve_variant

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cameras.json"
EXTERNAL = ("left_back_camera", "right_front_camera")
_UNIT = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
_FACES = np.array([[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]])  # -x +x -y +y -z +z


def is_movable(block_id: str) -> bool:
    return block_id.startswith("source_") and not block_id.startswith("source_base")


class Camera:
    def __init__(self, cfg: dict, size_wh=(640, 480)):
        self.K = np.array(cfg["K"], float)
        w, x, y, z = cfg["orient_wxyz"]
        R_usd = quat_xyzw_to_matrix(np.array([x, y, z, w]))
        self.R_wc = R_usd @ np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes in world
        self.t = np.array(cfg["translate_mm"], float) / 1000.0
        self.rot180 = bool(cfg.get("rotate_180", False))
        self.size = np.array(size_wh, float)

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points [..., 3] -> stored-image pixel coordinates [..., 2]."""
        Xc = (X - self.t) @ self.R_wc
        uv = Xc @ self.K.T
        uv = uv[..., :2] / uv[..., 2:3]
        return self.size - 1 - uv if self.rot180 else uv


def box_corners(poses: np.ndarray, dims) -> np.ndarray:
    """poses [N,7] (xyz + quat xyzw) -> corners [N,8,3]."""
    R = quat_xyzw_to_matrix(poses[:, 3:7])
    return np.einsum("nij,kj->nki", R, _UNIT * np.asarray(dims)) + poses[:, None, :3]


def edge_map(rgb: np.ndarray) -> np.ndarray:
    """Grey-level gradient magnitude, float32. Wood-on-wood structure the colour mask cannot see: the shaded side
    faces of a block resting on a wooden foundation, and the lines between its top and side faces."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def channel_quantiles(rgb: np.ndarray, levels) -> np.ndarray:
    """Per-channel grey values [3, L] at the given quantile levels (smallest v with CDF(v) >= level)."""
    rgb = np.ascontiguousarray(rgb)
    out = []
    for ch in range(3):
        cdf = np.cumsum(cv2.calcHist([rgb], [ch], None, [256], [0, 256]).ravel())
        out.append(np.searchsorted(cdf, np.asarray(levels) * cdf[-1]))
    return np.array(out, float)


def photometric_lut(rgb: np.ndarray, reference: np.ndarray, levels) -> np.ndarray:
    """Per-channel tone curve [256, 3] that maps the image's quantiles onto the nominal `reference` [3, L].

    Each channel is fitted as y = a x^b in log space (least squares over the quantiles that are neither clipped
    nor near black, b in [0.5, 2]): exact for a global gain, white balance (per-channel gain) and gamma, which
    is what moves the wood hue out of the mask under a colour cast. With fewer than three usable quantiles only
    the gain is fitted. On a nominal image the curve is close to the identity."""
    q = channel_quantiles(rgb, levels)
    x = np.arange(256.0) / 255.0
    lut = np.empty((256, 3), np.uint8)
    for ch in range(3):
        use = (q[ch] >= 3) & (q[ch] <= 251) & (reference[ch] >= 3)
        lx, ly = np.log(q[ch][use] / 255.0), np.log(reference[ch][use] / 255.0)
        if use.sum() >= 3:
            b = float(np.clip(np.polyfit(lx, ly, 1)[0], 0.5, 2.0))
            la = float(np.mean(ly - b * lx))
        else:
            b, la = 1.0, float(np.median(ly - lx)) if use.any() else 0.0
        lut[:, ch] = np.clip(np.round(255.0 * np.exp(la) * x ** b), 0, 255)
    return lut


def normalise(rgb: np.ndarray, reference: np.ndarray | None, levels) -> tuple[np.ndarray, np.ndarray]:
    """(image mapped to the nominal tone curves, valid pixels). Pixels with a saturated channel have lost their
    colour; the silhouette score and the confidence ignore them. Where the curve stretches the dark levels more than 3x (gain about 0.3 or a strong cast)
    the 8-bit steps become hue noise in the mask, so the result is box-filtered 3x3."""
    rgb = np.ascontiguousarray(rgb)
    valid = cv2.inRange(rgb, (0, 0, 0), (253, 253, 253)) > 0
    if reference is None:
        return rgb, valid
    lut = photometric_lut(rgb, reference, levels)
    out = cv2.LUT(rgb, lut[None])
    if lut[16].max() > 3 * 16:
        out = cv2.blur(out, (3, 3))
    return out, valid


def wood_mask(rgb: np.ndarray) -> np.ndarray:
    """Wood-coloured pixels (lit and shaded faces); robot black/white, table grey are rejected."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= 8) & (h <= 28) & (s >= 70) & (v >= 15)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


@dataclass
class OffsetResult:
    offset: Offset
    confidence: float
    ok: bool
    seconds: float
    per_camera_f1: dict
    saturated: float = 0.0  # largest per-camera share of search-ROI pixels with a clipped channel


def move_corners(C: np.ndarray, c: np.ndarray, x) -> np.ndarray:
    """Rigid xy motion (dx, dy, dpsi) about c applied to points [..., 3]; the components of x may be arrays
    that broadcast against C[..., 0] (a batch of motions)."""
    cs, sn = np.cos(x[2]), np.sin(x[2])
    out = np.broadcast_to(C, np.broadcast_shapes(C.shape, np.shape(x[0]) + (1,))).copy()
    rx, ry = C[..., 0] - c[0], C[..., 1] - c[1]
    out[..., 0] = c[0] + cs * rx - sn * ry + x[0]
    out[..., 1] = c[1] + sn * rx + cs * ry + x[1]
    return out


class _View:
    """One camera restricted to the ROI where the movable blocks can appear; renders at `scale`."""

    def __init__(self, cam: Camera, mask: np.ndarray, fixed_corners: np.ndarray, mov_corners: np.ndarray, c,
                 reach_m: float, scale: float, edges: np.ndarray | None = None, edge_weight: float = 0.0,
                 valid: np.ndarray | None = None):
        self.cam, self.scale, self.edge_weight = cam, scale, edge_weight
        ext = [cam.project(move_corners(mov_corners, c, (dx, dy, psi))).reshape(-1, 2)
               for dx in (-reach_m, reach_m) for dy in (-reach_m, reach_m) for psi in (-0.25, 0.0, 0.25)]
        ext = np.concatenate(ext)
        x0, y0 = np.floor(np.clip(ext.min(0) - 8, 0, cam.size - 1)).astype(int)
        x1, y1 = np.ceil(np.clip(ext.max(0) + 8, 0, cam.size - 1)).astype(int)
        self.origin = np.array([x0, y0], float)
        mask = mask[y0:y1 + 1, x0:x1 + 1]
        self.valid = np.ones(mask.shape, bool) if valid is None else valid[y0:y1 + 1, x0:x1 + 1]
        self.mask = mask.astype(bool) & self.valid
        size = (max(1, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale)))
        self.shape = (size[1], size[0])
        m = (2 * mask.astype(np.float32) - 1) * self.valid  # saturated pixels: 0, neither wood nor background
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
        self.M = cv2.GaussianBlur(m, (0, 0), max(1.0, 2.0 * scale))
        self.fixed = self.render(fixed_corners, np.zeros(self.shape, np.float32))
        self.E = None
        if edge_weight > 0 and edges is not None:
            e = cv2.resize(edges[y0:y1 + 1, x0:x1 + 1], size, interpolation=cv2.INTER_AREA)
            e = cv2.GaussianBlur(e, (0, 0), max(1.0, 1.5 * scale))
            self.E = np.minimum(e / max(float(np.percentile(e, 99)), 1e-6), 1.0)  # ~[0, 1], robust to contrast
            # camera-facing faces of the nominal movable blocks, fixed for the search: a face that turns within
            # the search range (25 mm, 15 deg) is nearly edge-on and its edges are shared with drawn faces
            F = mov_corners[:, _FACES]
            fc = F.mean(2)
            self.facing = np.einsum("nfk,nfk->nf", fc - mov_corners.mean(1, keepdims=True), cam.t - fc) > 0

    def _px(self, corners: np.ndarray) -> np.ndarray:
        """World corners [..., 3] -> ROI pixel coordinates at this scale, 1/16 px fixed point (cv2 shift=4)."""
        return np.round((self.cam.project(corners) - self.origin) * (16 * self.scale)).astype(np.int32)

    def render(self, corners: np.ndarray, canvas: np.ndarray, px: np.ndarray | None = None) -> np.ndarray:
        """Draw boxes given by world corners [N,8,3] into `canvas` (anti-aliased, 1/16 px)."""
        px = self._px(corners) if px is None else px
        if len(px) == 0:
            return canvas
        for pts in px:
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 1.0, lineType=cv2.LINE_AA, shift=4)
        return canvas

    def render_edges(self, px: np.ndarray) -> np.ndarray:
        """Visible edges (those of camera-facing faces) of the movable boxes, given as _px [N,8,2], 1 px lines."""
        canvas = np.zeros(self.shape, np.float32)
        cv2.polylines(canvas, list(px[:, _FACES][self.facing]), True, 1.0, 1, lineType=cv2.LINE_AA, shift=4)
        return canvas

    def score(self, corners: np.ndarray, px: np.ndarray | None = None) -> float:
        """Silhouette correlation with the +-1 wood mask, plus edge_weight x the edge map along the visible edges
        of the movable blocks (corners [N,8,3], or their _px if given)."""
        px = self._px(corners) if px is None else px
        s = float(np.vdot(self.render(corners, self.fixed.copy(), px), self.M))
        if self.E is not None:
            s += self.edge_weight * float(np.vdot(self.render_edges(px), self.E))
        return s


class OffsetEstimator:
    """`estimate(task_variant, images)` -> OffsetResult. One instance can serve all variants (stateless)."""

    def __init__(self, config: Path | str = DEFAULT_CONFIG, *, reach_m: float = 0.025, max_psi: float = np.radians(15),
                 min_confidence: float = 0.75, correct_bias: bool = True, edge_weight: float | None = None):
        """edge_weight None takes configs/cameras.json "estimator"."edge_weight" (0 disables the edge term); the
        per-variant estimator_bias and offset_gate in that file are fitted for that setting."""
        cfg = json.loads(Path(config).read_text())
        self.levels = np.array(cfg.get("estimator", {}).get("photometric_levels", []), float)
        self.cams = {k: Camera(v, cfg["image_size_wh"]) for k, v in cfg["cameras"].items()}
        self.dims = np.array(cfg["block_dimensions_m"], float)
        self.layouts = cfg["layouts"]["variants"]
        self.reach_m, self.max_psi, self.min_confidence = reach_m, max_psi, min_confidence
        self.correct_bias = correct_bias
        self.edge_weight = float(cfg.get("estimator", {}).get("edge_weight", 0.0) if edge_weight is None else edge_weight)
        self.saturated_gate = cfg.get("estimator", {}).get("saturated_gate")

    def template(self, task_variant: str):
        """(movable poses [N,7], fixed poses [M,7], centre c [2]) of the nominal layout."""
        blocks = self.layouts[resolve_variant(task_variant, self.layouts)]["blocks"]
        mov = np.array([v for k, v in blocks.items() if is_movable(k)], float)
        fix = np.array([v for k, v in blocks.items() if not is_movable(k)], float)
        return mov, fix, mov[:, :2].mean(0)

    def bias(self, task_variant: str) -> np.ndarray:
        b = self.layouts[resolve_variant(task_variant, self.layouts)].get("estimator_bias") if self.correct_bias else None
        return np.zeros(3) if b is None else np.array([b["dx"], b["dy"], b["dpsi"]], float)

    def footprint_error_mm(self, task_variant: str, a, b) -> np.ndarray:
        """Largest horizontal distance (mm) between the movable block corners placed by offsets a and b [..., 3]:
        the position error the policy sees at the blocks it grasps or pushes (blocks are 150 mm long, so
        1 deg of yaw is up to 1.3 mm at their ends)."""
        mov, _, c = self.template(task_variant)
        C = box_corners(mov, self.dims).reshape(-1, 3)
        a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
        pa = np.stack([move_corners(C, c, x)[:, :2] for x in a])
        pb = np.stack([move_corners(C, c, x)[:, :2] for x in b])
        return 1e3 * np.linalg.norm(pa - pb, axis=-1).max(-1)

    def photometric_reference(self, task_variant: str, camera: str) -> np.ndarray | None:
        """Nominal per-channel quantiles [3, L] of `camera` for the variant (layouts[v]["photometric_reference"],
        scripts/fit_kinematics.py --stage photometric); None leaves the image as is."""
        ref = (self.layouts[resolve_variant(task_variant, self.layouts)].get("photometric_reference") or {}).get(camera)
        return None if ref is None or not len(self.levels) else np.array(ref, float)

    def gate_config(self, task_variant: str, result: OffsetResult | None = None) -> dict:
        """The variant's offset_gate; for a `result` from a saturated image (estimator "saturated_gate"
        min_fraction) the yaw threshold is raised to yaw_deg_min. Clipping removes the colour of the lit faces
        and leaves small towers' yaw a few degrees off (val nominal frames at gain 2: up to 3.8 deg), above the
        nominal yaw gates of 0.5-2.1 deg; a 10 deg rule rotation still passes unchanged."""
        gate = self.layouts[resolve_variant(task_variant, self.layouts)].get("offset_gate") or {}
        sg = self.saturated_gate
        if gate and sg and result is not None and result.saturated > sg["min_fraction"]:
            gate = gate | {"yaw_deg": max(gate["yaw_deg"], sg["yaw_deg_min"])}
        return gate

    def estimate(self, task_variant: str, images: dict, *, yaw: bool | None = None,
                 template: tuple | None = None) -> OffsetResult:
        """yaw=False searches translation only (dpsi fixed at 0); None takes the variant's gate setting.
        `template` = (searched poses [N,7], context poses [M,7]) replaces the nominal layout (e.g. one held block
        against the rest of the scene); the per-variant bias then does not apply."""
        t0 = time.perf_counter()
        if yaw is None:
            yaw = self.gate_config(task_variant).get("yaw", True)
        if template is None:
            mov, fix, c = self.template(task_variant)
        else:
            mov, fix = (np.asarray(p, float) for p in template)
            c = mov[:, :2].mean(0)
        Cm, Cf = box_corners(mov, self.dims), box_corners(fix, self.dims)
        norm = {k: normalise(images[k], self.photometric_reference(task_variant, k), self.levels) for k in EXTERNAL}
        masks = {k: wood_mask(norm[k][0]) for k in EXTERNAL}
        edges = {k: edge_map(norm[k][0]) if self.edge_weight > 0 else None for k in EXTERNAL}
        coarse = [_View(self.cams[k], masks[k], Cf, Cm, c, self.reach_m, 0.5, edges[k], self.edge_weight, norm[k][1])
                  for k in EXTERNAL]

        def f(x, views):
            C = move_corners(Cm, c, x)
            return -sum(v.score(C) for v in views)

        # 1) grid at half resolution: translation 5 mm, yaw 5 deg (the score is flat in yaw for small towers,
        #    a yaw-0 grid followed by local search misses 10 deg rotations); all cells projected at once
        g = np.arange(-0.020, 0.0201, 0.005)
        psis = np.radians([-10.0, -5.0, 0.0, 5.0, 10.0]) if yaw else [0.0]
        X = np.array([(dx, dy, psi) for psi in psis for dx in g for dy in g])
        CX = move_corners(Cm[None], c, X.T[:, :, None, None])
        pxs = [v._px(CX) for v in coarse]
        vals = [-sum(v.score(None, p[i]) for v, p in zip(coarse, pxs)) for i in range(len(X))]
        # 2) pattern search from the three best cells, half resolution
        starts = [X[i] for i in sorted(range(len(X)), key=lambda i: (vals[i], *X[i]))[:3]]
        ys = 1.0 if yaw else 0.0  # a zero step is below any min_step: the yaw coordinate is never moved
        cands = sorted((self._pattern(f, s, coarse, (0.0025, 0.0025, ys * np.radians(2.0)), (1e-3, 1e-3, np.radians(0.5)))
                        for s in starts), key=lambda r: r[0])
        # 3) refine at full resolution
        fine = [_View(self.cams[k], masks[k], Cf, Cm, c, self.reach_m, 1.0, edges[k], self.edge_weight, norm[k][1])
                for k in EXTERNAL]
        _, x = self._pattern(f, cands[0][1], fine, (5e-4, 5e-4, ys * np.radians(0.25)), (1e-4, 1e-4, np.radians(0.05)))
        f1 = {k: self._f1(v, move_corners(Cm, c, x)) for k, v in zip(EXTERNAL, fine)}
        if template is None:
            x = x - self.bias(task_variant) * (1.0, 1.0, float(yaw))
        off = Offset(float(x[0]), float(x[1]), float(x[2]))
        conf = min(f1.values())
        plausible = np.hypot(off.dx, off.dy) <= np.sqrt(2) * self.reach_m and abs(off.dpsi) <= self.max_psi
        sat = max(1.0 - float(v.valid.mean()) for v in fine)
        return OffsetResult(off, conf, bool(conf >= self.min_confidence and plausible), time.perf_counter() - t0, f1,
                            sat)

    @staticmethod
    def _pattern(f, x0, views, step, min_step):
        x = np.array(x0, float)
        step, min_step = np.array(step, float), np.array(min_step, float)
        best = f(x, views)
        while np.any(step >= min_step):
            improved = False
            for i in range(3):
                if step[i] < min_step[i]:
                    continue
                for sgn in (1, -1):
                    y = x.copy()
                    y[i] += sgn * step[i]
                    v = f(y, views)
                    if v < best:
                        best, x, improved = v, y, True
                        break
            if not improved:
                step = step / 2
        return best, x

    @staticmethod
    def _f1(view: _View, corners: np.ndarray) -> float:
        S = view.render(corners, np.zeros(view.shape, np.float32)) > 0.5
        S |= view.fixed > 0.5
        S &= view.valid
        W = view.mask
        tp = (S & W).sum()
        return float(2 * tp / max(S.sum() + W.sum(), 1))


def true_offset(task_variant_blocks: dict, ids: list[str], poses: np.ndarray) -> Offset:
    """Ground truth T of a frame (sim/block_poses) relative to the nominal layout, in the estimator's
    convention: yaw from the block orientations, translation of the movable centroid after rotating the
    nominal blocks by that yaw about the nominal centre c."""
    nom = np.array([task_variant_blocks[k] for k in ids if is_movable(k)], float)
    act = np.array([poses[i] for i, k in enumerate(ids) if is_movable(k)], float)
    c = nom[:, :2].mean(0)
    Rr = quat_xyzw_to_matrix(act[:, 3:7]) @ np.swapaxes(quat_xyzw_to_matrix(nom[:, 3:7]), 1, 2)
    psi = float(np.arctan2(Rr[:, 1, 0].mean(), Rr[:, 0, 0].mean()))
    rot = move_corners(nom[:, :3], c, (0.0, 0.0, psi))
    d = act[:, :2].mean(0) - rot[:, :2].mean(0)
    return Offset(float(d[0]), float(d[1]), psi)


def synthesize_shift(image: np.ndarray, cam: Camera, poses: np.ndarray, movable: np.ndarray, c: np.ndarray,
                     off: Offset, dims) -> np.ndarray:
    """Pixel-level synthetic observation: move the movable blocks of a real frame by `off` about c.

    Each movable block is re-imaged through the homography of the horizontal plane at its own centre height
    (exact for its top face, approximate for the sides), pasted far-to-near; the uncovered area is inpainted.
    poses [N,7] are the frame's actual block poses, `movable` a boolean index into them."""
    h, w = image.shape[:2]
    x = (off.dx, off.dy, off.dpsi)
    P = poses[movable]
    C_old = box_corners(P, dims)
    C_new = move_corners(C_old, c, x)

    def sil(C):
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(cam.project(C)).astype(np.int32)), 1)
        return m.astype(bool)

    old = np.zeros((h, w), bool)
    for C in C_old:
        old |= sil(C)
    new_sils = [sil(C) for C in C_new]
    new = np.any(new_sils, 0)
    # inpaint old and new footprints together so the fill samples only the surroundings (table, foundation),
    # not the moved blocks; the new footprint is overwritten below
    hole = cv2.dilate((old | new).astype(np.uint8), np.ones((5, 5), np.uint8))
    out = cv2.inpaint(np.ascontiguousarray(image), hole, 5, cv2.INPAINT_TELEA)
    depth = ((P[:, :3] - cam.t) @ cam.R_wc[:, 2])
    for i in np.argsort(-depth):  # far first, near blocks overwrite
        z = P[i, 2]
        plane = np.array([[c[0] + sx, c[1] + sy, z] for sx in (-.1, .1) for sy in (-.1, .1)])
        H, _ = cv2.findHomography(cam.project(plane).astype(np.float32),
                                  cam.project(move_corners(plane, c, x)).astype(np.float32))
        warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)
        out[new_sils[i]] = warped[new_sils[i]]
    return out
