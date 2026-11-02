"""Image offset estimator: template renders, failure fallback, nominal first frames, pixel-level rule-box shifts."""
import io
import json
from dataclasses import astuple
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from PIL import Image

from jenga.kinematics import load_arms
from jenga.offset import (EXTERNAL, OffsetEstimator, box_corners, is_movable, move_corners, synthesize_shift,
                          true_offset)
from jenga.warp import FrameTransform, Offset, load_layouts
from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT  # env JENGA_DATA_ROOT
SPLIT = ROOT / "runs/prep/split.json"
CAMERAS = ROOT / "configs/cameras.json"
needs_config = pytest.mark.skipif(not CAMERAS.exists(), reason="configs/cameras.json missing")
needs_data = pytest.mark.skipif(not (DATA / "MANIFEST.jsonl").exists() or not SPLIT.exists(),
                                reason="needs the local TOWER release and runs/prep")
WOOD = (196, 150, 96)  # RGB, inside the wood mask
TABLE = (190, 188, 184)


def render_scene(est: OffsetEstimator, variant: str, off: Offset) -> dict:
    """Flat-shaded synthetic frame: nominal fixed blocks, movable blocks moved by `off`, grey table."""
    mov, fix, c = est.template(variant)
    corners = np.concatenate([box_corners(fix, est.dims), move_corners(box_corners(mov, est.dims), c,
                                                                        (off.dx, off.dy, off.dpsi))])
    out = {}
    for cam in EXTERNAL:
        img = np.full((480, 640, 3), TABLE, np.uint8)
        for C in corners:
            pts = np.round(est.cams[cam].project(C) * 16).astype(np.int32)
            cv2.fillConvexPoly(img, cv2.convexHull(pts), WOOD, lineType=cv2.LINE_AA, shift=4)
        out[cam] = img
    return out


def frame0(rel: str) -> tuple[dict, list, np.ndarray]:
    with h5py.File(DATA / rel, "r") as f:
        imgs = {c: np.asarray(Image.open(io.BytesIO(f["observations/images/" + c][0].tobytes())).convert("RGB"))
                for c in EXTERNAL}
        return imgs, json.loads(f["sim"].attrs["block_ids"]), f["sim/block_poses"][0].astype(float)


@needs_config
def test_template_render_recovery():
    """Camera model, template and search on flat-shaded renders. Silhouette term only: flat shading has none of
    the face edges the edge term looks for (the real-frame tests below cover it)."""
    est = OffsetEstimator(correct_bias=False, edge_weight=0.0)
    for variant in ("task3-2_vertical_stacking_2", "task4-2_full_tower_pushdown_2", "task5-3_tower_insertion_3"):
        for off in (Offset(0.015, -0.015, np.radians(10)), Offset(-0.012, 0.004, np.radians(-7)), Offset()):
            r = est.estimate(variant, render_scene(est, variant, off))
            assert r.ok and r.confidence > 0.9
            assert np.hypot(r.offset.dx - off.dx, r.offset.dy - off.dy) < 1e-3
            assert abs(np.degrees(r.offset.dpsi - off.dpsi)) < 1.0


@needs_config
def test_low_confidence_falls_back_to_identity():
    est = OffsetEstimator()
    blank = {c: np.full((480, 640, 3), TABLE, np.uint8) for c in EXTERNAL}
    assert not est.estimate("task1-3_tower_transfer_3", blank).ok
    ft = FrameTransform.from_observation({"task_variant": "task1-3_tower_transfer_3", "images": blank}, estimator=est)
    assert ft.identity and ft.offset == Offset()


@needs_config
@needs_data
def test_nominal_first_frames():
    """Unshifted held-out scenes, one per variant: the estimate is close to the layout jitter, a nominal scene is
    never rotated, and whatever translation the gate applies moves it toward its true layout, not away."""
    est, arms, layouts = OffsetEstimator(), load_arms(), load_layouts()
    split = json.loads(SPLIT.read_text())
    errs = []
    for variant, t in sorted(split["tasks"].items()):
        imgs, ids, P = frame0(t["val"][0])
        truth = astuple(true_offset(est.layouts[variant]["blocks"], ids, P))
        ft = FrameTransform.from_observation({"task_variant": variant, "images": imgs}, estimator=est, arms=arms,
                                             layouts=layouts)
        raw = astuple(ft.raw_offset)
        errs.append(np.hypot(raw[0] - truth[0], raw[1] - truth[1]))
        assert ft.offset.dpsi == 0, (variant, ft.raw_offset)
        fp = est.footprint_error_mm(variant, [astuple(ft.offset), (0, 0, 0)], truth)
        assert fp[0] <= fp[1] + 0.5, (variant, ft.raw_offset, fp)
    assert np.mean(errs) < 2e-3 and np.max(errs) < 4e-3


@needs_config
@needs_data
def test_colour_cast_nominal_frame():
    """A dark warm cast inside the training lighting augmentation (gain 0.3, 3000 K; before the photometric
    normalisation every nominal frame fell back to the identity) is estimated like the original frame."""
    from jenga.dataset import kelvin_rgb
    est = OffsetEstimator()
    split = json.loads(SPLIT.read_text())
    variant = next(k for k in split["tasks"] if k.startswith("task3-2"))
    imgs, _, _ = frame0(split["tasks"][variant]["val"][0])
    cast = {c: np.clip(np.round(im * (0.3 * kelvin_rgb(3000.0))), 0, 255).astype(np.uint8) for c, im in imgs.items()}
    r0, r = est.estimate(variant, imgs), est.estimate(variant, cast)
    assert r.ok and r.confidence > 0.8, r
    assert np.hypot(r.offset.dx - r0.offset.dx, r.offset.dy - r0.offset.dy) < 1e-3
    assert abs(np.degrees(r.offset.dpsi - r0.offset.dpsi)) < 1.0


@needs_config
@needs_data
def test_rule_box_shift_still_corrected():
    """A 15 mm / 10 deg pixel-level shift passes the gate and is corrected. task3-2 and task5-2 are the small
    assemblies whose silhouette alone left yaw/translation ambiguous (task5-2 up to 38 mm corner error before the
    edge term); task4-3 is the push family."""
    est, arms, layouts = OffsetEstimator(), load_arms(), load_layouts()
    split = json.loads(SPLIT.read_text())
    for prefix in ("task1-3", "task3-2", "task4-3", "task5-2"):
        variant, rel = next((k, t["val"][0]) for k, t in split["tasks"].items() if k.startswith(prefix))
        imgs, ids, P = frame0(rel)
        t0 = true_offset(est.layouts[variant]["blocks"], ids, P)
        _, _, c = est.template(variant)
        mv = np.array([is_movable(b) for b in ids])
        for d1 in ((0.015, -0.015, np.radians(10)), (-0.015, 0.015, np.radians(-10))):
            syn = {cam: synthesize_shift(imgs[cam], est.cams[cam], P, mv, c, Offset(*d1), est.dims) for cam in EXTERNAL}
            ft = FrameTransform.from_observation({"task_variant": variant, "images": syn}, estimator=est, arms=arms,
                                                 layouts=layouts)
            cs, sn = np.cos(d1[2]), np.sin(d1[2])
            truth = (cs * t0.dx - sn * t0.dy + d1[0], sn * t0.dx + cs * t0.dy + d1[1], t0.dpsi + d1[2])
            assert not ft.identity and ft.offset == ft.raw_offset  # beyond 2 tau: applied unchanged
            fp = est.footprint_error_mm(variant, [astuple(ft.offset), (0, 0, 0)], truth)
            assert fp[0] < 8.0 and fp[0] < 0.3 * fp[1], (variant, d1, ft.offset, fp)
            assert abs(np.degrees(ft.offset.dpsi - truth[2])) < 3.0, (variant, d1, ft.offset)
