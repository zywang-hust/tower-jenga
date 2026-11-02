"""Train/serve consistency: the online adapter must reproduce TowerDataset inputs bit for bit on dataset frames."""
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from jenga.hdf5_dataset import CAMERA_MAP
from jenga.instructions import TRAIN_INSTRUCTIONS
from jenga.policy import ObservationAdapter, clip_to_bounds, state16
from jenga.paths import DATA_ROOT

ROOT = Path(__file__).resolve().parents[1]
DATA = DATA_ROOT  # env JENGA_DATA_ROOT
PREP = ROOT / "runs/prep"
STARTER = ROOT / "external/tower-starter"
needs_data = pytest.mark.skipif(not (DATA / "MANIFEST.jsonl").exists() or not (PREP / "split.json").exists(),
                                reason="needs the local TOWER release and runs/prep")


def wire_observation(path: Path, frame: int) -> dict:
    """What the worker would send for this frame, through the official codec when the starter is present."""
    with h5py.File(path, "r") as f:
        q = f["observations/qpos"][frame].astype(np.float32)
        images = {cam: np.asarray(Image.open(io.BytesIO(f[f"observations/images/{cam}"][frame].tobytes())).convert("RGB"))
                  for cam in CAMERA_MAP.values()}
    obs = {
        "schema": "tower-challenge-observation-v1",
        "step_index": frame,
        "task_variant": path.parent.name,
        "instruction": "unused by prompt='train'",
        "joint_positions": np.stack([q[:7], q[8:15]]),
        "gripper_widths_m": q[[7, 15]],
        "images": images,
    }
    if STARTER.exists():
        sys.path.insert(0, str(STARTER))
        from tower_protocol import pack, unpack

        obs = unpack(pack({"observation": obs}))["observation"]
    return obs


@needs_data
def test_adapter_matches_training_inputs_bitwise():
    import openpi.shared.normalize as normalize

    from jenga.dataset import TowerDataset

    stats = normalize.load(PREP / "norm_stats")
    split = json.loads((PREP / "split.json").read_text())
    adapter = ObservationAdapter(stats, prompt="train", jpeg_quality=None)
    for variant in sorted(split["tasks"]):
        ep = sorted(split["tasks"][variant]["val"])[0]
        ds = TowerDataset(DATA, [ep], stats)
        for i in (0, len(ds) // 2):
            ref = ds[i]
            frame = int(ds.frames.index[i, 1])
            obs = wire_observation(DATA / ep, frame)
            got = adapter(obs)
            for key in ref["image"]:
                np.testing.assert_array_equal(got["image"][key], ref["image"][key], err_msg=f"{ep} {key}")
            for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
                np.testing.assert_array_equal(got[key], ref[key], err_msg=f"{ep} {key}")
            # Action decoding inverts the training label transform back to the absolute leader commands.
            raw = ds.frames[i]
            decoded = adapter.decode(ref["actions"], state16(obs))
            np.testing.assert_allclose(decoded, raw["actions"], atol=2e-6, err_msg=ep)
            assert TRAIN_INSTRUCTIONS[variant.split("_")[0]] == raw["prompt"]


def test_clip_to_bounds_keeps_grippers_physical():
    actions = np.zeros((2, 16), np.float32)
    actions[0, 7], actions[1, 15], actions[0, 0] = 0.2, -0.1, 9.0
    out = clip_to_bounds(actions, -np.full(16, 3.0), np.full(16, 3.0))
    assert out.dtype == np.dtype("<f4") and out[0, 0] == 3.0 and out[0, 7] == pytest.approx(0.1) and out[1, 15] == 0.0


@needs_data
def test_official_prompt_matches_wire_instruction():
    import openpi.shared.normalize as normalize

    from jenga.instructions import OFFICIAL_INSTRUCTIONS
    from jenga.dataset import TowerDataset

    stats = normalize.load(PREP / "norm_stats")
    split = json.loads((PREP / "split.json").read_text())
    adapter = ObservationAdapter(stats, prompt="official", jpeg_quality=None)
    for variant in sorted(split["tasks"]):
        ep = sorted(split["tasks"][variant]["val"])[0]
        ref = TowerDataset(DATA, [ep], stats, prompt="official")[0]
        obs = wire_observation(DATA / ep, 0) | {"instruction": OFFICIAL_INSTRUCTIONS[variant[:5]]}
        got = adapter(obs)
        np.testing.assert_array_equal(got["tokenized_prompt"], ref["tokenized_prompt"], err_msg=ep)


@needs_data
def test_integer_bin_state_noise_moves_the_state_token_by_whole_bins():
    import openpi.shared.normalize as normalize

    from jenga.dataset import TowerDataset
    from jenga.transforms import JOINTS

    stats = normalize.load(PREP / "norm_stats")
    ep = sorted(json.loads((PREP / "split.json").read_text())["tasks"]["task4-2_full_tower_pushdown_2"]["val"])[0]
    clean = TowerDataset(DATA, [ep], stats)
    noisy = TowerDataset(DATA, [ep], stats, state_noise_bins=2, state_noise_p=1.0)
    edges = np.linspace(-1, 1, 257)[:-1]  # the PaliGemma tokenizer's state discretization
    shifts = []
    for i in range(0, 200, 7):
        a, b = clean[i]["state"][JOINTS], noisy[i]["state"][JOINTS]
        k = np.digitize(b, edges) - np.digitize(a, edges)  # every joint, saturated ones included
        np.testing.assert_allclose((b - a) * 128.0, k, atol=1e-3)
        shifts.extend(k.tolist())
    assert set(shifts) <= {-2, -1, 0, 1, 2} and len(set(shifts)) == 5
