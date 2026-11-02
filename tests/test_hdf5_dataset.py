import pickle

import h5py
import numpy as np

from jenga.hdf5_dataset import MODEL_CAMERAS, TowerFrames, window_starts
from fixtures import generate


def test_window_starts_require_full_valid_horizon():
    valid = np.array([True] * 5 + [False])
    np.testing.assert_array_equal(window_starts(valid, 3), [0, 1, 2])
    assert window_starts(valid[:2], 3).size == 0


def test_frames_match_file_contents(tmp_path):
    root = tmp_path / "fixtures"
    generate(root, episodes=2, frames=40, size=32)
    files = sorted(root.glob("*.hdf5"))
    ds = TowerFrames(files, horizon=32, image_size=32)
    assert len(ds) == 2 * (39 - 32 + 1)
    sample = ds[len(ds) - 1]
    assert set(sample["images"]) == set(MODEL_CAMERAS)
    assert all(img.shape == (32, 32, 3) and img.dtype == np.uint8 for img in sample["images"].values())
    with h5py.File(files[sample["file_id"]], "r") as f:
        np.testing.assert_array_equal(sample["state"], f["observations/qpos"][sample["frame"]])
        np.testing.assert_array_equal(sample["actions"], f["leader/qpos"][sample["frame"]:sample["frame"] + 32])
    # Picklable for DataLoader workers even after handles were opened.
    clone = pickle.loads(pickle.dumps(ds))
    np.testing.assert_array_equal(clone[0]["actions"], ds[0]["actions"])


def test_tail_hold_repeats_last_valid_command(tmp_path):
    root = tmp_path / "fixtures"
    generate(root, episodes=2, frames=40, size=32)
    files = sorted(root.glob("*.hdf5"))
    ds = TowerFrames(files[:1], horizon=32, image_size=32, tail_hold=True)
    with h5py.File(files[0], "r") as f:
        valid = f["sim/action_valid"][:]
        leader = f["leader/qpos"][:]
    last = int(np.flatnonzero(valid)[-1])
    assert len(ds) == last + 1  # every valid row is a start, not only full windows
    tail = ds[len(ds) - 1]["actions"]
    assert tail.shape == (32, 16)
    np.testing.assert_array_equal(tail, np.repeat(leader[last : last + 1], 32, axis=0))
    mid = ds[last - 4]["actions"]
    np.testing.assert_array_equal(mid[:5], leader[last - 4 : last + 1])
    np.testing.assert_array_equal(mid[5:], np.repeat(leader[last : last + 1], 27, axis=0))
    np.testing.assert_array_equal(ds[0]["actions"], leader[:32])


def test_decode_matches_openpi_resize_with_pad(tmp_path):
    from openpi_client.image_tools import resize_with_pad

    root = tmp_path / "fixtures"
    generate(root, episodes=2, frames=40, size=48)
    files = sorted(root.glob("*.hdf5"))
    ds = TowerFrames(files, horizon=32, image_size=32)
    with h5py.File(files[0], "r") as f:
        raw = f["observations/images/left_back_camera"][0]
    from PIL import Image
    import io

    full = np.asarray(Image.open(io.BytesIO(raw.tobytes())).convert("RGB"))
    np.testing.assert_array_equal(ds[0]["images"]["base_0_rgb"], resize_with_pad(full, 32, 32))


def test_contact_events_and_radius():
    from jenga.hdf5_dataset import contact_events, near

    T = 20
    active = np.full(T, -1, np.int16)
    active[5:12] = 3  # block 3 held from frame 5, cleared at 12
    released = np.zeros((T, 2), bool)
    released[12:, 1] = True
    grip = np.full((T, 2), 0.1, np.float32)
    grip[6:8, 0] = [0.07, 0.04]  # closes over frames 6-7
    grip[8:13, 0] = 0.04
    grip[13, 0] = 0.1  # opens at 13
    grasp, release = contact_events(active, released, grip)
    np.testing.assert_array_equal(np.flatnonzero(grasp), [5, 6, 7])
    np.testing.assert_array_equal(np.flatnonzero(release), [12, 13])
    np.testing.assert_array_equal(np.flatnonzero(near(release, 2)), [10, 11, 12, 13, 14, 15])
    np.testing.assert_array_equal(near(np.eye(1, 3, 0, dtype=bool)[0], 5), [True] * 3)  # radius beyond the episode


def test_contact_mask_follows_sample_index(tmp_path):
    root = tmp_path / "fixtures"
    generate(root, episodes=2, frames=40, size=32)
    files = sorted(root.glob("*.hdf5"))
    for path in files:
        with h5py.File(path, "a") as f:
            active = np.full(40, -1, np.int16)
            active[10:30] = 0
            released = np.zeros((40, 1), bool)
            released[30:] = True
            f["sim/active_block"], f["sim/block_released"] = active, released
            leader = f["leader/qpos"][:]
            leader[:, [7, 15]] = 0.1  # fixture gripper wiggles; hold it still so only the block events count
            f["leader/qpos"][...] = leader
    ds = TowerFrames(files, horizon=32, image_size=32, tail_hold=True, contact_radius=3)
    expected = np.zeros(40, bool)
    expected[7:14] = expected[27:34] = True
    np.testing.assert_array_equal(ds.contact, np.concatenate([expected[:39], expected[:39]]))
    assert TowerFrames(files, horizon=32, image_size=32, tail_hold=True).contact is None


def test_derived_episode_takes_contact_from_source(tmp_path):
    root = tmp_path / "root"
    generate(root / "data/task1-2_fixture", episodes=2, frames=40, size=32)
    demo = sorted((root / "data/task1-2_fixture").glob("*.hdf5"))[0]
    with h5py.File(demo, "a") as f:
        active = np.full(40, -1, np.int16)
        active[10:30] = 0
        f["sim/active_block"], f["sim/block_released"] = active, np.zeros((40, 1), bool)
        f["leader/qpos"][:, 7] = f["leader/qpos"][:, 15] = 0.1  # still grippers: only the block events count
    copy = tmp_path / "dart/data/task1-2_fixture" / demo.name  # DART layout: no sim event fields
    copy.parent.mkdir(parents=True)
    with h5py.File(demo, "r") as src, h5py.File(copy, "w") as f:
        for key in ("observations/qpos", "leader/qpos", "sim/action_valid"):
            f[key] = src[key][:]
        f.attrs["source"] = str(demo.relative_to(root))
    both = TowerFrames([demo, copy], horizon=32, image_size=32, tail_hold=True, contact_radius=3, source_root=root)
    np.testing.assert_array_equal(both.contact[:39], both.contact[39:])
    assert both.contact.any() and not both.contact.all()
