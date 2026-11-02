import numpy as np
import pytest

from jenga.dataset import ACTION_DIM, TowerDataset, collate, kelvin_rgb, lighting
from jenga.paths import OPENPI_DATA_HOME
from fixtures import generate
from jenga.transforms import GRIPPER_FULL_WIDTH_M, from_model_actions

TOKENIZER = OPENPI_DATA_HOME / "big_vision/paligemma_tokenizer.model"


def test_kelvin_neutral_and_lighting_shared_across_views():
    np.testing.assert_allclose(kelvin_rgb(6500.0), 1.0)
    assert kelvin_rgb(3000.0)[2] < 1.0 < kelvin_rgb(3000.0)[0] + 1e-6  # warm light: less blue
    images = {k: np.full((4, 4, 3), 100, np.uint8) for k in ("a", "b")}
    out = lighting(images, np.random.default_rng(1), p=1.0)
    np.testing.assert_array_equal(out["a"], out["b"])
    assert lighting(images, np.random.default_rng(1), p=0.0) is images


@pytest.mark.skipif(not TOKENIZER.exists(), reason="PaliGemma tokenizer not cached")
def test_sample_round_trips_to_physical_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(TOKENIZER.parents[1]))
    from openpi.shared.normalize import NormStats

    root = tmp_path / "root"
    generate(root / "task4-2_fixture", episodes=2, frames=40, size=32)
    files = sorted(str(p.relative_to(root)) for p in (root / "task4-2_fixture").glob("*.hdf5"))
    ones = np.ones(16, np.float32)
    stats = {k: NormStats(mean=0 * ones, std=ones, q01=-2 * ones, q99=2 * ones) for k in ("state", "actions")}
    ds = TowerDataset(root, files, stats)
    assert ds.sample_variants()[0] == "task4-2"
    w = ds.sampler_weights()
    np.testing.assert_allclose(w.sum(), 1.0)
    x = ds[5]
    assert x["actions"].shape == (32, ACTION_DIM) and x["state"].shape == (ACTION_DIM,)
    assert not x["actions"][:, 16:].any() and not x["state"][16:].any()
    raw = ds.frames[5]
    model = (x["actions"][:, :16] + 1) / 2 * (4 + 1e-6) - 2  # un-normalize with q01=-2, q99=2
    np.testing.assert_allclose(from_model_actions(model, raw["state"], GRIPPER_FULL_WIDTH_M), raw["actions"], atol=1e-5)
    batch = collate([ds[0], ds[1]])
    assert batch["image"]["base_0_rgb"].shape == (2, 224, 224, 3) and batch["tokenized_prompt"].shape == (2, 128)


@pytest.mark.skipif(not TOKENIZER.exists(), reason="PaliGemma tokenizer not cached")
def test_state_noise_moves_whole_bins_and_keeps_absolute_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(TOKENIZER.parents[1]))
    from openpi.shared.normalize import NormStats

    from jenga.transforms import JOINTS

    root = tmp_path / "root"
    generate(root / "task4-2_fixture", episodes=2, frames=40, size=32)
    files = sorted(str(p.relative_to(root)) for p in (root / "task4-2_fixture").glob("*.hdf5"))[:1]
    ones = np.ones(16, np.float32)
    stats = {k: NormStats(mean=0 * ones, std=ones, q01=-2 * ones, q99=2 * ones) for k in ("state", "actions")}
    ds = TowerDataset(root, files, stats, state_noise_bins=2, state_noise_p=1.0)
    bin_rad = (4.0 + 1e-6) / 256.0
    s = stats["state"]
    eps = []
    for i in range(0, 30):
        raw, item = ds.frames[i], ds[i]
        seen = (item["state"][:16] + 1.0) / 2.0 * (s.q99 - s.q01 + 1e-6) + s.q01
        eps.append(seen[JOINTS] - raw["state"][JOINTS])
        noisy = raw["state"].copy()
        noisy[JOINTS] = seen[JOINTS]
        model_space = (item["actions"][:, :16] + 1.0) / 2.0 * (s.q99 - s.q01 + 1e-6) + s.q01
        np.testing.assert_allclose(from_model_actions(model_space, noisy, GRIPPER_FULL_WIDTH_M), raw["actions"],
                                   atol=5e-5)
    k = np.concatenate(eps) / bin_rad
    np.testing.assert_allclose(k, np.round(k), atol=1e-3)  # whole bins
    assert set(np.round(k).astype(int).tolist()) == {-2, -1, 0, 1, 2}


def _stats():
    from openpi.shared.normalize import NormStats

    ones = np.ones(16, np.float32)
    return {k: NormStats(mean=0 * ones, std=ones, q01=-2 * ones, q99=2 * ones) for k in ("state", "actions")}


def test_synthetic_share_within_each_generated_variant(tmp_path):
    from jenga.recipe import TASK_SCORE_WEIGHTS, sample_weights

    root, extra = tmp_path / "root", tmp_path / "dart"
    generate(root / "task1-2_fixture", episodes=3, frames=40, size=32)
    generate(root / "task4-2_fixture", episodes=2, frames=40, size=32)
    generate(extra / "data/task4-2_fixture", episodes=2, frames=60, size=32)
    files = sorted(str(p.relative_to(root)) for p in root.glob("*/*.hdf5"))
    synthetic = sorted(extra.glob("data/*/*.hdf5"))
    plain = TowerDataset(root, files, _stats())
    variants = plain.sample_variants()
    score = {v: TASK_SCORE_WEIGHTS[int(v[4])] for v in variants}
    np.testing.assert_array_equal(plain.sampler_weights(), sample_weights(variants, score))  # unused: as before

    ds = TowerDataset(root, files, _stats(), synthetic_files=synthetic)
    assert ds.sample_variants()[-1] == "task4-2" and len(ds) == len(plain) + 2 * 59
    w = ds.sampler_weights(synthetic_frac=0.3)
    syn = ds.synthetic[ds.frames.index[:, 0]]
    t4 = np.asarray(ds.sample_variants()) == "task4-2"
    p, pv = plain.sampler_weights(), np.asarray(variants) == "task4-2"
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w[t4].sum(), p[pv].sum())  # variant shares unchanged
    np.testing.assert_allclose(w[syn].sum(), 0.3 * p[pv].sum())  # task4-2 has copies: 30% of its draws
    np.testing.assert_allclose(w[~syn & ~t4], p[~pv])  # task1-2 has none: all its draws stay on demos
    assert np.ptp(w[syn]) < 1e-12  # uniform within the variant's synthetic samples
    with pytest.raises(ValueError):
        ds.sampler_weights()  # synthetic files given but not mixed in
    with pytest.raises(ValueError):
        plain.sampler_weights(synthetic_frac=0.3)


def test_contact_weight_keeps_variant_shares(tmp_path):
    import h5py

    root = tmp_path / "root"
    generate(root / "task1-2_fixture", episodes=2, frames=40, size=32)
    generate(root / "task5-2_fixture", episodes=2, frames=40, size=32)
    files = sorted(str(p.relative_to(root)) for p in root.glob("*/*.hdf5"))
    for f in files:
        with h5py.File(root / f, "a") as h:
            active = np.full(40, -1, np.int16)
            active[20:] = 0
            h["sim/active_block"], h["sim/block_released"] = active, np.zeros((40, 1), bool)
            h["leader/qpos"][:, 7] = h["leader/qpos"][:, 15] = 0.1  # still grippers: only the block event counts
    plain = TowerDataset(root, files, _stats()).sampler_weights()
    ds = TowerDataset(root, files, _stats(), contact_radius=3)
    w = ds.sampler_weights(contact_weight=4.0)
    variants = np.asarray(ds.sample_variants())
    for v in set(variants.tolist()):
        np.testing.assert_allclose(w[variants == v].sum(), plain[variants == v].sum())
    c = ds.frames.contact
    assert c.any() and not c.all()
    v = variants == variants[0]
    np.testing.assert_allclose(w[c & v].max() / w[~c & v].max(), 4.0)
