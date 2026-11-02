"""CPU regression checks for the throughput switches patched into OpenPI.

Each switch must compute the same function as the upstream path; these tests pin that down on small inputs.
"""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0 as pi0_module
from openpi.models import siglip
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "second_base_rgb")


def _fake_img(images, train=False):
    # Per-image deterministic "encoder": 3 tokens that depend only on that image.
    del train
    pooled = images.mean(axis=(1, 2))
    return jnp.stack([pooled, 2 * pooled, pooled**2], axis=1), None


def _fake_llm(tokens, method=None):
    assert method == "embed"
    return jnp.repeat(tokens[..., None].astype(jnp.float32), 3, axis=-1)


@pytest.mark.parametrize("batch", [1, 3])
def test_batched_view_encoding_matches_per_view(monkeypatch, batch):
    config = Pi0Config(pi05=True, image_keys=CAMERAS)
    obs = config.fake_obs(batch_size=batch)
    rng = np.random.default_rng(0)
    images = {k: jnp.asarray(rng.uniform(-1, 1, v.shape), jnp.float32) for k, v in obs.images.items()}
    masks = {k: jnp.asarray(rng.integers(0, 2, v.shape).astype(bool)) for k, v in obs.image_masks.items()}
    obs = dataclasses.replace(obs, images=images, image_masks=masks)
    harness = SimpleNamespace(PaliGemma=SimpleNamespace(img=_fake_img, llm=_fake_llm))
    outputs = {}
    for batched in (False, True):
        monkeypatch.setattr(pi0_module, "BATCH_IMAGE_VIEWS", batched)
        outputs[batched] = Pi0.embed_prefix(harness, obs)
    for a, b in zip(outputs[False], outputs[True], strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_natural_layout_attention_matches_upstream(monkeypatch):
    config = gemma.get_config("dummy")
    assert config.num_kv_heads == 1
    module = gemma.Attention(configs=[config])
    b, t = 2, 7
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((b, t, config.width)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(t), (b, t))
    mask = jnp.asarray(rng.integers(0, 2, (b, 1, t, t)).astype(bool)) | jnp.eye(t, dtype=bool)
    params = module.init(jax.random.key(0), [x], positions, mask, None)
    outputs = {}
    for natural in (False, True):
        monkeypatch.setattr(gemma, "ATTENTION_NATURAL_LAYOUT", natural)
        (out,), (k, v) = module.apply(params, [x], positions, mask, None)
        outputs[natural] = (out, k, v)
    for a, c in zip(outputs[False], outputs[True], strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-5, atol=1e-5)


def test_fused_siglip_attention_rejects_masks_and_dropout():
    q = jnp.zeros((1, 4, 2, 8))
    with pytest.raises(ValueError):
        siglip._fused_attention(q, q, q, mask=jnp.ones((1, 1, 4, 4), bool))
    with pytest.raises(ValueError):
        siglip._fused_attention(q, q, q, dropout_rate=0.1, deterministic=False)
