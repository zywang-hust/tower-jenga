"""Exercise patched OpenPI camera routing on CPU, without constructing model weights."""

import dataclasses
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config


CAMERAS = (*model.IMAGE_KEYS, "second_base_rgb")


class PrefixReached(Exception):
    pass


class OpenPICameraTests(unittest.TestCase):
    def test_default_compatibility_and_four_view_spec(self):
        default = Pi0Config()
        self.assertEqual(default.image_keys, tuple(model.IMAGE_KEYS))
        self.assertEqual(tuple(default.inputs_spec()[0].images), tuple(model.IMAGE_KEYS))
        config = Pi0Config(image_keys=CAMERAS)
        observation, actions = config.inputs_spec(batch_size=2)
        self.assertEqual(tuple(observation.images), CAMERAS)
        self.assertEqual(tuple(observation.image_masks), CAMERAS)
        self.assertEqual(observation.images[CAMERAS[-1]].shape, (2, 224, 224, 3))
        self.assertEqual(actions.shape[-1], 32)

    def test_invalid_camera_configuration(self):
        for keys in ((), ("a", "a"), ("",), ("   ",), (None,), "camera"):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                Pi0Config(image_keys=keys)
        # Normalize an accepted list to an immutable configuration value.
        self.assertEqual(Pi0Config(image_keys=list(CAMERAS)).image_keys, CAMERAS)

    def test_both_model_paths_preserve_fourth_view_and_mask(self):
        config = Pi0Config(image_keys=CAMERAS)
        observation = config.fake_obs()
        images = dict(observation.images)
        images[CAMERAS[-1]] = jnp.full_like(images[CAMERAS[-1]], 0.75)
        masks = dict(observation.image_masks)
        masks[CAMERAS[-1]] = jnp.array([False])
        observation = dataclasses.replace(observation, images=images, image_masks=masks)
        seen = []

        def capture_prefix(processed):
            seen.append(processed)
            raise PrefixReached

        # Stop only at the first learned layer; execute each real method's preprocessing.
        harness = SimpleNamespace(
            image_keys=config.image_keys,
            image_geometry_augment=True,
            action_dim=32,
            action_horizon=config.action_horizon,
            embed_prefix=capture_prefix,
        )
        rng = jax.random.key(0)
        actions = jnp.zeros((1, config.action_horizon, 32))
        with self.assertRaises(PrefixReached):
            Pi0.compute_loss(harness, rng, observation, actions, train=False)
        with self.assertRaises(PrefixReached):
            Pi0.sample_actions(harness, rng, observation, noise=actions)
        self.assertEqual(len(seen), 2)
        for processed in seen:
            self.assertEqual(tuple(processed.images), CAMERAS)
            np.testing.assert_array_equal(processed.images[CAMERAS[-1]], images[CAMERAS[-1]])
            np.testing.assert_array_equal(processed.image_masks[CAMERAS[-1]], [False])

    def test_missing_fourth_camera_is_not_silently_accepted(self):
        config = Pi0Config(image_keys=CAMERAS)
        observation = Pi0Config().fake_obs()
        with self.assertRaises(ValueError):
            model.preprocess_observation(None, observation, image_keys=config.image_keys)




def test_geometry_augment_toggle():
    """image_geometry_augment=False keeps colour jitter but no crop/rotate: a flat image stays flat."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import openpi.models.model as _model

    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "second_base_rgb")
    flat = jnp.full((8, 224, 224, 3), 0.2, jnp.float32)
    obs = _model.Observation(images={k: flat for k in keys}, image_masks={k: jnp.ones(8, bool) for k in keys},
                             state=jnp.zeros((8, 32)))
    rng = jax.random.key(0)
    off = _model.preprocess_observation(rng, obs, train=True, image_keys=keys, geometry_augment=False)
    on = _model.preprocess_observation(rng, obs, train=True, image_keys=keys)
    per_image_std = lambda x: np.asarray(x).reshape(8, -1).std(-1)  # noqa: E731
    assert per_image_std(off.images["base_0_rgb"]).max() < 1e-5
    assert per_image_std(on.images["base_0_rgb"]).max() > 1e-2  # rotation fills corners


def test_continuous_state_is_exact_noop_at_init_and_trainable():
    """pi05 continuous_state adds a zero-initialized state projection: outputs equal the plain model's at init, and
    the projection receives a gradient (so finetuning can use the in-bin residual the discrete token drops)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from openpi.models.pi0_config import Pi0Config

    kw = dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4,
              max_token_len=16, image_keys=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
              pytorch_compile_mode=None)
    plain = Pi0Config(**kw).create(jax.random.key(0))
    cont = Pi0Config(**kw, continuous_state=True).create(jax.random.key(0))
    obs = Pi0Config(**kw).fake_obs(2)
    obs = obs.replace(state=jax.random.uniform(jax.random.key(1), obs.state.shape, minval=-1, maxval=1))
    noise = jax.random.normal(jax.random.key(2), (2, 4, 32))
    t = jnp.full((2,), 0.5)
    a = plain.embed_suffix(obs, noise, t)[0]
    b = cont.embed_suffix(obs, noise, t)[0]
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def loss(model):
        return jnp.sum(model.embed_suffix(obs, noise, t)[0] ** 2)

    grads = nnx.grad(loss)(cont)
    assert float(jnp.abs(grads.state_proj.kernel.value).sum()) > 0


if __name__ == "__main__":
    unittest.main()
