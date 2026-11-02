import numpy as np
import pytest

from jenga.transforms import GRIPPERS, JOINTS, from_model_actions, norm_stats, to_model_actions


def test_action_round_trip_and_semantics():
    rng = np.random.default_rng(0)
    state = rng.normal(size=16).astype(np.float32)
    actions = rng.normal(size=(32, 16)).astype(np.float32)
    actions[:, GRIPPERS] = rng.uniform(0, 0.08, size=(32, 2))
    model = to_model_actions(actions, state, gripper_max=0.08)
    np.testing.assert_allclose(model[:, JOINTS], actions[:, JOINTS] - state[JOINTS], atol=1e-6)
    np.testing.assert_allclose(to_model_actions(np.full((1, 16), 0.08), state, 0.08)[0, GRIPPERS], 0.0, atol=1e-7)
    np.testing.assert_allclose(to_model_actions(np.zeros((1, 16)), state, 0.08)[0, GRIPPERS], 1.0)
    np.testing.assert_allclose(from_model_actions(model, state, 0.08), actions, atol=1e-6)


def test_action_transforms_batch_and_reject_bad_width():
    state = np.zeros((4, 16), np.float32)
    actions = np.ones((4, 32, 16), np.float32)
    assert to_model_actions(actions, state, 1.0).shape == (4, 32, 16)
    with pytest.raises(ValueError):
        to_model_actions(actions, state, 0.0)


def test_norm_stats_quantiles():
    x = np.arange(1000, dtype=np.float32)[:, None].repeat(3, 1)
    s = norm_stats([x[:500], x[500:]])
    np.testing.assert_allclose(s["mean"], 499.5, rtol=1e-4)
    assert abs(float(s["q01"][0]) - 10) < 3 and abs(float(s["q99"][0]) - 989) < 3
