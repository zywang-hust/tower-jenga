import numpy as np

from jenga.recipe import TASK_SCORE_WEIGHTS, sample_weights, variant_mix

# Frame shares (%) from the data card, as tabulated in docs/recipe.md.
FRAMES = {"1-2": 11.2, "1-3": 16.8, "2-2": 5.7, "2-3": 8.9, "3-2": 5.6, "3-3": 8.1,
          "4-2": 10.1, "4-3": 14.8, "5-2": 5.1, "5-3": 13.7}
SCORE = {v: TASK_SCORE_WEIGHTS[int(v[0])] for v in FRAMES}
RECIPE = {"1-2": 8.3, "1-3": 10.1, "2-2": 5.9, "2-3": 7.4, "3-2": 5.8, "3-3": 7.0,
          "4-2": 13.6, "4-3": 16.5, "5-2": 9.6, "5-3": 15.8}


def test_variant_mix_reproduces_recipe_table():
    mix = variant_mix(FRAMES, SCORE)
    for v, pct in RECIPE.items():
        assert abs(100 * mix[v] - pct) < 0.11, (v, 100 * mix[v], pct)
    assert abs(100 * sum(mix[v] for v in mix if v[0] in "45") - 55.5) < 0.2


def test_sample_weights_hit_target_mix():
    variants = ["1-2"] * 300 + ["5-2"] * 100
    score = {"1-2": 1.0, "5-2": 3.0}
    w = sample_weights(variants, score)
    mix = variant_mix({"1-2": 300, "5-2": 100}, score)
    assert np.isclose(w[:300].sum() / w.sum(), mix["1-2"])
    assert np.isclose(w[300:].sum() / w.sum(), mix["5-2"])



def test_emphasize_keeps_group_totals():
    from jenga.recipe import emphasize

    w = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])
    groups = np.array([0, 0, 0, 0, 1, 1])
    mask = np.array([True, False, False, False, True, True])  # group 1 is all contact: unchanged
    out = emphasize(w, groups, mask, 3.0)
    np.testing.assert_allclose([out[:4].sum(), out[4:].sum()], [0.4, 0.6])
    np.testing.assert_allclose(out[0] / out[1], 3.0)
    np.testing.assert_allclose(out[4:], w[4:])
