import numpy as np

from jenga.stitching import ChunkFader


def test_fader_reproduces_consistent_plan_and_ramps_boundary_steps():
    traj = np.cumsum(np.random.default_rng(0).normal(0, 0.01, (200, 16)), 0).astype(np.float32)
    fader, k = ChunkFader(4, smooth=0), 8
    executed = [fader(traj[c * k : c * k + 32], k)[:k] for c in range(5)]
    np.testing.assert_allclose(np.concatenate(executed), traj[:40], atol=1e-6)

    fader.reset()
    np.testing.assert_array_equal(fader(np.zeros((32, 2)), k), 0.0)  # first chunk passes through
    out = fader(np.ones((32, 2)), k)  # a 1.0 step at the boundary becomes five 0.2 steps
    np.testing.assert_allclose(out[:5, 0], [0.2, 0.4, 0.6, 0.8, 1.0], atol=1e-6)
    out = fader(np.zeros((32, 2)), 30)  # only 2 overlapping steps remain
    np.testing.assert_allclose(out[:3, 0], [0.8, 0.6, 0.0], atol=1e-6)


def test_fader_skips_unaligned_overlap():
    a, b = np.zeros((32, 16), np.float32), np.ones((32, 16), np.float32)
    # re-sent frame, step went backwards, gap past the previous chunk, more rows than were sent (issued=8)
    for executed in (0, -3, 32, 40, 12):
        f = ChunkFader(fade=4, issued=8)
        f(a, executed=8)
        np.testing.assert_array_equal(f(b, executed=executed), b)


def test_smoothing_keeps_linear_motion_and_grippers():
    t = np.arange(32, dtype=np.float32)[:, None]
    chunk = np.tile(0.01 * t, (1, 16))
    chunk[:, 7] = (t[:, 0] > 10) * 0.1  # a gripper step must stay a step
    rough = chunk + np.random.default_rng(0).normal(0, 4e-4, chunk.shape).astype(np.float32) * (np.arange(16) != 7)
    out = ChunkFader(fade=2, smooth=9)(rough, executed=8)
    np.testing.assert_array_equal(out[:, 7], chunk[:, 7])
    assert np.abs(np.diff(out[:, :7], 2, axis=0)).mean() < 0.5 * np.abs(np.diff(rough[:, :7], 2, axis=0)).mean()
    np.testing.assert_allclose(out[:, :7], chunk[:, :7], atol=1e-3)
