import numpy as np

from scf.coarsen import (
    always_on_features, dominant_feature, build_argmax_space,
)


def test_always_on_features_detects_constant_firing():
    support = np.array([
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 0, 1, 1],
    ])
    # features 0 and 2 fire on every row
    assert sorted(always_on_features(support).tolist()) == [0, 2]


def test_dominant_feature_is_argmax_excluding():
    mags = np.array([
        [9.0, 1.0, 2.0],   # argmax 0, but if 0 excluded -> 2
        [0.5, 4.0, 1.0],   # argmax 1
    ])
    assert dominant_feature(mags).tolist() == [0, 1]
    assert dominant_feature(mags, exclude=[0]).tolist() == [2, 1]


def test_build_argmax_space_excludes_always_on_and_pools_tail():
    rng = np.random.default_rng(0)
    N, F = 500, 50
    support = (rng.random((N, F)) < 0.1).astype(np.uint8)  # ~sparse firing
    support[:, 7] = 1                                       # feature 7 always on
    mags = rng.random((N, F)).astype(np.float32) * support
    mags[:, 7] = 10.0                                       # feature 7 always the max
    space, ids = build_argmax_space(mags, support=support, n_states=8)
    # always-on feature 7 must be excluded from the argmax
    assert 7 in space.extra["excluded_features"]
    assert 7 not in space.extra["named_features"]
    # exactly n_states states, every id valid, 'other' present
    assert space.n_states == 8
    assert ids.min() >= 0 and ids.max() < 8
    assert space.id_to_label[-1] == "other"


def test_build_argmax_space_is_deterministic():
    rng = np.random.default_rng(1)
    mags = rng.random((300, 40)).astype(np.float32)
    s1, i1 = build_argmax_space(mags, n_states=16, exclude=[])
    s2, i2 = build_argmax_space(mags, n_states=16, exclude=[])
    assert np.array_equal(i1, i2)             # no randomness, unlike KMeans
