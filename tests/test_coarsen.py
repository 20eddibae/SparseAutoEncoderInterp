import numpy as np

from scf.coarsen import topk_state, build_topk_space, build_cluster_space


def test_topk_state_picks_largest_indices():
    mags = np.array([0.0, 5.0, 0.1, 4.0, 0.0])
    assert topk_state(mags, k=2) == (1, 3)


def test_topk_state_drops_zeros():
    mags = np.zeros(10)
    mags[3] = 1.0
    assert topk_state(mags, k=3) == (3,)


def test_build_topk_space_assigns_unique_ids():
    rng = np.random.default_rng(0)
    mags = [rng.random(20) for _ in range(50)]
    space, ids = build_topk_space(mags, k=3)
    assert space.method == "topk"
    assert len(ids) == 50
    assert max(ids) < space.n_states


def test_build_cluster_space_returns_valid_ids():
    rng = np.random.default_rng(1)
    mags = [rng.random(30) for _ in range(100)]
    space, ids = build_cluster_space(mags, n_clusters=5)
    assert space.n_states == 5
    assert all(0 <= i < 5 for i in ids)
