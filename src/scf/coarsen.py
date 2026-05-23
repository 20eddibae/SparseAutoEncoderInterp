"""
State-space coarsening for the conversation Markov chain.

The raw state is a binary vector in {0,1}^F with F = 4096. We cannot estimate
a transition kernel on 2^F states from any realistic corpus. Two coarsenings:

1. Top-k. State = sorted tuple of the k most-active feature indices in the
   turn (k small, e.g. 3–5). This collapses the chain to a chain on
   at-most C(F, k) ordered subsets — still huge, but in practice the realized
   support is small (a few thousand at most).

2. Co-activation cluster. Cluster features into G groups by k-means on the
   feature-feature co-occurrence matrix; map each turn to its dominant group.
   This gives a chain on exactly G states (e.g. G = 50).

Both return integer state ids and a reversible id <-> label mapping.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class StateSpace:
    method: str
    id_to_label: list           # state id -> serialisable label (tuple of ints or int)
    label_to_id: dict
    extra: dict                 # method-specific bookkeeping

    @property
    def n_states(self) -> int:
        return len(self.id_to_label)


# ---------------------------------------------------------------- top-k


def topk_state(magnitudes: np.ndarray, k: int) -> tuple[int, ...]:
    """Return the sorted indices of the k largest entries of `magnitudes`."""
    if k <= 0:
        raise ValueError("k must be positive")
    if magnitudes.ndim != 1:
        raise ValueError("magnitudes must be 1-D")
    idx = np.argpartition(-magnitudes, kth=min(k, magnitudes.size - 1))[:k]
    return tuple(sorted(int(i) for i in idx if magnitudes[i] > 0))


def build_topk_space(
    per_turn_magnitudes: Sequence[np.ndarray], k: int
) -> Tuple[StateSpace, List[int]]:
    """Walk the corpus once, assign each turn a state id."""
    label_to_id: dict[tuple, int] = {}
    id_to_label: list[tuple] = []
    state_ids: list[int] = []
    for mags in per_turn_magnitudes:
        lbl = topk_state(mags, k)
        sid = label_to_id.get(lbl)
        if sid is None:
            sid = len(id_to_label)
            label_to_id[lbl] = sid
            id_to_label.append(lbl)
        state_ids.append(sid)
    return (
        StateSpace(method="topk", id_to_label=id_to_label, label_to_id=label_to_id, extra={"k": k}),
        state_ids,
    )


# ---------------------------------------------------------------- cluster


def build_cluster_space(
    per_turn_magnitudes: Sequence[np.ndarray],
    n_clusters: int,
    rng_seed: int = 0,
    max_iter: int = 50,
) -> Tuple[StateSpace, List[int]]:
    """
    K-means on the feature-feature co-activation matrix, mapping each feature
    to a group. A turn's state is then the group of its argmax feature.
    """
    X = np.stack([m for m in per_turn_magnitudes], axis=0)   # (N, F)
    binary = (X > 0).astype(np.float32)
    co = binary.T @ binary                                   # (F, F)
    co /= np.maximum(np.diag(co)[:, None], 1.0)              # normalised co-act

    centers, assignments = _kmeans(co, n_clusters, rng_seed=rng_seed, max_iter=max_iter)

    state_ids: list[int] = []
    for mags in per_turn_magnitudes:
        if mags.max() == 0:
            state_ids.append(int(assignments[0]))            # degenerate empty turn
            continue
        top_feat = int(np.argmax(mags))
        state_ids.append(int(assignments[top_feat]))

    id_to_label = list(range(n_clusters))
    label_to_id = {i: i for i in id_to_label}
    return (
        StateSpace(
            method="cluster",
            id_to_label=id_to_label,
            label_to_id=label_to_id,
            extra={"n_clusters": n_clusters, "feature_to_cluster": assignments.tolist()},
        ),
        state_ids,
    )


def _kmeans(
    X: np.ndarray, k: int, rng_seed: int = 0, max_iter: int = 50, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny numpy k-means. Not a substitute for scikit, but adequate here."""
    rng = np.random.default_rng(rng_seed)
    n = X.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()
    for _ in range(max_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)   # (n, k)
        assignments = np.argmin(d, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = assignments == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = X[rng.integers(0, n)]
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers, assignments
