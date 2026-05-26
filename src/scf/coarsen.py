"""
State-space definition for the conversation Markov chain (Experiment 4).

The raw per-turn object is a magnitude vector in R^F (F = 4096). A Markov chain
on the full vector is impossible, so we need a *state* that is (a) a deterministic
function of the activation vector and (b) a course-native probability object.

We use the **argmax (dominant) feature**: the single most-active feature of the
turn. This is the maximum of F random variables — an order statistic (course
items 18-19, Apr 16) — not the output of any clustering/optimization step. There
is no KMeans and no fragile K hyperparameter here.

Two practical refinements, both justified inside the course:

  1. Drop always-on features. A feature that fires on every turn has firing
     probability 1, so it carries zero information (its indicator is a.s.
     constant) and would otherwise win every argmax. We detect these by their
     empirical firing rate == 1 and exclude them from the argmax.

  2. Pool the rare tail. The argmax realises ~100 distinct features, but a long
     tail of them are the dominant feature only a handful of times — too few to
     estimate an outgoing transition row, and they create absorbing traps that
     break ergodicity. We keep the M-1 most frequent dominant features as named
     states and pool the rest into a single 'other' state. M is a *reporting
     granularity* (like a histogram bin count), not a clustering hyperparameter;
     the headline comparisons are stable across M (see RESULTS.md).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class StateSpace:
    method: str
    id_to_label: list           # state id -> serialisable label (feature index, or "other")
    label_to_id: dict
    extra: dict                 # method-specific bookkeeping

    @property
    def n_states(self) -> int:
        return len(self.id_to_label)


# ---------------------------------------------------------------- helpers


def always_on_features(support: np.ndarray) -> np.ndarray:
    """Indices of features that fire on every turn (firing rate == 1).

    Such a feature's indicator is almost-surely constant -> zero information,
    so it is excluded from the argmax (otherwise it dominates every turn).
    """
    rate = np.asarray(support, dtype=np.float64).mean(axis=0)
    return np.where(rate >= 1.0)[0]


def dominant_feature(magnitudes: np.ndarray, exclude: Sequence[int] = ()) -> np.ndarray:
    """argmax feature per turn (an order statistic), excluding `exclude` features.

    magnitudes: (N, F).  Returns (N,) feature indices.
    """
    M = np.asarray(magnitudes, dtype=np.float32).copy()
    if len(exclude):
        M[:, np.asarray(exclude, dtype=int)] = -np.inf
    return M.argmax(axis=1)


# ---------------------------------------------------------------- argmax space


def build_argmax_space(
    magnitudes: np.ndarray,
    support: np.ndarray | None = None,
    n_states: int = 32,
    exclude: Sequence[int] | None = None,
) -> Tuple[StateSpace, np.ndarray]:
    """Define states as the dominant (argmax) feature, top-(n_states-1) named +
    one pooled 'other'.

    Parameters
    ----------
    magnitudes : (N, F) per-turn SAE magnitudes.
    support    : (N, F) 0/1; used to auto-detect always-on features when
                 `exclude` is None. If both are None, nothing is excluded.
    n_states   : reporting granularity M. n_states-1 named feature-states + 'other'.
    exclude    : feature indices to drop from the argmax. Defaults to the
                 always-on features detected from `support`.

    Returns (StateSpace, state_ids[(N,)]).
    """
    if exclude is None:
        exclude = always_on_features(support) if support is not None else np.array([], dtype=int)
    exclude = np.asarray(exclude, dtype=int)

    feat = dominant_feature(magnitudes, exclude)
    vals, counts = np.unique(feat, return_counts=True)
    keep = vals[np.argsort(-counts)[: max(n_states - 1, 1)]]      # M-1 named states
    remap = {int(f): i for i, f in enumerate(keep)}
    other_id = len(keep)                                          # pooled tail

    id_to_label: list = [int(f) for f in keep] + ["other"]
    label_to_id = {lbl: i for i, lbl in enumerate(id_to_label)}
    state_ids = np.array([remap.get(int(f), other_id) for f in feat], dtype=int)

    coverage = float(counts[np.argsort(-counts)[: max(n_states - 1, 1)]].sum() / counts.sum())
    return (
        StateSpace(
            method="argmax",
            id_to_label=id_to_label,
            label_to_id=label_to_id,
            extra={
                "n_states": len(id_to_label),
                "excluded_features": exclude.tolist(),
                "named_features": [int(f) for f in keep],
                "named_coverage": coverage,
                "n_realised_argmax": int(vals.size),
            },
        ),
        state_ids,
    )
