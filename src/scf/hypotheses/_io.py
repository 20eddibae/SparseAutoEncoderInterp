"""
Shared loader for the `.npz` produced by scripts/extract_features.py.

The npz layout:
  magnitudes:  (N_turns, F) float32, dense (use np.savez_compressed)
  support:     (N_turns, F) uint8  (binary)
  role:        (N_turns,)   uint8  (0 = human, 1 = ai)
  conv_id:     (N_turns,)   int32  (which conversation each turn belongs to)
  turn_idx:    (N_turns,)   int32  (position within conversation)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureBundle:
    magnitudes: np.ndarray
    support: np.ndarray
    role: np.ndarray
    conv_id: np.ndarray
    turn_idx: np.ndarray

    @classmethod
    def load(cls, path: str) -> "FeatureBundle":
        z = np.load(path)
        return cls(
            magnitudes=z["magnitudes"],
            support=z["support"].astype(bool),
            role=z["role"].astype(int),
            conv_id=z["conv_id"].astype(int),
            turn_idx=z["turn_idx"].astype(int),
        )

    def by_conversation(self) -> list[np.ndarray]:
        """Return list of magnitude arrays grouped by conv_id, in turn order."""
        out: dict[int, list] = {}
        order: dict[int, list] = {}
        for i in range(self.conv_id.size):
            cid = int(self.conv_id[i])
            out.setdefault(cid, []).append(self.magnitudes[i])
            order.setdefault(cid, []).append(int(self.turn_idx[i]))
        result = []
        for cid in sorted(out):
            idx = np.argsort(order[cid])
            stacked = np.stack(out[cid], axis=0)[idx]
            result.append(stacked)
        return result
