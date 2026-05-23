#!/usr/bin/env python3
"""
Walk a corpus, run each turn through the transformer + SAE, save per-turn
feature vectors as a single compressed .npz.

Usage:
  python scripts/extract_features.py \
      --config configs/default.yaml \
      --out artifacts/features.npz
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List

import numpy as np
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from scf.config import Config, default_config_path
from scf.corpus import load_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config_path())
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    from scf.feature_extractor import FeatureExtractor   # imports torch + sdl repo
    extractor = FeatureExtractor(cfg)

    mags: List[np.ndarray] = []
    supports: List[np.ndarray] = []
    roles: List[int] = []
    conv_ids: List[int] = []
    turn_indices: List[int] = []

    for cid, conv in enumerate(tqdm(load_corpus(cfg), desc="conversations")):
        for ti, turn in enumerate(conv.turns):
            tf = extractor.encode_turn(turn.text)
            mags.append(tf.magnitudes)
            supports.append(tf.support.astype(np.uint8))
            roles.append(0 if turn.role == "human" else 1)
            conv_ids.append(cid)
            turn_indices.append(ti)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        magnitudes=np.stack(mags),
        support=np.stack(supports),
        role=np.array(roles, dtype=np.uint8),
        conv_id=np.array(conv_ids, dtype=np.int32),
        turn_idx=np.array(turn_indices, dtype=np.int32),
    )
    print(f"wrote {args.out}: {len(mags)} turns")


if __name__ == "__main__":
    main()
