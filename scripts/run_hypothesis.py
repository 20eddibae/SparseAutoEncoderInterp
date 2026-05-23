#!/usr/bin/env python3
"""
Run one of the four hypothesis tests on an extracted features file.

Usage:
  python scripts/run_hypothesis.py --h 1 --features artifacts/features.npz \
      [--config configs/default.yaml] [--out artifacts/h1.json]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from scf.config import Config, default_config_path


def _default_to_jsonable(obj):
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default=default_config_path())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.h == 1:
        from scf.hypotheses.h1_markovianity import run
    elif args.h == 2:
        from scf.hypotheses.h2_human_vs_ai import run
    elif args.h == 3:
        from scf.hypotheses.h3_concentration import run
    else:
        from scf.hypotheses.h4_stationarity import run

    result = run(args.features, cfg)
    serialised = json.dumps(result, indent=2, default=_default_to_jsonable)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(serialised)
        print(f"wrote {args.out}")
    else:
        print(serialised)


if __name__ == "__main__":
    main()
