#!/usr/bin/env python3
"""
Run one of the CPU-only experiments on an extracted features file.

  --exp 1   Is L0 Poisson?              (tail-sum, Poisson-as-limit, Markov/Chebyshev)
  --exp 2   Does the sample mean concentrate?  (WLLN + CLT)
  --exp 3   Are the roles different distributions?  (entropy + delta method + KL/Jensen)

Experiment 4 (the conversation-mode Markov chain) and the generative MCMC live in
scripts/mcmc_conversations.py, because they emit large array artifacts.

Usage:
  python scripts/run_hypothesis.py --exp 1 --features artifacts/features_stripped.npz \
      [--config configs/discovery.yaml] [--out results/data/exp1_stripped.json]
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
    ap.add_argument("--exp", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default=default_config_path())
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.exp == 1:
        from scf.hypotheses.exp1_poisson import run
    elif args.exp == 2:
        from scf.hypotheses.exp2_clt import run
    else:
        from scf.hypotheses.exp3_roles import run

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
