#!/usr/bin/env python3
"""
Download a subset of allenai/WildChat-1M into a local ShareGPT-format JSON so
the existing loader (src/scf/corpus/sharegpt.py) can consume it unchanged.

Why: WildChat is a clean-text human/AI corpus (real user prompts + GPT-3.5/4
replies) with NO HTML wrapping, unlike ShareGPT52K. Running the role-separability
probe on it cross-checks that the ~0.92 stripped-ShareGPT number is genuine
human/AI language and not a ShareGPT-specific tell.

Run on a node WITH internet (Discovery login/frontend, not a gpuq compute node):
  python scripts/fetch_wildchat.py --n 10000 --lang English \
      --out data/raw/wildchat.json

Output format: [{"conversations": [{"from": "human"|"gpt", "value": str}, ...]}]
matching ShareGPT, so configs point `sharegpt_path` at it with `name: sharegpt`.
"""
from __future__ import annotations
import argparse
import json
import os

_ROLE = {"user": "human", "assistant": "gpt"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="conversations to keep")
    ap.add_argument("--lang", default="English", help="language filter; '' = no filter")
    ap.add_argument("--dataset", default="allenai/WildChat-1M")
    ap.add_argument("--out", default="data/raw/wildchat.json")
    ap.add_argument("--min-turns", type=int, default=2)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True)

    out: list[dict] = []
    scanned = 0
    for row in ds:
        scanned += 1
        if args.lang and row.get("language") != args.lang:
            continue
        conv_in = row.get("conversation") or []
        turns = []
        for t in conv_in:
            role = _ROLE.get(t.get("role"))
            val = t.get("content") or ""
            if role is None or not val.strip():
                continue
            turns.append({"from": role, "value": val})
        if len(turns) < args.min_turns:
            continue
        out.append({"conversations": turns})
        if len(out) >= args.n:
            break
        if len(out) % 1000 == 0:
            print(f"[fetch] kept={len(out)} scanned={scanned}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    n_turns = sum(len(c["conversations"]) for c in out)
    print(f"[fetch] wrote {args.out}: {len(out)} convs, {n_turns} turns "
          f"(scanned {scanned} rows, lang={args.lang or 'any'})")


if __name__ == "__main__":
    main()
