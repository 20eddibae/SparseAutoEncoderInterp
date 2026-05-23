#!/usr/bin/env bash
# Tokenize OpenWebText into train.bin and val.bin (no GPU; CPU + disk only).
# Run inside the sparse-dictionary-learning clone.
#
# Output: $SDL_REPO/transformer/data/openwebtext/{train.bin, val.bin}
# train.bin is ~17GB; budget ~1h CPU and ~30GB disk.

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
cd "$SDL_REPO/transformer"
python data/openwebtext/prepare.py
