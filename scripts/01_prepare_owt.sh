#!/usr/bin/env bash
# Tokenize OpenWebText into train.bin and val.bin (no GPU; CPU + disk only).
# Run inside the sparse-dictionary-learning clone.
#
# Output: $SDL_REPO/transformer/data/openwebtext/{train.bin, val.bin}
# train.bin is ~17GB; budget ~1h CPU and ~30GB disk.

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
cd "$SDL_REPO/transformer"

# The OpenWebText download from HuggingFace can hit transient
# ChunkedEncodingError / IncompleteRead on the shared cluster network.
# Retry with linear backoff; HF datasets skips already-completed shards on
# re-invocation, so retries make forward progress.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
attempts="${PREP_MAX_ATTEMPTS:-5}"
for i in $(seq 1 "$attempts"); do
    echo "[prep] attempt $i/$attempts: python data/openwebtext/prepare.py"
    if python data/openwebtext/prepare.py; then
        echo "[prep] succeeded on attempt $i"
        exit 0
    fi
    echo "[prep] attempt $i failed (likely transient network); backing off" >&2
    sleep $(( i * 30 ))
done
echo "[prep] all $attempts attempts failed" >&2
exit 1
