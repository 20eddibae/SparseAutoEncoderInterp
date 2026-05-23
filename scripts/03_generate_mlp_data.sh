#!/usr/bin/env bash
# Collect MLP activations from the trained transformer to feed the SAE.
#
# Lightweight regime: 500k contexts * 200 tokens = 1e8 activations (~50GB f16).
# Original paper used 4B (20M contexts). Statistics tests are robust to this.
#
# Output: $SDL_REPO/autoencoder/data/openwebtext/512/<partition_*.pt>
# RAM: ~80GB peak with default 20 partitions; reduce num_contexts if tight.

set -euo pipefail

SDL_REPO="${SDL_REPO:-/Users/eddiebae/CS/sparse-dictionary-learning}"
NUM_CONTEXTS="${NUM_CONTEXTS:-500000}"
NUM_PARTITIONS="${NUM_PARTITIONS:-20}"

cd "$SDL_REPO/autoencoder"
python -u prepare.py \
    --num_contexts="$NUM_CONTEXTS" \
    --num_sampled_tokens=200 \
    --num_partitions="$NUM_PARTITIONS" \
    --dataset=openwebtext \
    --gpt_ckpt_dir=out \
    --device=cuda
