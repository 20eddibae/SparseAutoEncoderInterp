#!/bin/bash -l
# Sourced (not executed) by every Discovery sbatch script.
#
# Responsibilities:
#   - load the system python module so `conda` is on PATH
#   - activate the project's conda env
#   - set SCF_REPO / SDL_REPO env vars (override-friendly)
#   - cd into the scf repo so scripts/* paths resolve

set -euo pipefail

# Default install locations on Discovery (override in the sbatch script if you put them elsewhere).
: "${SCF_REPO:=${SLURM_SUBMIT_DIR:-$HOME/SparseAutoEncoderInterp}}"
: "${SDL_REPO:=/dartfs/rc/lab/S/SongL/EddieBae/sparse-dictionary-learning}"
: "${CONDA_ENV:=scf-env}"

export SCF_REPO SDL_REPO CONDA_ENV
export SCF_SDL_REPO="$SDL_REPO"          # picked up by src/scf/config.py

# Compute nodes have no TTY and no W&B API key, so online wandb.init() fails.
# Log offline (saved under wandb/); run `wandb sync` from a login node later.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Discovery uses environment modules; load python to get conda on PATH.
module purge
module load python

# Activate the user's conda env (created once, see slurm/SETUP.md).
source activate "$CONDA_ENV"

cd "$SCF_REPO"
mkdir -p logs artifacts

echo "[env] node=$(hostname)  date=$(date -u)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<none>}"
echo "[env] SCF_REPO=$SCF_REPO"
echo "[env] SDL_REPO=$SDL_REPO"
echo "[env] CONDA_ENV=$CONDA_ENV  python=$(which python)"
