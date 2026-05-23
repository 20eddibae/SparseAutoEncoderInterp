#!/bin/bash -l
# End-to-end Discovery bootstrap.
#
# Subcommands:
#   install    one-time: unpack SDL, create conda env, install both packages
#              (must be run on a compute node via `srun --pty`)
#   submit     submit the 5-job pipeline (run on the login node)
#   all        run `install` inside an srun allocation, then `submit`
#              (run on the login node)
#
# Typical first-run usage on Discovery (from $HOME/stochastic-conversation-features):
#   bash bootstrap.sh all
#
# Subsequent submissions:
#   bash bootstrap.sh submit

set -euo pipefail

SCF_REPO="${SCF_REPO:-$(cd "$(dirname "$0")" && pwd)}"
SDL_DIR="${SDL_DIR:-$HOME/sparse-dictionary-learning}"
CONDA_ENV="${CONDA_ENV:-scf-env}"
ACCOUNT="${ACCOUNT:-rc}"

cmd="${1:-help}"

_install() {
  echo "[install] unpacking SDL tarball -> $SDL_DIR"
  if [[ ! -d "$SDL_DIR" ]]; then
    tar -xzf "$SCF_REPO/vendor/sparse-dictionary-learning.tar.gz" -C "$(dirname "$SDL_DIR")"
  else
    echo "[install] $SDL_DIR already exists; skipping unpack"
  fi

  echo "[install] loading python module + creating conda env $CONDA_ENV"
  module purge
  module load python
  if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -n "$CONDA_ENV" python=3.9 -y
  fi
  source activate "$CONDA_ENV"

  echo "[install] pip install SDL + SCF"
  pip install --quiet -r "$SDL_DIR/requirements.txt"
  pip install --quiet -e "$SCF_REPO"

  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
PY
  echo "[install] done"
}

_submit() {
  cd "$SCF_REPO"
  mkdir -p logs artifacts data/raw
  bash slurm/submit_all.sh
  echo "[submit] use `squeue --me` to monitor"
}

case "$cmd" in
  install) _install ;;
  submit)  _submit ;;
  all)
    echo "[all] running install inside an srun allocation"
    srun --account="$ACCOUNT" --partition=standard \
         --cpus-per-task=4 --mem=16G --time=01:00:00 \
         bash "$SCF_REPO/bootstrap.sh" install
    _submit
    ;;
  help|*)
    sed -n '2,15p' "$0"
    ;;
esac
