#!/bin/bash -l
# bootstrap.sh — one-command setup + job submission for the
# stochastic-conversation-features Discovery pipeline.
#
# Subcommands:
#   all       unpack SDL tarball, build the conda env on a compute node,
#             then submit all 5 chained Slurm jobs (--dependency=afterok).
#   install   build/refresh the conda env only. Auto-grabs a compute node via
#             srun if not already inside an allocation (never installs heavy
#             packages on the shared login node).
#   unpack    unpack the vendored SDL tarball only.
#   submit    submit the Slurm jobs only (passes flags through to
#             slurm/submit_all.sh, e.g. --skip-train).
#   help      show this message.
#
# Override-friendly env vars:
#   SDL_REPO       (default: $HOME/sparse-dictionary-learning)
#   CONDA_ENV      (default: scf-env)
#   SRUN_ACCOUNT   (default: lsonglab)
#   SRUN_PARTITION (default: standard)

set -euo pipefail

SCF_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCF_REPO"

: "${SDL_REPO:=$HOME/sparse-dictionary-learning}"
: "${CONDA_ENV:=scf-env}"
: "${SRUN_ACCOUNT:=lsonglab}"
: "${SRUN_PARTITION:=standard}"
TARBALL="$SCF_REPO/vendor/sparse-dictionary-learning.tar.gz"

export SDL_REPO CONDA_ENV SCF_REPO

log() { echo "[bootstrap] $*"; }

cmd_unpack() {
  if [[ -e "$SDL_REPO/requirements.txt" ]]; then
    log "SDL already unpacked at $SDL_REPO — skipping"
    return 0
  fi
  [[ -f "$TARBALL" ]] || { log "ERROR: tarball not found: $TARBALL"; exit 1; }
  log "unpacking $TARBALL -> $(dirname "$SDL_REPO")/"
  tar -xzf "$TARBALL" -C "$(dirname "$SDL_REPO")"
  [[ -e "$SDL_REPO/requirements.txt" ]] || {
    log "ERROR: expected $SDL_REPO/requirements.txt after unpack"; exit 1; }
  log "unpacked OK"
}

cmd_install() {
  # If we are not inside a Slurm allocation, re-launch this same subcommand on a
  # compute node. The login node is shared/throttled — never pip-install there.
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    log "not in a Slurm allocation — requesting a compute node via srun"
    exec srun --account="$SRUN_ACCOUNT" --partition="$SRUN_PARTITION" \
         --cpus-per-task=4 --mem=16G --time=01:00:00 \
         bash "$SCF_REPO/bootstrap.sh" install
  fi

  log "on compute node $(hostname); building env $CONDA_ENV"
  module purge
  module load python

  if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    log "conda env '$CONDA_ENV' already exists — reusing"
  else
    log "creating conda env '$CONDA_ENV' (python 3.9)"
    conda create -n "$CONDA_ENV" python=3.9 -y
  fi

  source activate "$CONDA_ENV"

  log "installing SDL requirements ($SDL_REPO/requirements.txt)"
  pip install -r "$SDL_REPO/requirements.txt"

  log "installing analysis layer (editable: $SCF_REPO)"
  pip install -e "$SCF_REPO"

  log "verifying torch + CUDA"
  python -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"
  log "install complete"
}

cmd_submit() {
  log "submitting Slurm jobs via slurm/submit_all.sh $*"
  bash "$SCF_REPO/slurm/submit_all.sh" "$@"
}

cmd_all() {
  cmd_unpack
  cmd_install
  cmd_submit
  log "all done — track with: squeue --me"
}

cmd_help() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-help}" in
  all)     cmd_all ;;
  install) cmd_install ;;
  unpack)  cmd_unpack ;;
  submit)  shift; cmd_submit "$@" ;;
  help|-h|--help) cmd_help ;;
  *) log "unknown subcommand: $1"; echo; cmd_help; exit 1 ;;
esac
