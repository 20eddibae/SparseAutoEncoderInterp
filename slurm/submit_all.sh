#!/bin/bash -l
# Chain the 5 Discovery jobs with --dependency=afterok:
#
#   01_prepare_owt        (CPU, ~1h)
#   -> 02_train_transformer  (GPU A100, ~6h)
#   -> 03_generate_mlp_data  (GPU A100, ~2h, 256G RAM)
#   -> 04_train_sae          (GPU A100, ~4h)
#   -> 05_extract_features   (GPU A100, ~30min)
#
# Usage:
#   cd $SCF_REPO
#   bash slurm/submit_all.sh
#
# Each step waits for the previous to finish successfully. If you have a
# checkpoint already (e.g. owt is tokenized), pass --skip-owt or --skip-train
# to drop earlier steps and re-anchor the dependency chain.

set -euo pipefail

SCF_REPO="${SCF_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$SCF_REPO"
mkdir -p logs

SKIP_OWT=0
SKIP_TRAIN=0
SKIP_MLP=0
SKIP_SAE=0
for a in "$@"; do
  case "$a" in
    --skip-owt) SKIP_OWT=1 ;;
    --skip-train) SKIP_OWT=1; SKIP_TRAIN=1 ;;
    --skip-mlp)   SKIP_OWT=1; SKIP_TRAIN=1; SKIP_MLP=1 ;;
    --skip-sae)   SKIP_OWT=1; SKIP_TRAIN=1; SKIP_MLP=1; SKIP_SAE=1 ;;
    *) echo "unknown flag: $a"; exit 1 ;;
  esac
done

dep=""
submit() {
  local script="$1"
  local args="${dep:+--dependency=afterok:$dep}"
  local jid
  jid=$(sbatch --parsable $args "$script")
  echo "submitted $script -> $jid${dep:+ (depends on $dep)}"
  dep="$jid"
}

[[ $SKIP_OWT  -eq 0 ]] && submit slurm/01_prepare_owt.sbatch
[[ $SKIP_TRAIN -eq 0 ]] && submit slurm/02_train_transformer.sbatch
[[ $SKIP_MLP   -eq 0 ]] && submit slurm/03_generate_mlp_data.sbatch
[[ $SKIP_SAE   -eq 0 ]] && submit slurm/04_train_sae.sbatch
submit slurm/05_extract_features.sbatch
