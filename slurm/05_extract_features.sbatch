#!/bin/bash -l
#SBATCH --job-name=scf-extract
#SBATCH --partition=gpuq
#SBATCH --account=lsonglab
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

# Walk the conversation corpus, run each turn through transformer+SAE, save features.npz.
# Small job; GPU only makes a 10x difference but corpus is small so could also run on CPU.
# Override: SCF_CONFIG=configs/discovery.yaml OUT=artifacts/features_sharegpt.npz sbatch slurm/05_extract_features.sbatch

source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"

SCF_CONFIG="${SCF_CONFIG:-$SCF_REPO/configs/discovery.yaml}"
OUT="${OUT:-$SCF_REPO/artifacts/features.npz}"
echo "[job] SCF_CONFIG=$SCF_CONFIG  OUT=$OUT"

python "$SCF_REPO/scripts/extract_features.py" --config "$SCF_CONFIG" --out "$OUT"
