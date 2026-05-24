#!/bin/bash -l
#SBATCH --job-name=scf-separability
#SBATCH --partition=standard
#SBATCH --account=lsonglab
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

# CPU-only sanity + role-separability check on artifacts/features.npz.
# Override: FEATURES=artifacts/features.npz PLOT=artifacts/separability.png sbatch slurm/06_check_separability.sbatch

source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"

FEATURES="${FEATURES:-$SCF_REPO/artifacts/features.npz}"
PLOT="${PLOT:-$SCF_REPO/artifacts/separability.png}"
echo "[job] FEATURES=$FEATURES  PLOT=$PLOT"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python "$SCF_REPO/scripts/check_separability.py" --features "$FEATURES" --plot "$PLOT"
