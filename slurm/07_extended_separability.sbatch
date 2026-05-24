#!/bin/bash -l
#SBATCH --job-name=scf-sep-ext
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

# CPU-only extended diagnosis of role separability on artifacts/features.npz.
source "${SLURM_SUBMIT_DIR:-/dartfs-hpc/rc/home/w/f0079dw/SparseAutoEncoderInterp}/slurm/00_env.sh"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
python "$SCF_REPO/scripts/extended_separability.py" --features "$SCF_REPO/artifacts/features.npz"
