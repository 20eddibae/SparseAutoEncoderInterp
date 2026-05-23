#!/bin/bash -l
#SBATCH --job-name=scf-prep-owt
#SBATCH --partition=standard
#SBATCH --account=rc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --mail-type=END,FAIL

# Tokenize OpenWebText on CPU. ~1h wallclock, ~30GB disk.
# Override defaults: SDL_REPO=...  sbatch slurm/01_prepare_owt.sbatch

source "$(dirname "$0")/00_env.sh"
bash "$SCF_REPO/scripts/01_prepare_owt.sh"
