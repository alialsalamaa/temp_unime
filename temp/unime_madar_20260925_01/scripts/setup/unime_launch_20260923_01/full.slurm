#!/usr/bin/env bash
# Only submitted after all smoke gates pass; afterok releases the smoke GPU first.
#SBATCH --job-name=unime-full
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=14-00:00:00
#SBATCH --output=logs/full-%j.out
#SBATCH --error=logs/full-%j.err
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd -- /adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/setup/unime_launch_20260923_01
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID SLURM_EXPORT_ENV=ALL WANDB_MODE=disabled
export TORCHINDUCTOR_COMPILE_THREADS=4
exec /adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/.venv-unime/bin/python -B orchestrate.py full
