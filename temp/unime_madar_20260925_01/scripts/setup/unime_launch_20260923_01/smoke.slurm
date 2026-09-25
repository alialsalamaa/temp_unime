#!/usr/bin/env bash
# One EXTRA allocation authorized while the existing V11 t1 job stays untouched.
#SBATCH --job-name=unime-smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
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
exec /adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/.venv-unime/bin/python -B orchestrate.py smoke
