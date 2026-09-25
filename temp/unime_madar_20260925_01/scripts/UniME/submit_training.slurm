#!/usr/bin/env bash
#SBATCH --job-name=unime-paper-s40
#SBATCH --partition=dcv-p4d-ond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --open-mode=append

set -euo pipefail
unime_run_dir="${1:?Pass the absolute run directory containing source/ and split.json}"
cd -- "$unime_run_dir/source"
exec 9>"$unime_run_dir/.run.lock"
flock -n 9 || { echo "This run is already active" >&2; exit 3; }

export UNIME_PYTHON=/home/u23102893/anaconda03/envs/unime/bin/python
export UNIME_DATA_PATH="$unime_run_dir/data"
export UNIME_PRETRAIN_LOG_ROOT="$unime_run_dir/pretrain"
export UNIME_FINETUNE_LOG_ROOT="$unime_run_dir/finetune"
export UNIME_SEED=40
export UNIME_WANDB_MODE=disabled
# Preserve Slurm's assigned GPU visibility in both Python entrypoints.
export UNIME_GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CC=/home/u23102893/anaconda03/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/u23102893/anaconda03/bin/x86_64-conda-linux-gnu-g++
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="/tmp/unime-${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="/tmp/unime-${SLURM_JOB_ID}/triton"
export XDG_CACHE_HOME="/tmp/unime-${SLURM_JOB_ID}/cache"
export PYTHONPYCACHEPREFIX="/tmp/unime-${SLURM_JOB_ID}/pycache"
mkdir -p "$unime_run_dir/logs" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

unime_stage=initializing
trap 'unime_code=$?; printf "stage=%s exit_code=%s time=%s\n" "$unime_stage" "$unime_code" "$(date -u +%FT%TZ)" > "$unime_run_dir/status.txt"' EXIT
printf 'Job %s on %s; partition=%s\n' "$SLURM_JOB_ID" "$HOSTNAME" "$SLURM_JOB_PARTITION"
nvidia-smi
"$CXX" --version

unime_stage=gpu_check
printf '%s\n' "$unime_stage" > "$unime_run_dir/status.txt"
"$UNIME_PYTHON" scripts/smoke_training_gpu.py --stage pretrain --compile 2>&1 | tee "$unime_run_dir/logs/smoke_pretrain.log"
"$UNIME_PYTHON" scripts/smoke_training_gpu.py --stage finetune --compile 2>&1 | tee "$unime_run_dir/logs/smoke_finetune.log"

unime_stage=data_preparation
printf '%s\n' "$unime_stage" > "$unime_run_dir/status.txt"
"$UNIME_PYTHON" scripts/brats23_process.py \
    --raw-dir /home/u23102893/MICAD/brats2023_simple_mamba_study/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
    --out-dir "$UNIME_DATA_PATH/BRATS2023" \
    --split-json "$unime_run_dir/split.json" --write-mapping \
    2>&1 | tee "$unime_run_dir/logs/preparation.log"
"$UNIME_PYTHON" scripts/verify_study_split.py \
    --data-root "$UNIME_DATA_PATH/BRATS2023" --split-json "$unime_run_dir/split.json" \
    | tee "$unime_run_dir/split_verification.json"

unime_stage=pretraining
printf '%s\n' "$unime_stage" > "$unime_run_dir/status.txt"
bash scripts/pretrain.sh 2>&1 | tee "$unime_run_dir/logs/pretraining.log"

unime_stage=finetuning_and_evaluation
printf '%s\n' "$unime_stage" > "$unime_run_dir/status.txt"
bash scripts/finetune.sh 2>&1 | tee "$unime_run_dir/logs/finetuning.log"
unime_stage=completed
