#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Paper Sec. 4.1. Activate the unime environment before launching.
# Seed/EMA are retained implementation choices, not specified by the paper.
"${UNIME_PYTHON:-python}" pretrain.py \
    --gpu_ids "${UNIME_GPU_IDS:-0}" \
    --model_name UniEncoder \
    --dataset_name BRATS2023 \
    --data_path "${UNIME_DATA_PATH:-./data}" \
    --split_type Normal \
    --log_root "${UNIME_PRETRAIN_LOG_ROOT:-log_pretrain_paper}" \
    --seed "${UNIME_SEED:-40}" \
    --batch_size 4 \
    --num_workers 8 \
    --persistent_workers \
    --prefetch_factor 2 \
    --amp \
    --compile \
    --crop_size 96 \
    --original_shape 160 180 210 \
    --num_epochs 600 \
    --iter_per_epoch 250 \
    --mask_ratio 0.75 \
    --modality_mask_prob 0.5 \
    --num_mask_modalities 3 \
    --regulization_rate 0.005 \
    --weight_decay 1e-4 \
    --warmup_lr 1e-5 \
    --warmup_ratio 0.05 \
    --base_lr 3e-4 \
    --min_lr 1e-6 \
    --use_ema \
    --ema_validate_interval 1 \
    --ema_validate_start_epoch 500 \
    --wandb_mode "${UNIME_WANDB_MODE:-disabled}" \
    --wandb_project UniME \
    --wandb_run_name UniME-BRATS2023-Paper-Pretrain \
    "$@"
