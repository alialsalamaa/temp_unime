#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

unime_seed="${UNIME_SEED:-40}"
unime_checkpoint="${UNIME_PRETRAIN_CHECKPOINT:-${UNIME_PRETRAIN_LOG_ROOT:-log_pretrain_paper}/BRATS2023-${unime_seed}-Normal/UniEncoder/ema_best_checkpoint.pth}"
if [[ ! -f "$unime_checkpoint" ]]; then
    echo "Missing pretrained UniEncoder checkpoint: $unime_checkpoint" >&2
    exit 1
fi

"${UNIME_PYTHON:-python}" main.py \
    --gpu_ids "${UNIME_GPU_IDS:-0}" \
    --split_type Normal \
    --dataset_name BRATS2023 \
    --data_path "${UNIME_DATA_PATH:-./data}" \
    --log_root "${UNIME_FINETUNE_LOG_ROOT:-log_paper}" \
    --model_name UniME \
    --batch_size 4 \
    --amp \
    --use_ema \
    --compile \
    --warmup_lr 1e-5 \
    --warmup_ratio 0.05 \
    --base_lr 3e-4 \
    --min_lr 1e-6 \
    --weight_decay 1e-4 \
    --layer_decay 0.75 \
    --iter_per_epoch 250 \
    --num_epochs 600 \
    --crop_size 96 \
    --seed "$unime_seed" \
    --uni_encoder_name UniEncoder \
    --uni_encoder_checkpoint "$unime_checkpoint" \
    --wandb_mode "${UNIME_WANDB_MODE:-disabled}" \
    --wandb_project UniME \
    --wandb_run_name UniME-BRATS2023-Paper-Finetune \
    "$@"
