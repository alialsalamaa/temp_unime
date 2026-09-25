#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# One source of truth for the original paper settings and checkpoint paths.
# A failed pretraining stage must not fall through to stale pretrained weights.
bash scripts/pretrain.sh
bash scripts/finetune.sh
