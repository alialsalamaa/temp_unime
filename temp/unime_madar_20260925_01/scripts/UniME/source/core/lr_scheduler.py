from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler

if TYPE_CHECKING:
    from source.config import TrainingConfig
    from source.pretrain.parse import PretrainConfig


def setup_scheduler(
    args: TrainingConfig | PretrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None,
) -> Scheduler:
    """
    Build the released code's step-level scheduler using the configured LR values:
        - Linear warmup: warmup_lr_init -> base_lr, default to 5% of total steps
        - Cosine clock spans all training steps, including the warmup interval

    Args:
        args (TrainingConfig): Training configuration
        optimizer (torch.optim.Optimizer): Optimizer
        steps_per_epoch (int): Actual steps per epoch, if None uses args.iter_per_epoch

    Returns:
        Scheduler: Learning rate scheduler
    """
    if steps_per_epoch is None:
        steps_per_epoch = args.iter_per_epoch

    num_steps = int(args.num_epochs * steps_per_epoch)
    if num_steps <= 0:
        raise ValueError("The learning-rate schedule requires a positive total number of steps.")
    warmup_ratio = float(args.warmup_ratio)
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be finite and in [0, 1).")
    warmup_steps = int(warmup_ratio * num_steps)

    # Retain the official implementation's clock: warmup is not prefixed to the
    # cosine period. Consequently, the first post-warmup value is slightly below
    # base_lr; do not silently substitute a warmup-then-full-cosine schedule.
    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_steps,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        warmup_prefix=False,
        cycle_limit=1,
        t_in_epochs=False
    )

    return lr_scheduler


def setup_lldr_scheduler(
    args: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None
) -> Scheduler:
    """
    Build the cosine scheduler for Uni-Encoder when LLDR is enabled.

    This wraps ``setup_scheduler`` to keep the existing behavior while
    providing a dedicated entry-point that can evolve independently if the
    LLDR pathway requires custom scheduling logic in the future.
    """
    return setup_scheduler(args, optimizer, steps_per_epoch)
