#!/usr/bin/env python3
"""Check full-size training memory on synthetic data; never write checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed

from models import MODEL_CHECKPOINT_CONFIGS, get_model
from pretrain_models import get_model as get_pretrain_model
from source.config import TrainingConfig, setup_seed
from source.core.loss_function import LossStrategy
from source.core.optimizer import setup_lldr_optimizer, setup_optimizer
from source.pretrain.loss_function import ReconstructionLoss
from source.pretrain.parse import PretrainConfig
from source.train import _EMAWeights
from source.utils.runtime import set_model_is_training, unwrap_model


def memory_report(stage: str, **extra: object) -> None:
    gib = 1024 ** 3
    print(json.dumps({
        "stage": stage,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3),
        **extra,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint-cnn", action="store_true",
                        help="Enable the existing UniME CNN activation checkpoints.")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.checkpoint_cnn and args.stage != "finetune":
        parser.error("--checkpoint-cnn applies only to finetuning")
    if not torch.cuda.is_available():
        raise RuntimeError("This full-size smoke check requires an allocated CUDA GPU.")

    accelerator = Accelerator(mixed_precision="fp16")
    if accelerator.num_processes != 1:
        raise RuntimeError("Run this check with one process: the required global batch is four.")
    setup_seed(40)
    set_seed(40, device_specific=True)
    device = accelerator.device
    torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({
        "stage": args.stage, "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "batch_size": 4, "crop_size": 96,
        "compile": args.compile, "checkpoint_cnn": args.checkpoint_cnn,
        "precision": "fp16", "synthetic_data": True,
        "weights": "random initialization; this does not verify pretrained transfer",
    }), flush=True)
    started = time.monotonic()

    if args.stage == "pretrain":
        config = PretrainConfig(base_lr=3e-4, weight_decay=1e-4)
        model = get_pretrain_model("UniEncoder")(
            in_channels=4, out_channels=4, crop_size=96,
            original_shape=tuple(config.original_shape), patch_mask_ratio=0.75,
            modality_mask_prob=0.5, num_mask_modalities=3,
        )
        criterion = ReconstructionLoss(regulization_rate=0.005)
    else:
        if args.checkpoint_cnn:
            MODEL_CHECKPOINT_CONFIGS["UniME"] = {
                "encoder": True, "regularizer": True, "decoder": True,
            }
        config = TrainingConfig(base_lr=3e-4, weight_decay=1e-4, layer_decay=0.75)
        # The production wrapper explicitly permits missing pretrained weights;
        # a unique, empty temporary directory guarantees this is a random smoke.
        with tempfile.TemporaryDirectory(prefix="unime-smoke-weights-") as empty_dir:
            model = get_model("UniME")(
                num_modals=4, num_classes=4, layer_decay=0.75,
                pretrained_path=str(Path(empty_dir) / "not-a-checkpoint.pth"),
            )
        criterion = LossStrategy(required_auxiliary=True)

    if args.compile:
        model = torch.compile(model, mode="max-autotune" if args.stage == "pretrain" else "default")
    optimizer = (setup_optimizer(config, model) if args.stage == "pretrain"
                 else setup_lldr_optimizer(config, model))
    # Match the real first-step warmup LR, including layer multipliers.
    for group in optimizer.param_groups:
        group["lr"] = 1e-5 * group.get("lr_scale", 1.0)
    model, optimizer = accelerator.prepare(model, optimizer)
    base_model = unwrap_model(model)
    ema = _EMAWeights(base_model, decay=0.999)
    criterion = criterion.to(device)
    model.train()
    set_model_is_training(model, True)

    images = torch.randn(4, 4, 96, 96, 96, device=device)
    labels = None
    if args.stage == "finetune":
        classes = torch.randint(0, 4, (4, 96, 96, 96), device=device)
        labels = F.one_hot(classes, num_classes=4).movedim(-1, 1).float().contiguous()
        del classes
    masks = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0],
                          [0, 1, 1, 0], [0, 0, 0, 1]], dtype=torch.bool, device=device)
    locations = None
    if args.stage == "pretrain":
        # Include the far boundary of every configured prior axis. The old
        # synthetic H/W origins assumed the larger local prior and no longer fit.
        maximum_starts = [int(size) - 96 for size in config.original_shape]
        if min(maximum_starts) < 0:
            raise ValueError("Synthetic training crop does not fit the configured prior.")
        origins = [torch.tensor([0, limit // 3, 2 * limit // 3, limit], device=device)
                   for limit in maximum_starts]
        locations = tuple((origin, origin + 96) for origin in origins)

    step_times: list[float] = []
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize(device)
        step_started = time.monotonic()
        # Preserve each production loop's ordering: pretraining clears old
        # gradients after the next forward, while finetuning clears beforehand.
        if args.stage == "finetune":
            optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            if args.stage == "pretrain":
                reconstruction, prior = model(images, location=locations)
                loss = criterion(reconstruction, prior.to(reconstruction.dtype),
                                 images.to(reconstruction.dtype))
            else:
                fusion, auxiliary, deep = model(images, masks)
                losses = criterion(fusion_output=fusion, auxiliary_outputs=auxiliary,
                                   deep_supervision_outputs=deep, target=labels)
                loss = sum(losses)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite {args.stage} loss at step {step}: {loss.item()}")
        if args.stage == "pretrain":
            optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"Nonfinite {args.stage} gradient norm at step {step}: {grad_norm.item()}")
        optimizer.step()
        if accelerator.optimizer_step_was_skipped:
            raise RuntimeError(f"AMP skipped {args.stage} optimizer step {step}.")
        ema.update(base_model)
        torch.cuda.synchronize(device)
        step_seconds = time.monotonic() - step_started
        step_times.append(step_seconds)
        memory_report(args.stage, step=step, loss=float(loss.item()),
                      grad_norm=float(grad_norm.item()),
                      step_seconds=round(step_seconds, 4), status="step_passed")

    steady_times = step_times[2:]
    memory_report(
        args.stage, status="passed", elapsed_seconds=round(time.monotonic() - started, 2),
        timing_warmup_steps=2, timing_sample_count=len(steady_times),
        steady_step_median_seconds=(round(statistics.median(steady_times), 4) if steady_times else None),
        steady_step_mean_seconds=(round(statistics.mean(steady_times), 4) if steady_times else None),
    )
    accelerator.end_training()


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        memory_report("failure", status="out_of_memory")
        raise
