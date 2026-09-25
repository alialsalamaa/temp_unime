#!/usr/bin/env python3
"""Fail-closed, full-UniME native-test/offline GPU inference parity preflight.

Uses seeded random weights and exactly the first frozen selection80 case, never
test-set payloads. This checks inference plumbing, not trained-model accuracy or
pretrained transfer. No training, checkpoint selection, or checkpoint writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def choose_case(protocol):
    """A fixed non-test case; no user-tunable case search or best-case choice."""
    ids = protocol["case_ids"]
    selected = ids["selection80"]
    if len(selected) != 80 or not selected:
        raise ValueError("Exactly the frozen selection80 population is required")
    cid = selected[0]
    if cid not in ids["val"] or cid in ids["train"] or cid in ids["test"]:
        raise ValueError("Parity case must belong exclusively to validation")
    pid = protocol["mapped_ids"]["selection80"][0]
    if protocol["processed_to_case"].get(pid) != cid:
        raise ValueError("Parity case mapping mismatch")
    return cid, pid


def require_scores(scores, expected):
    if set(scores) != set(expected) or any(
        not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1
        for value in scores.values()
    ):
        raise ValueError("Invalid finite Dice score fields/range")


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(json.dumps([name, tensor_digest(tensor)], separators=(",", ":")).encode())
    return digest.hexdigest()


def progress(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New exclusive JSON success receipt; its parent must already exist.")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("--output must be a new non-symlink file in an existing directory")
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("An authorized existing Slurm GPU allocation is required")
    if os.environ.get("SLURM_NTASKS", "1") != "1" or os.environ.get("WORLD_SIZE", "1") != "1":
        raise RuntimeError("Exactly one task/process is required")

    import numpy as np
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from models import get_model
    from source.benchmark_capture import source_fingerprints
    from source.benchmark_metrics import (
        avmur_region_dice, canonical_mask, native_region_metrics,
        raw_geometry, restore_full_grid,
    )
    from source.benchmark_protocol import (
        CONFIGURATIONS, REGIONS, read_json, require, sha256_file,
        validate_protocol, write_json_exclusive,
    )
    from source.benchmark_runner import build_model, case_file_identity, runtime_provenance
    from source.config import setup_seed
    from source.core.evaluation import evaluate_single_sample
    from source.core.inference import sliding_window_inference
    from source.dataset.dataset import BratsTestDataset, BratsValidationDataset
    from source.utils.runtime import set_model_is_training

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Exactly one allocated CUDA device must be visible")
    protocol_path = args.protocol.resolve(strict=True)
    protocol_sha = sha256_file(protocol_path)
    protocol = validate_protocol(read_json(protocol_path))
    require(protocol["array_audit"].get("mode") == "all_headers_and_exact_payload_sizes"
            and protocol["array_audit"].get("arrays_checked") == 2502,
            "Fresh full-array protocol is required, not a manifest-only protocol")
    cid, pid = choose_case(protocol)
    sources = source_fingerprints(REPO)
    identity = case_file_identity(protocol, cid, pid)
    payload_sha = {path: sha256_file(path) for path in identity}
    started = time.monotonic()
    accelerator = Accelerator(mixed_precision="fp16")
    require(accelerator.num_processes == 1, "One Accelerate process is required")
    require(accelerator.mixed_precision == "fp16", "Native test requires fp16 preparation")
    setup_seed(40)
    set_seed(40, device_specific=True)
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))

    # Both dataset classes read the validation manifest; only this selected
    # index is materialized. The native test transform is tested without ever
    # opening a test volume/label or a test data loader.
    datasets = [cls(protocol["data_root"], protocol["input_paths"]["val_manifest"], num_classes=4)
                for cls in (BratsTestDataset, BratsValidationDataset)]
    try:
        native_sample = datasets[0][datasets[0].names.index(pid)]
        offline_sample = datasets[1][datasets[1].names.index(pid)]
        require(native_sample[3] == offline_sample[3] == pid, "Wrong dataset case")
        require(torch.equal(native_sample[0], offline_sample[0])
                and torch.equal(native_sample[1], offline_sample[1]),
                "Native test and offline validation input/target tensors differ")
        images = native_sample[0].unsqueeze(0).clone().to("cuda", dtype=torch.float32)
        target = native_sample[1].argmax(dim=0).unsqueeze(0).clone().to("cuda")
    finally:
        for dataset in datasets:
            dataset.close()
    del native_sample, offline_sample, datasets, dataset
    require(bool(torch.isfinite(images).all()), "Nonfinite real preprocessed input")
    require(tuple(images.shape[2:]) == tuple(target.shape[1:]) and min(images.shape[2:]) >= 128,
            "Unexpected padded real-case shape")
    data_root = Path(protocol["data_root"])
    geometry = raw_geometry(Path(protocol["raw_data_root"]) / cid,
                            data_root / "seg" / f"{pid}_seg.npy",
                            processed_vol_path=data_root / "vol" / f"{pid}_vol.npy")
    require(geometry["raw_headers_verified"] and geometry["processed_volume_verified"],
            "Raw/processed geometry verification did not complete")
    full_target = torch.from_numpy(np.asarray(geometry["target"])).to("cuda")
    input_digest = tensor_digest(images)

    class FiniteFP32Forward(torch.nn.Module):
        """Observe native/offline forward boundaries without changing tensors."""
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.calls = 0

        def forward(self, inputs, masks):
            result = self.module(inputs, masks)
            require(isinstance(result, torch.Tensor)
                    and result.dtype == torch.float32
                    and tuple(result.shape) == (inputs.shape[0], 4, *inputs.shape[2:]),
                    "SWI must receive correctly shaped FP32 logits")
            require(bool(torch.isfinite(result).all()), "Nonfinite model logits")
            self.calls += 1
            return result

    progress(stage="inference_parity", state="starting", case_id=cid, processed_id=pid,
             partition="selection80", configuration_count=15, image_shape=list(images.shape),
             weights="seed40_random_initialization", test_payloads_loaded=False)
    # The production constructor is retained. The deliberately nonexistent
    # temporary checkpoint triggers its documented random-initialization path.
    with tempfile.TemporaryDirectory(prefix="unime-parity-") as scratch:
        native_base = get_model("UniME")(
            num_modals=4, num_classes=4, layer_decay=0.75,
            pretrained_path=str(Path(scratch) / "no-pretrained-weights.pth"))
    state = {name: tensor.detach().cpu().clone() for name, tensor in native_base.state_dict().items()}
    require(all(bool(torch.isfinite(value).all()) for value in state.values()), "Nonfinite random state")
    state_sha = state_digest(state)
    parameter_count = sum(value.numel() for value in native_base.parameters())
    # Match main.py + train_brats: compile first, then Accelerator.prepare.
    native = accelerator.prepare(torch.compile(native_base, mode="default"))
    native.eval()
    set_model_is_training(native, False)
    native_guard = FiniteFP32Forward(native)
    native_predictions = {}
    native_scores = {}
    native_call_counts = {}
    with torch.no_grad():  # Native source/test.py uses no_grad, not inference_mode.
        for index, (name, mask_value) in enumerate(CONFIGURATIONS, 1):
            mask = torch.tensor([canonical_mask(mask_value)], dtype=torch.bool, device="cuda")
            before = native_guard.calls
            prediction = sliding_window_inference(native_guard, images, 4, 96, overlap=0.5, masks=mask)
            _, scores = evaluate_single_sample(prediction, target)
            values = dict(zip((*REGIONS, "ET_postpro"), map(float, scores[0])))
            require_scores(values, (*REGIONS, "ET_postpro"))
            native_predictions[mask_value] = prediction.to("cpu", dtype=torch.uint8)
            native_scores[mask_value] = values
            native_call_counts[mask_value] = native_guard.calls - before
            progress(stage="native_reference", completed=index, total=15,
                     configuration=name, mask_value=mask_value, state="passed")
    require(state_digest(native_base.state_dict()) == state_sha, "Native inference changed model state")
    del native_guard, native, native_base, prediction
    accelerator.free_memory()
    gc.collect()
    torch.cuda.empty_cache()

    # Use the actual offline production constructor/forward wrapper. The path
    # is only resolved in scratch mode; it is not interpreted as a checkpoint.
    offline_base, offline = build_model({"compile": True, "layer_decay": 0.75}, protocol_path)
    offline_base.load_state_dict(state, strict=True)
    require(state_digest(offline_base.state_dict()) == state_sha, "Strict transferred state mismatch")
    del state
    offline_guard = FiniteFP32Forward(offline)
    rows = []
    with torch.inference_mode():  # Exact offline evaluator grad-mode boundary.
        for index, (name, mask_value) in enumerate(CONFIGURATIONS, 1):
            mask = torch.tensor([canonical_mask(mask_value)], dtype=torch.bool, device="cuda")
            before = offline_guard.calls
            prediction = sliding_window_inference(offline_guard, images, 4, 96, overlap=0.5, masks=mask)
            expected = native_predictions.pop(mask_value).to("cuda", dtype=prediction.dtype)
            mismatches = int((prediction != expected).sum())
            require(tuple(prediction.shape) == tuple(target.shape) and mismatches == 0,
                    f"Native/offline predicted labels differ for mask {mask_value}: {mismatches} voxels")
            require(offline_guard.calls - before == native_call_counts[mask_value],
                    "Native/offline sliding-window count differs")
            values = native_region_metrics(prediction, target)
            require_scores(values, (*REGIONS, "ET_postpro"))
            require(values == native_scores[mask_value], "Native Dice metric parity failed")
            restored = restore_full_grid(prediction[0], geometry["source_shape"],
                                         geometry["crop_bounds"], geometry["padding"])
            require(tuple(restored.shape) == tuple(full_target.shape) == tuple(geometry["source_shape"]),
                    "Original MRI grid restoration shape differs")
            common = avmur_region_dice(restored, full_target)
            require_scores(common, REGIONS)
            rows.append(dict(configuration=name, mask_value=mask_value, label_mismatches=mismatches,
                             label_sha256=tensor_digest(prediction.to(dtype=torch.uint8)),
                             native_scores=values, common_grid_scores=common,
                             sliding_windows=offline_guard.calls - before))
            progress(stage="offline_parity", completed=index, total=15,
                     configuration=name, mask_value=mask_value, state="passed", label_mismatches=0)
    require(state_digest(offline_base.state_dict()) == state_sha, "Offline inference changed model state")
    require(tensor_digest(images) == input_digest, "Inference changed the input tensor")
    require(len(rows) == 15 and len({row["mask_value"] for row in rows}) == 15,
            "Incomplete modality parity coverage")
    require(case_file_identity(protocol, cid, pid) == identity
            and {path: sha256_file(path) for path in identity} == payload_sha,
            "Case input bytes changed during parity")
    require(sha256_file(protocol_path) == protocol_sha, "Frozen protocol changed during parity")
    validate_protocol(read_json(protocol_path))
    require(source_fingerprints(REPO) == sources, "Source changed during parity")
    receipt = dict(
        schema_version=1, state="passed", check="full_unime_native_test_offline_inference_parity",
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
        protocol_path=str(protocol_path), protocol_sha256=protocol_sha,
        script_sha256=sha256_file(__file__), source_sha256=sources,
        case_id=cid, processed_id=pid, partition="selection80", case_choice="first_frozen_case",
        test_payloads_loaded=False, inference_cases=1, configuration_count=15,
        weights="seed40_random_initialization_not_trained", model_state_sha256=state_sha,
        parameter_count=parameter_count, seed=40, compile=True, precision="fp16_forward_fp32_swi",
        native_grad_mode="no_grad", offline_grad_mode="inference_mode",
        input_tensor_sha256=input_digest, input_file_sha256=payload_sha, rows=rows,
        geometry={key: value for key, value in geometry.items() if key != "target"},
        runtime=runtime_provenance(),
        limits=["One selection case and random weights; not trained-model accuracy or checkpoint transfer.",
                "Compares native prepared test path, not the unwrapped native training-validation path.",
                "Production TF32/cuDNN settings retained; seeded state does not guarantee kernel determinism.",
                "Common-grid Dice intentionally differs from native-grid Dice; no equality is asserted."])
    write_json_exclusive(receipt, args.output)
    progress(stage="inference_parity", state="passed", receipt=str(args.output.resolve()),
             configuration_count=15, label_mismatches=0)
    accelerator.end_training()


if __name__ == "__main__":
    main()
