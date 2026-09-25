"""Explicit offline scoring utilities; importing this module selects no protocol.

UniME's native inference and preprocessing remain untouched.  Full-grid helpers
are opt-in building blocks for a separately declared evaluation mode.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


REGIONS = ("WT", "TC", "ET")


def canonical_mask(mask_value: int) -> tuple[bool, bool, bool, bool]:
    """Map AVMUR bits (T1,T1c,T2,FLAIR) to UniME (FLAIR,T1c,T1,T2)."""
    if isinstance(mask_value, bool) or not isinstance(mask_value, Integral):
        raise ValueError("mask_value must be an integer from 1 through 15")
    if not 1 <= mask_value <= 15:
        raise ValueError("mask_value must be an integer from 1 through 15")
    return tuple(bool(mask_value & (1 << bit)) for bit in (3, 1, 0, 2))


def _case_labels(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    value = torch.as_tensor(value, device=device)
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1 or any(n < 1 for n in value.shape):
        raise ValueError("Expected one nonempty case with shape (H,W,D) or (1,H,W,D)")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise ValueError("Class labels must have an integer dtype")
    if bool(((value < 0) | (value > 3)).any()):
        raise ValueError("Class labels must lie in 0..3; map legacy ET label 4 before scoring")
    return value


def _aligned_labels(prediction: Any, target: Any) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _case_labels(prediction)
    target = _case_labels(target, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    return prediction, target


def native_metrics(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    """Delegate single-case scoring, including native ET postprocessing, unchanged."""
    from source.core.evaluation import evaluate_single_sample

    prediction, target = _aligned_labels(prediction, target)
    return evaluate_single_sample(prediction, target)


def native_region_metrics(prediction: Any, target: Any) -> dict[str, float]:
    """Native single-case WT/TC/ET and auxiliary ET_postpro scores."""
    _, scores = native_metrics(prediction, target)
    return dict(zip((*REGIONS, "ET_postpro"), map(float, scores[0])))


def avmur_region_dice(prediction: Any, target: Any) -> dict[str, float]:
    """Match V11 study.batch_dice, given already decoded single-case labels.

    Reductions and arithmetic are FP32, epsilon is 1e-6, both-empty is 1,
    and a disjoint nonempty pair keeps its small epsilon contribution.
    This function applies no postprocessing and performs no patient averaging.
    """
    prediction, target = _aligned_labels(prediction, target)

    def regions(labels: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (labels > 0, (labels == 1) | (labels == 3), labels == 3), dim=1
        ).float()

    p, g = regions(prediction), regions(target)
    dims = tuple(range(2, p.ndim))
    dice = (2 * (p * g).sum(dims) + 1e-6) / ((p + g).sum(dims) + 1e-6)
    values = dice.mean(0).float().cpu().tolist()
    return dict(zip(REGIONS, map(float, values)))


def _integers(values: Sequence[int], *, length: int, name: str) -> tuple[int, ...]:
    if len(values) != length or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError(f"{name} must contain {length} integers")
    return tuple(int(v) for v in values)


def _geometry(
    source_shape: Sequence[int],
    crop_bounds: Sequence[Sequence[int]],
    padding: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    shape = _integers(source_shape, length=3, name="source_shape")
    if any(n < 1 for n in shape) or len(crop_bounds) != 3 or len(padding) != 3:
        raise ValueError("Geometry requires three nonempty spatial axes")
    bounds = tuple(_integers(pair, length=2, name="crop_bounds") for pair in crop_bounds)
    pads = tuple(_integers(pair, length=2, name="padding") for pair in padding)
    for size, (start, stop), (before, after) in zip(shape, bounds, pads):
        if not 0 <= start < stop <= size or before < 0 or after < 0:
            raise ValueError("Invalid crop bounds or padding")
    return shape, bounds, pads


def restore_full_grid(
    prediction: np.ndarray | torch.Tensor,
    source_shape: Sequence[int],
    crop_bounds: Sequence[Sequence[int]],
    padding: Sequence[Sequence[int]],
) -> np.ndarray | torch.Tensor:
    """Unpad a 3-D prediction and insert it into its original grid as background.

    No resampling, orientation change, target-dependent crop, or metric is applied.
    Input dtype/device are preserved.  Incorrect geometry always raises.
    """
    if not isinstance(prediction, (np.ndarray, torch.Tensor)) or prediction.ndim != 3:
        raise ValueError("prediction must be a 3-D NumPy array or Torch tensor")
    shape, bounds, pads = _geometry(source_shape, crop_bounds, padding)
    expected = tuple(stop - start + before + after
                     for (start, stop), (before, after) in zip(bounds, pads))
    if tuple(prediction.shape) != expected:
        raise ValueError(f"Prediction shape {tuple(prediction.shape)} does not match geometry {expected}")
    unpad = tuple(slice(before, before + stop - start)
                  for (start, stop), (before, _) in zip(bounds, pads))
    destination = tuple(slice(start, stop) for start, stop in bounds)
    if isinstance(prediction, torch.Tensor):
        restored = prediction.new_zeros(shape)
    else:
        restored = np.zeros(shape, dtype=prediction.dtype)
    restored[destination] = prediction[unpad]
    return restored


def _verify_raw_nifti_geometry(raw_case: Path) -> dict[str, Any]:
    """Check spatial metadata and segmentation values before native lossy casts."""
    import nibabel as nib

    reference_shape = None
    reference_affine = None
    reference_orientation = None
    raw_labels = None
    for suffix in ("t2f", "t1c", "t1n", "t2w", "seg"):
        path = raw_case / f"{raw_case.name}-{suffix}.nii.gz"
        image = nib.load(str(path))
        shape = tuple(int(n) for n in image.shape)
        affine = np.asarray(image.affine, dtype=np.float64)
        if len(shape) != 3 or any(n < 1 for n in shape):
            raise ValueError(f"Raw NIfTI must have three nonempty spatial axes: {path}")
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"Raw NIfTI affine must be finite: {path}")
        if np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError(f"Raw NIfTI affine is singular: {path}")
        orientation = tuple(nib.aff2axcodes(affine))
        if any(axis is None for axis in orientation):
            raise ValueError(f"Raw NIfTI has undefined orientation: {path}")
        if reference_shape is None:
            reference_shape, reference_affine = shape, affine
            reference_orientation = orientation
        elif (shape != reference_shape or orientation != reference_orientation
              or not np.allclose(affine, reference_affine, rtol=0, atol=1e-5)):
            raise ValueError(f"Raw NIfTI shape/affine/orientation mismatch: {path}")
        if suffix == "seg":
            # Preserve the on-disk/scaled precision here. Native get_fdata(float32)
            # followed by uint8 could otherwise hide fractional, negative, or
            # wrapped labels such as 256, and must happen only after this check.
            raw_labels = np.asanyarray(image.dataobj)
            if (raw_labels.dtype.kind not in "iuf" or not np.isfinite(raw_labels).all()
                    or not np.isin(raw_labels, (0, 1, 2, 3, 4)).all()):
                raise ValueError("Raw segmentation must contain finite integer values 0,1,2,3,4 before casting")
    return {
        "shape": reference_shape,
        "affine": reference_affine.tolist(),
        "orientation": reference_orientation,
        "labels": np.asarray(raw_labels, dtype=np.uint8),
    }


def raw_geometry(
    raw_case: str | Path,
    processed_seg_path: str | Path,
    min_size: int = 128,
    *,
    processed_vol_path: str | Path | None = None,
    check_raw_headers: bool = True,
) -> dict[str, Any]:
    """Reconstruct native geometry and verify the processed segmentation exactly.

    Returns full raw target labels (legacy 4 mapped to 3), crop bounds, padding,
    and the number of tumor voxels outside the native crop.  Outside-crop tumor
    is deliberately retained in ``target`` so full-grid scoring counts errors.
    By default all raw NIfTI grids are checked and segmentation values validated
    before native uint8 casting. ``check_raw_headers=False`` is for synthetic
    in-memory tests only. If supplied, ``processed_vol_path`` is compared exactly
    against native normalization, channel order, crop, and padding as well.
    No file is written and no evaluation protocol is automatically activated.
    """
    from scripts import brats23_process as processing

    if isinstance(min_size, bool) or not isinstance(min_size, Integral) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    if not isinstance(check_raw_headers, bool):
        raise ValueError("check_raw_headers must be an explicit boolean")
    checked = _verify_raw_nifti_geometry(Path(raw_case)) if check_raw_headers else None
    volume, raw_target = processing.load_modalities(Path(raw_case))
    if volume.ndim != 4 or volume.shape[0] != 4 or raw_target.shape != volume.shape[1:]:
        raise ValueError("Raw images and segmentation must share one four-channel grid")
    if not np.isfinite(volume).all():
        raise ValueError("Raw images contain non-finite values")
    if not np.issubdtype(raw_target.dtype, np.integer) or not np.isin(raw_target, (0, 1, 2, 3, 4)).all():
        raise ValueError("Raw segmentation must use integer labels 0,1,2,3,4")
    if checked is not None and (tuple(volume.shape[1:]) != checked["shape"]
                                or not np.array_equal(raw_target, checked["labels"])):
        raise ValueError("Native raw loader does not match the verified NIfTI geometry/labels")
    target = raw_target.copy()
    target[target == 4] = 3
    slices = processing.find_crop_bounds(np.any(volume != 0, axis=0), min_size=int(min_size))
    bounds = tuple((int(axis.start), int(axis.stop)) for axis in slices)
    crop = target[slices]
    padding = tuple((max(0, int(min_size) - size) // 2,
                     max(0, int(min_size) - size) - max(0, int(min_size) - size) // 2)
                    for size in crop.shape)
    volume_crop = volume[(slice(None), *slices)]
    if processed_vol_path is not None:
        volume_crop = processing.normalize_channels(volume_crop)
    expected_volume, expected = processing.pad_to_min_size(
        np.transpose(volume_crop, (1, 2, 3, 0)).astype(np.float32), crop, int(min_size)
    )
    processed = np.load(Path(processed_seg_path), allow_pickle=False)
    if not np.issubdtype(processed.dtype, np.integer) or not np.array_equal(processed, expected):
        raise ValueError("Processed segmentation does not exactly match raw crop, label mapping, and padding")
    if processed_vol_path is not None:
        processed_volume = np.load(Path(processed_vol_path), allow_pickle=False, mmap_mode="r")
        if processed_volume.dtype != np.float32 or not np.array_equal(processed_volume, expected_volume):
            raise ValueError("Processed volume does not exactly match native channel order, normalization, crop, and padding")
    outside_count = int(np.count_nonzero(target)) - int(np.count_nonzero(crop))
    return {
        "source_shape": tuple(int(size) for size in target.shape),
        "crop_bounds": bounds,
        "padding": padding,
        "target": target,
        "outside_crop_tumor_voxels": outside_count,
        "raw_headers_verified": checked is not None,
        "source_affine": checked["affine"] if checked is not None else None,
        "source_orientation": checked["orientation"] if checked is not None else None,
        "processed_volume_verified": processed_vol_path is not None,
    }


def _outputs_to_fp32(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value
    if isinstance(value, tuple):
        values = tuple(_outputs_to_fp32(item) for item in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    if isinstance(value, list):
        return [_outputs_to_fp32(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _outputs_to_fp32(item) for key, item in value.items()}
    return value


class OfflineForwardFP32(nn.Module):
    """Autocast only the model forward, then return FP32 outputs to native SWI.

    This mirrors Accelerate's autocast-forward/convert-to-FP32 boundary without
    involving its training launcher.  The caller must explicitly select dtype;
    the default FP16 is not a statement about every historical training run.
    Native softmax and probability accumulation happen outside this context.
    """
    def __init__(
        self, model: nn.Module, *, dtype: torch.dtype = torch.float16,
        device_type: str = "cuda", enabled: bool = True,
    ) -> None:
        super().__init__()
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Offline mixed precision must be float16 or bfloat16")
        self.module = model
        self.dtype = dtype
        self.device_type = device_type
        self.enabled = bool(enabled)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=self.enabled):
            output = self.module(*args, **kwargs)
        return _outputs_to_fp32(output)
