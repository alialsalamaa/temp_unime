"""Verify processed UniME inputs against the frozen study split before training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from brats23_process import load_existing_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads((args.data_root / "mapping.json").read_text())
    expected = load_existing_splits(args.split_json, mapping)
    counts = {}
    maximum_shape = np.zeros(3, dtype=int)
    for partition, ids in expected.items():
        actual = (args.data_root / f"{partition}.txt").read_text().splitlines()
        if actual != ids:
            raise RuntimeError(f"{partition}: processed membership/order differs from the study split")
        for case in ids:
            volume = np.load(args.data_root / "vol" / f"{case}_vol.npy", mmap_mode="r", allow_pickle=False)
            label = np.load(args.data_root / "seg" / f"{case}_seg.npy", mmap_mode="r", allow_pickle=False)
            try:
                if volume.ndim != 4 or volume.shape[-1] != 4 or volume.shape[:3] != label.shape:
                    raise RuntimeError(f"{case}: invalid volume/label shapes {volume.shape}/{label.shape}")
                if volume.dtype != np.float32 or label.dtype != np.uint8:
                    raise RuntimeError(f"{case}: invalid dtypes {volume.dtype}/{label.dtype}")
                if min(label.shape) < 128:
                    raise RuntimeError(f"{case}: minimum-size padding missing")
                dhw = (volume.shape[2], volume.shape[0], volume.shape[1])
                if partition == "train" and any(size > limit for size, limit in zip(dhw, (160, 180, 210))):
                    raise RuntimeError(f"{case}: training shape {dhw} exceeds the configured prior")
                maximum_shape = np.maximum(maximum_shape, dhw)
            finally:
                volume._mmap.close()
                label._mmap.close()
        counts[partition] = len(ids)
    print(json.dumps({"status": "verified", "counts": counts,
                      "max_shape_dhw": maximum_shape.tolist(),
                      "split_sha256": hashlib.sha256(args.split_json.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
