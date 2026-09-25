#!/usr/bin/env python3
"""Prepare the frozen UniME dataset using unchanged native case preprocessing.

Four single-thread workers run on CPUs in an existing allocation. Validated
case pairs are committed by atomic renames; a completion record is written last.
A restart validates both arrays and provenance before reusing a completed case.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "UniME"
SPLIT = ROOT / "artifacts/splits/official_imfuse_split.json"
RAW = ROOT / "data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
DATA = ROOT / "artifacts/unime/data/BRATS2023"
PROOF = ROOT / "artifacts/unime/preprocessing"
EXPECTED_SHA = "84c430662622560f70055313417373f8dad0729fb20773e7e2e13238d215b831"
COUNTS = {"train": 875, "val": 125, "test": 251}
SUFFIXES = ("t2f", "t1c", "t1n", "t2w", "seg")
sys.path.insert(0, str(REPO / "scripts"))
from brats23_process import load_existing_splits, process_case, write_existing_splits


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def stat(path):
    item = Path(path).stat()
    return {"size": item.st_size, "mtime_ns": item.st_mtime_ns, "ctime_ns": item.st_ctime_ns}


def raw_stats(case):
    return {suffix: stat(RAW / case / f"{case}-{suffix}.nii.gz") for suffix in SUFFIXES}


def validate_pair(vol_path, seg_path):
    volume = np.load(vol_path, mmap_mode="r", allow_pickle=False)
    label = None
    try:
        label = np.load(seg_path, mmap_mode="r", allow_pickle=False)
        if volume.ndim != 4 or label.ndim != 3 or volume.shape != (*label.shape, 4):
            raise ValueError(f"Invalid shapes: {volume.shape}, {label.shape}")
        if volume.dtype != np.float32 or label.dtype != np.uint8:
            raise ValueError(f"Invalid dtypes: {volume.dtype}, {label.dtype}")
        if min(label.shape) < 128:
            raise ValueError(f"Missing native minimum-128 padding: {label.shape}")
        if not np.isfinite(volume).all():
            raise ValueError(f"Non-finite image values: {vol_path}")
        unique = np.unique(label).tolist()
        if not set(unique).issubset({0, 1, 2, 3}):
            raise ValueError(f"Invalid labels: {unique}")
        return {"shape_hwdc": list(volume.shape), "volume_dtype": str(volume.dtype),
                "segmentation_dtype": str(label.dtype), "labels": unique,
                "all_image_values_finite": True, "valid_label_values": True}
    finally:
        volume._mmap.close()
        if label is not None:
            label._mmap.close()


def prepare_one(task):
    out_id, case, partition, provenance = task
    record_path = PROOF / "cases" / f"{out_id}.json"
    vol_path = DATA / "vol" / f"{out_id}_vol.npy"
    seg_path = DATA / "seg" / f"{out_id}_seg.npy"
    before_raw = raw_stats(case)
    expected = {"output_id": out_id, "source_case": case, "partition": partition,
                "provenance": provenance, "raw_file_stats": before_raw}
    started = time.monotonic()
    reused = False
    recovery_reason = None
    if record_path.exists() and vol_path.exists() and seg_path.exists():
        try:
            previous = json.loads(record_path.read_text())
            if any(previous.get(key) != value for key, value in expected.items()):
                raise ValueError("Source/provenance metadata changed")
            if previous.get("output_file_stats") != {"vol": stat(vol_path), "seg": stat(seg_path)}:
                raise ValueError("Output metadata changed")
            detail = validate_pair(vol_path, seg_path)
            reused = True
        except (ValueError, OSError, KeyError, EOFError) as error:
            recovery_reason = str(error)
    if not reused:
        with tempfile.TemporaryDirectory(prefix=f"{out_id}-", dir=DATA / ".staging") as temporary:
            stage = Path(temporary)
            (stage / "vol").mkdir()
            (stage / "seg").mkdir()
            changed = process_case(RAW / case, out_id, stage / "vol", stage / "seg", 128, False)
            if not changed:
                raise RuntimeError(f"Native preprocessing unexpectedly skipped {case}")
            stage_vol = stage / "vol" / vol_path.name
            stage_seg = stage / "seg" / seg_path.name
            detail = validate_pair(stage_vol, stage_seg)
            if raw_stats(case) != before_raw:
                raise RuntimeError(f"Raw data changed during preprocessing: {case}")
            os.replace(stage_vol, vol_path)
            os.replace(stage_seg, seg_path)
    if raw_stats(case) != before_raw:
        raise RuntimeError(f"Raw data changed during validation: {case}")
    result = {**expected, **detail, "status": "passed", "reused": reused,
              "recovery_reason": recovery_reason, "completed_utc": utc(),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "output_file_stats": {"vol": stat(vol_path), "seg": stat(seg_path)}}
    atomic_json(record_path, result)
    return result


def run(workers):
    started = time.monotonic()
    if sha(SPLIT) != EXPECTED_SHA:
        raise RuntimeError("Authoritative split hash changed")
    split = json.loads(SPLIT.read_text())
    if {key: len(split[key]) for key in COUNTS} != COUNTS:
        raise RuntimeError("Unexpected split counts")
    case_dirs = sorted(path for path in RAW.iterdir() if path.is_dir())
    mapping = {f"BraTS2023_{index:05d}": path.name for index, path in enumerate(case_dirs)}
    native_splits = load_existing_splits(SPLIT, mapping)
    partitions = {case: partition for partition in COUNTS for case in split[partition]}
    source_hashes = {name: sha(REPO / "scripts" / name)
                     for name in ("brats23_process.py", "verify_study_split.py")}
    provenance = {"split_sha256": EXPECTED_SHA, "native_source_sha256": source_hashes,
                  "minimum_spatial_size": 128, "output_prefix": "BraTS2023",
                  "channel_order": ["FLAIR", "T1c", "T1", "T2"],
                  "raw_root": str(RAW), "native_process_case_unchanged": True}
    for folder in (DATA / "vol", DATA / "seg", DATA / ".staging", PROOF / "cases"):
        folder.mkdir(parents=True, exist_ok=True)
    mapping_path = DATA / "mapping.json"
    if mapping_path.exists() and json.loads(mapping_path.read_text()) != mapping:
        raise RuntimeError("Existing production mapping differs from the exact native sorted mapping")
    for kind in ("vol", "seg"):
        expected_names = {f"{out_id}_{kind}.npy" for out_id in mapping}
        extra = {path.name for path in (DATA / kind).iterdir()} - expected_names
        if extra:
            raise RuntimeError(f"Unexpected {kind} outputs: {sorted(extra)[:5]}")
    atomic_json(mapping_path, mapping)
    write_existing_splits(DATA, native_splits)
    manifest = {"status": "running", "started_utc": utc(), "workers": workers,
                "threads_per_worker": 1, "gpu_used": False, "data_root": str(DATA),
                "split_counts": COUNTS, "total_cases": len(mapping), "provenance": provenance,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                "new_slurm_jobs_submitted": 0,
                "baseline_source_edited": False, "environment_modified": False,
                "raw_files_modified": False, "segmentation_statistics_shared_across_cases": False}
    atomic_json(PROOF / "run_manifest.json", manifest)
    atomic_json(PROOF / "status.json", {"status": "running", "completed_cases": 0, "total_cases": len(mapping)})
    tasks = [(out_id, case, partitions[case], provenance) for out_id, case in mapping.items()]
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(prepare_one, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                record = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                atomic_json(PROOF / "error.json", {"status": "failed", "case": task[1],
                            "time_utc": utc(), "traceback": traceback.format_exc()})
                raise
            records.append(record)
            progress = {"status": "running", "completed_cases": len(records),
                        "total_cases": len(mapping), "latest_case": task[1], "updated_utc": utc()}
            atomic_json(PROOF / "status.json", progress)
            print(json.dumps({**progress, "reused": record["reused"],
                              "shape_hwdc": record["shape_hwdc"]}), flush=True)
    if sha(SPLIT) != EXPECTED_SHA or any(sha(REPO / "scripts" / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("Split or baseline source changed during preprocessing")
    for kind in ("vol", "seg"):
        if {path.name for path in (DATA / kind).iterdir()} != {f"{out_id}_{kind}.npy" for out_id in mapping}:
            raise RuntimeError(f"Final {kind} inventory does not exactly match the frozen dataset")
    for record in records:
        out_id = record["output_id"]
        final_stats = {kind: stat(DATA / kind / f"{out_id}_{kind}.npy") for kind in ("vol", "seg")}
        if final_stats != record["output_file_stats"]:
            raise RuntimeError(f"Prepared outputs changed after their full value scan: {out_id}")
        if raw_stats(record["source_case"]) != record["raw_file_stats"]:
            raise RuntimeError(f"Raw input changed: {record['source_case']}")
    command = [sys.executable, "-B", str(REPO / "scripts/verify_study_split.py"),
               "--data-root", str(DATA), "--split-json", str(SPLIT)]
    check = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (PROOF / "native_split_verification.log").write_text(check.stdout)
    if check.returncode:
        raise RuntimeError(f"Native split verifier failed ({check.returncode}): {check.stdout}")
    native_result = json.loads(check.stdout)
    if native_result.get("status") != "verified" or native_result.get("counts") != COUNTS:
        raise RuntimeError(f"Unexpected native verifier result: {native_result}")
    report = {**manifest, "status": "passed", "completed_utc": utc(),
              "full_dataset_preprocessed": True, "full_preprocessing_complete": True,
              "full_value_scan_passed": True, "arrays_validated": len(records) * 2,
              "cases_validated": len(records), "processed_cases": sum(not r["reused"] for r in records),
              "reused_cases": sum(r["reused"] for r in records), "minimum_spatial_size": 128,
              "split_sha256": EXPECTED_SHA, "elapsed_seconds": round(time.monotonic() - started, 3),
              "native_split_verification": native_result, "native_split_verification_command": command,
              "source_hashes_unchanged": True,
              "prepared_bytes": sum(v["size"] for r in records for v in r["output_file_stats"].values()),
              "case_records_directory": str(PROOF / "cases"),
              "array_scan_scope": "Every image value and every segmentation label checked during case preparation or resume validation; final file metadata and native split rechecked."}
    atomic_json(PROOF / "preprocessing_report.json", report)
    atomic_json(PROOF / "status.json", {"status": "passed", "completed_cases": len(records),
                "total_cases": len(mapping), "completed_utc": utc(),
                "report": str(PROOF / "preprocessing_report.json")})
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    PROOF.mkdir(parents=True, exist_ok=True)
    with (PROOF / "preprocessing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.workers)
        except BaseException:
            atomic_json(PROOF / "status.json", {"status": "failed", "updated_utc": utc(),
                        "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
