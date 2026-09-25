"""Pure-standard-library membership, provenance and scoring contract.

No Torch/NumPy imports, patient-image decoding, training, or GPU work occur here.
Native channel order is FLAIR,T1c,T1,T2; canonical mask bits are T1=1,T1c=2,
T2=4,FLAIR=8. WT/TC/ET are raw original-MRI-grid Dice (epsilon1e-6).
ET_postpro is a native-processed-grid diagnostic (epsilon1e-8), never used in
mean_raw45 selection. The runner also records native_WT/native_TC/native_ET
separately; these additional fields never enter this helper's primary average.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

PROTOCOL_ID = "unime_original80_all15_rawdice_v1"
SPLIT_SHA256 = "84c430662622560f70055313417373f8dad0729fb20773e7e2e13238d215b831"
PARTITION_SHA256 = "7f4a427d3c0098cdd23e49bbfc45225ce0fe6378461b01d50955e4ebbea23bf3"
PREPROCESSOR_SHA256 = "4c9477ef950013dd3bd6f1178ec45ff211d841424e94fa3beca37d3ecc4f8b42"
TRAINING_PRIOR_SHAPE = (160, 180, 210)  # D,H,W; effective official original pipeline
EVALUATOR_SHA256 = "a4f4551d8bef58f9de00c5acf71e7d58ad9d7101b6eeb40f174db11f8cbf9464"
CHANNEL_ORDER = ["FLAIR", "T1c", "T1", "T2"]
LABEL_MAPPING = {"background": 0, "NCR_NET": 1, "edema": 2, "ET": 3}
REGIONS = ("WT", "TC", "ET")
SCORES = (*REGIONS, "ET_postpro")
CONFIGURATIONS = (("FLAIR", 8), ("T1", 1), ("T1c", 2), ("T2", 4),
                  ("FLAIR+T1", 9), ("FLAIR+T1c", 10), ("FLAIR+T2", 12),
                  ("T1+T1c", 3), ("T1+T2", 5), ("T1c+T2", 6),
                  ("FLAIR+T1+T1c", 11), ("FLAIR+T1+T2", 13),
                  ("FLAIR+T1c+T2", 14), ("T1+T1c+T2", 7),
                  ("FLAIR+T1+T1c+T2", 15))
EXPECTED_EPOCHS = [epoch for epoch in range(1, 601)
                   if (epoch > 450 and epoch % 10 == 0) or epoch > 590 or epoch == 600]
CASE_PATTERN = re.compile(r"BraTS-GLI-\d{5}-\d{3}\Z")
PROCESSED_PATTERN = re.compile(r"BraTS2023_\d{5}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(value, path):
    """Create only; never replace an existing frozen protocol or selection."""
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    require(not path.is_symlink(), "Refusing output symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(value, path):
    """Atomic update for mutable progress; use exclusive writes for protocols."""
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    require(not path.is_symlink(), "Refusing output symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


write_json_atomic = atomic_json


def _case_list(values, count, name):
    require(isinstance(values, list) and len(values) == count, f"Wrong {name} count")
    require(all(isinstance(cid, str) and CASE_PATTERN.fullmatch(cid) for cid in values),
            f"Invalid source case ID in {name}")
    require(len(set(values)) == count, f"Duplicate case in {name}")
    patients = [cid.rsplit("-", 1)[0] for cid in values]
    # The frozen official training partition includes distinct scans of the same
    # patient. Only cross-partition patient overlap is forbidden; case IDs remain
    # unique within every partition.
    return set(patients)


def validate_split(split):
    require(isinstance(split, dict), "Split must be an object")
    patients = {}
    for name, count in (("train", 875), ("val", 125), ("test", 251)):
        patients[name] = _case_list(split.get(name), count, name)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        require(not patients[left] & patients[right], f"Patient overlap: {left}/{right}")
    return {name: list(split[name]) for name in ("train", "val", "test")}


def validate_partition(partition, split):
    validate_split(split)
    require(isinstance(partition, dict), "Partition must be an object")
    tune = partition.get("tune_cases")
    development = partition.get("lock_cases")
    tune_patients = _case_list(tune, 80, "selection80")
    development_patients = _case_list(development, 45, "development45")
    require(not tune_patients & development_patients, "80/45 patient overlap")
    require(set(tune) | set(development) == set(split["val"]), "80/45 union is not official validation")
    require(partition.get("validation_cases") == 125, "Partition validation count differs")
    return {"selection80": list(tune), "development45": list(development)}


def validate_mapping(mapping, split):
    validate_split(split)
    require(isinstance(mapping, dict) and len(mapping) == 1251, "Mapping must contain1251 entries")
    require(all(isinstance(pid, str) and PROCESSED_PATTERN.fullmatch(pid) for pid in mapping),
            "Invalid processed ID (path components are forbidden)")
    require(all(isinstance(cid, str) and CASE_PATTERN.fullmatch(cid) for cid in mapping.values()),
            "Invalid source ID in mapping")
    require(len(set(mapping.values())) == 1251, "Mapping is not one-to-one")
    require(set(mapping.values()) == set().union(*(set(split[name]) for name in ("train", "val", "test"))),
            "Mapping population differs from official split")
    return {case: processed for processed, case in mapping.items()}


def native_mask(mask_value):
    """Convert canonical T1,T1c,T2,FLAIR bits to native FLAIR,T1c,T1,T2."""
    require(type(mask_value) is int and 1 <= mask_value <= 15, "Invalid canonical mask")
    return [bool(mask_value & bit) for bit in (8, 2, 1, 4)]


def declared_mapping_provenance():
    """Pin the existing native adapter semantics, without importing its dependencies."""
    package = Path(__file__).resolve().parents[1]
    sources = {"scripts/brats23_process.py": PREPROCESSOR_SHA256,
               "source/core/evaluation.py": EVALUATOR_SHA256}
    for name, expected in sources.items():
        require(sha256_file(package / name) == expected, f"Native mapping source changed: {name}")
    return {"channel_order": list(CHANNEL_ORDER), "label_mapping": dict(LABEL_MAPPING),
            "raw_label_conversion": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 3},
            "canonical_mask_bits": {"T1": 1, "T1c": 2, "T2": 4, "FLAIR": 8},
            "region_labels": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]},
            "native_source_sha256": sources,
            "verification_scope": "Pinned declared source semantics; array headers cannot prove channel contents"}


def read_npy_header(path):
    """Validate only header and exact payload length; never decode array values."""
    path = Path(path)
    with path.open("rb") as stream:
        require(stream.read(6) == b"\x93NUMPY", f"Invalid NPY magic: {path}")
        version = stream.read(2)
        require(version in (b"\x01\x00", b"\x02\x00", b"\x03\x00"), "Unsupported NPY version")
        length_size = 2 if version[0] == 1 else 4
        encoded_length = stream.read(length_size)
        require(len(encoded_length) == length_size, "Truncated NPY header length")
        length = int.from_bytes(encoded_length, "little")
        require(0 < length <= 65536, "Invalid/oversized NPY header")
        header_bytes = stream.read(length)
        require(len(header_bytes) == length, "Truncated NPY header")
        header = ast.literal_eval(header_bytes.decode("utf-8" if version[0] == 3 else "latin1"))
        require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"},
                "Unexpected NPY header fields")
        require(header["descr"] in ("<f4", "=f4", "|u1"), "Unexpected NPY dtype")
        require(type(header["fortran_order"]) is bool, "Invalid NPY array order")
        shape = header["shape"]
        require(isinstance(shape, tuple) and 1 <= len(shape) <= 4
                and all(type(dim) is int and dim > 0 for dim in shape), "Invalid NPY shape")
        itemsize = 1 if header["descr"] == "|u1" else 4
        require(path.stat().st_size == stream.tell() + math.prod(shape) * itemsize,
                "NPY payload size mismatch")
    return header


def audit_array_headers(data_root, mapped_ids):
    root = Path(data_root).resolve()
    checked = 0
    for partition in ("train", "val", "test"):
        for pid in mapped_ids[partition]:
            paths = [root / "vol" / f"{pid}_vol.npy", root / "seg" / f"{pid}_seg.npy"]
            require(all(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)
                        for path in paths), f"Missing/unsafe array pair: {pid}")
            volume, label = [read_npy_header(path) for path in paths]
            require(volume["descr"] in ("<f4", "=f4") and label["descr"] == "|u1", "Wrong array dtypes")
            require(len(volume["shape"]) == 4 and volume["shape"][-1] == 4
                    and volume["shape"][:3] == label["shape"], f"Misaligned shapes: {pid}")
            require(min(label["shape"]) >= 128, f"Native minimum128 padding absent: {pid}")
            if partition == "train":
                h, w, d = label["shape"]
                require(all(size <= limit for size, limit in zip((d, h, w), TRAINING_PRIOR_SHAPE)),
                        f"Training shape {(d, h, w)} exceeds learned prior {TRAINING_PRIOR_SHAPE}: {pid}")
            checked += 2
    require(checked == 2502, "Incomplete array header audit")
    return {"mode": "all_headers_and_exact_payload_sizes", "arrays_checked": checked,
            "array_values_scanned": False, "gpu_used": False}


def prepare_protocol(split_json, validation_partition, data_root, *, raw_data_root=None, verify_arrays=True):
    require(raw_data_root is not None, "Original MRI raw_data_root is required for full-grid scoring")
    raw_data_root = Path(raw_data_root).resolve()
    require(raw_data_root.is_dir(), "Original MRI raw_data_root is not a directory")
    split_json, validation_partition, data_root = [Path(path).resolve()
                                                   for path in (split_json, validation_partition, data_root)]
    require(sha256_file(split_json) == SPLIT_SHA256, "Not the frozen official split")
    require(sha256_file(validation_partition) == PARTITION_SHA256, "Not the original frozen80/45 partition")
    split = validate_split(read_json(split_json))
    subsets = validate_partition(read_json(validation_partition), split)
    mapping_path = data_root / "mapping.json"
    mapping = read_json(mapping_path)
    inverse = validate_mapping(mapping, split)
    cases = {**split, **subsets}
    mapped = {name: [inverse[cid] for cid in ids] for name, ids in cases.items()}
    input_paths = {"split_json": str(split_json), "validation_partition": str(validation_partition),
                   "mapping": str(mapping_path)}
    for name in ("train", "val", "test"):
        path = data_root / f"{name}.txt"
        require(path.read_text(encoding="utf-8").splitlines() == mapped[name],
                f"Native {name}.txt membership/order differs")
        input_paths[name + "_manifest"] = str(path)
    provenance = declared_mapping_provenance()
    arrays = (audit_array_headers(data_root, mapped) if verify_arrays else
              {"mode": "manifest_only", "arrays_checked": 0, "array_values_scanned": False, "gpu_used": False})
    result = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "case_ids": cases,
              "mapped_ids": mapped, "processed_to_case": mapping, "data_root": str(data_root),
              "raw_data_root": str(raw_data_root),
              "input_paths": input_paths,
              "input_hashes": {name: sha256_file(path) for name, path in input_paths.items()},
              "expected_epochs": list(EXPECTED_EPOCHS), "selection_cases": 80,
              "development_cases": 45, "test_cases": 251, "configuration_count": 15,
              "selection_metric": "mean_raw45", "tie_break": "earliest_epoch",
              "selection_score_fields": list(REGIONS), "diagnostic_score_fields": ["ET_postpro"],
              "scoring_domain": "original_mri_grid", "dice_epsilon": 1e-6,
              "common_full_grid_scoring_approved": True,
              "native_diagnostic_grid": "native_processed_grid", "native_dice_epsilon": 1e-8,
              "native_score_fields": ["native_WT", "native_TC", "native_ET", "ET_postpro"],
              "ET_postpro_diagnostic_grid": "native_processed_grid",
              "aggregation": "python_sum_in_supplied_case_then_configuration_row_order",
              "pretraining_selection_unchanged": True, "model_variant": "native_EMA",
              "development_used_for_selection": False, "test_used_for_selection": False,
              "development_is_independent_unseen": False, "test_is_independent_unseen": False,
              "native_mapping_provenance": provenance, "array_audit": arrays}
    validate_protocol(result)
    return result


def validate_protocol(protocol, *, verify_input_files=True):
    require(isinstance(protocol, dict) and protocol.get("protocol_id") == PROTOCOL_ID, "Wrong benchmark protocol")
    cases = protocol.get("case_ids", {})
    split = validate_split(cases)
    validate_partition({"tune_cases": cases.get("selection80"), "lock_cases": cases.get("development45"),
                        "validation_cases": 125}, split)
    inverse = validate_mapping(protocol.get("processed_to_case"), split)
    expected_mapped = {name: [inverse[cid] for cid in cases[name]]
                       for name in ("train", "val", "selection80", "development45", "test")}
    require(protocol.get("mapped_ids") == expected_mapped, "Mapped partition IDs changed")
    for field, expected in {"expected_epochs": EXPECTED_EPOCHS, "selection_cases": 80,
                            "development_cases": 45, "test_cases": 251, "configuration_count": 15,
                            "selection_metric": "mean_raw45", "tie_break": "earliest_epoch",
                            "selection_score_fields": list(REGIONS), "diagnostic_score_fields": ["ET_postpro"],
                            "scoring_domain": "original_mri_grid", "dice_epsilon": 1e-6,
                            "common_full_grid_scoring_approved": True,
                            "native_diagnostic_grid": "native_processed_grid", "native_dice_epsilon": 1e-8,
                            "native_score_fields": ["native_WT", "native_TC", "native_ET", "ET_postpro"],
                            "ET_postpro_diagnostic_grid": "native_processed_grid",
                            "aggregation": "python_sum_in_supplied_case_then_configuration_row_order",
                            "model_variant": "native_EMA", "pretraining_selection_unchanged": True,
                            "development_used_for_selection": False, "test_used_for_selection": False}.items():
        require(protocol.get(field) == expected, f"Protocol field changed: {field}")
    raw_data_root = protocol.get("raw_data_root")
    require(isinstance(raw_data_root, str) and Path(raw_data_root).is_absolute(),
            "Original MRI raw_data_root must be an absolute path")
    provenance = protocol.get("native_mapping_provenance", {})
    require(provenance.get("channel_order") == CHANNEL_ORDER and provenance.get("label_mapping") == LABEL_MAPPING,
            "Native channel/label declaration differs")
    require(provenance.get("native_source_sha256") == {
        "scripts/brats23_process.py": PREPROCESSOR_SHA256, "source/core/evaluation.py": EVALUATOR_SHA256},
        "Native mapping provenance changed")
    hashes = protocol.get("input_hashes", {})
    require(hashes.get("split_json") == SPLIT_SHA256 and hashes.get("validation_partition") == PARTITION_SHA256,
            "Frozen split/partition identity changed")
    paths = protocol.get("input_paths", {})
    required = {"split_json", "validation_partition", "mapping", "train_manifest", "val_manifest", "test_manifest"}
    require(set(paths) == set(hashes) == required, "Input provenance fields differ")
    require(all(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
            "Invalid provenance digest")
    if verify_input_files:
        require(Path(raw_data_root).is_dir(), "Original MRI raw_data_root is not a directory")
        for name, path in paths.items():
            require(sha256_file(path) == hashes[name], f"Protocol input bytes changed: {name}")
        source_split = validate_split(read_json(paths["split_json"]))
        require(all(cases[name] == source_split[name] for name in ("train", "val", "test")),
                "Embedded official case membership/order changed")
        source_subsets = validate_partition(read_json(paths["validation_partition"]), source_split)
        require(all(cases[name] == source_subsets[name] for name in ("selection80", "development45")),
                "Embedded80/45 membership/order changed")
        require(read_json(paths["mapping"]) == protocol["processed_to_case"], "Embedded mapping changed")
        for name in ("train", "val", "test"):
            require(Path(paths[name + "_manifest"]).read_text(encoding="utf-8").splitlines() == expected_mapped[name],
                    "Native manifest differs from embedded IDs: " + name)
    return protocol


def validate_rows(rows, case_ids):
    """Return complete, unrounded equal-case/mask/region macro averages.

    Preserve supplied row order and Python ``sum`` exactly as V11's
    ``summarize_rows``. The runner emits cases in its declared case order and
    masks in CONFIGURATIONS order. ET_postpro is a native-grid diagnostic only;
    additional native_* columns do not contribute to primary mean_raw45.
    """
    case_ids = list(case_ids)
    require(case_ids and len(set(case_ids)) == len(case_ids) and all(isinstance(cid, str) for cid in case_ids),
            "Empty/duplicate expected case IDs")
    require(isinstance(rows, list), "Rows must be a list")
    expected = {(cid, mask) for cid in case_ids for mask in range(1, 16)}
    keys = []
    for row in rows:
        require(isinstance(row, dict), "Row must be an object")
        cid, mask = row.get("case_id"), row.get("mask_value")
        require(isinstance(cid, str) and type(mask) is int and 1 <= mask <= 15, "Invalid case/mask key")
        keys.append((cid, mask))
        for score in SCORES:
            value = row.get(score)
            require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
                    f"Invalid/missing score: {score}")
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete/duplicate case-by-mask coverage")
    by_configuration = {str(mask): {score: sum(row[score] for row in rows if row["mask_value"] == mask) / len(case_ids)
                                     for score in SCORES} for _, mask in CONFIGURATIONS}
    means = {score: sum(row[score] for row in rows) / len(rows) for score in SCORES}
    return {**means, "mean_raw45": sum(means[score] for score in REGIONS) / 3,
            "by_configuration": by_configuration, "case_count": len(case_ids), "row_count": len(rows),
            "configuration_count": 15}


def rank_records(records, expected_epochs=None):
    require(isinstance(records, list) and records, "No selection records")
    epochs = [row.get("epoch") for row in records]
    require(all(type(epoch) is int and epoch in EXPECTED_EPOCHS for epoch in epochs)
            and len(set(epochs)) == len(epochs), "Invalid/duplicate checkpoint epochs")
    if expected_epochs is not None:
        require(sorted(epochs) == sorted(expected_epochs), "Missing/extra selection checkpoints")
    for row in records:
        require(row.get("case_count") == 80 and row.get("row_count") == 1200,
                "Selection requires80cases times15masks")
        value = row.get("mean_raw45")
        require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1, "Invalid selection score")
    return sorted(records, key=lambda row: (-row["mean_raw45"], row["epoch"]))
