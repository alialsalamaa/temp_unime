"""Offline original80 selection and one-checkpoint historical-test evaluation.

No training code or optimizer is called here. Native UniME inference is retained;
only inverse crop/pad geometry and the common full-grid metric are added.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

from source.benchmark_protocol import (
    CONFIGURATIONS, REGIONS, PROTOCOL_ID, read_json, sha256_file,
    validate_protocol, validate_rows, rank_records, write_json_exclusive,
    atomic_json, require,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_child(root, name):
    root = Path(root).resolve()
    require(isinstance(name, str) and bool(name), "Unsafe artifact path")
    path = root / name
    require(isinstance(name, str) and not Path(name).is_absolute()
            and path.resolve().is_relative_to(root) and not path.is_symlink(),
            "Unsafe artifact path")
    return path


def verify_capture(directory):
    """Reject incomplete captures, changed source, or mixed-run candidates."""
    from source.benchmark_capture import source_fingerprints
    directory = Path(directory).resolve()
    protocol_path = directory / "protocol.json"
    protocol = validate_protocol(read_json(protocol_path))
    audit = protocol["array_audit"]
    require(audit.get("mode") == "all_headers_and_exact_payload_sizes"
            and type(audit.get("arrays_checked")) is int and audit["arrays_checked"] == 2502,
            "Manifest-only protocol cannot be used for inference")
    digest = sha256_file(protocol_path)
    complete = read_json(directory / "TRAINING_COMPLETE.json")
    manifest = read_json(directory / "candidates.json")
    started = read_json(directory / "capture_started.json")
    for record in (complete, manifest, started):
        require(record.get("protocol_id") == PROTOCOL_ID
                and record.get("protocol_sha256") == digest
                and record.get("model_variant") == "EMA"
                and record.get("expected_epochs") == protocol["expected_epochs"],
                "Capture/protocol identity mismatch")
    require(complete.get("state") == "complete" and complete.get("completed_epochs") == 600
            and complete.get("automatic_test_performed") is False,
            "Training capture is incomplete or native testing was enabled")
    require(complete.get("manifest_sha256") == sha256_file(directory / "candidates.json"),
            "Candidate manifest changed")
    entries = manifest.get("entries", [])
    require([entry.get("epoch") for entry in entries] == protocol["expected_epochs"]
            and complete.get("candidate_count") == len(entries), "Incomplete native candidate schedule")
    require(started.get("source_sha256") == source_fingerprints(Path(__file__).resolve().parents[1]),
            "Code/dependencies changed since checkpoint capture")
    pretrained = started.get("pretrained_checkpoint", {})
    require(isinstance(pretrained.get("path"), str)
            and sha256_file(pretrained["path"]) == pretrained.get("sha256"),
            "Pretrained checkpoint provenance changed")
    for entry in entries:
        path = safe_child(directory, entry["path"])
        require(entry.get("model_variant") == "EMA" and sha256_file(path) == entry.get("sha256"),
                "Candidate weights changed")
        require(read_json(path.with_suffix(".json")) == entry, "Candidate sidecar changed")
    validate_training_config(started["training_config"])
    return protocol, entries, started


def validate_training_config(config):
    # These are the pre-existing, explicitly pinned local paper-configuration
    # values. This guard does not retune them or certify unresolved paper/code
    # differences as an exact reproduction of the authors' published run.
    expected = dict(model_name="UniME", uni_encoder_name="UniEncoder", seed=40, compile=True,
                    dataset_name="BRATS2023", split_type="Normal",
                    num_classes=4, crop_size=96, batch_size=4, num_epochs=600,
                    iter_per_epoch=250, use_ema=True, amp=True, bfloat16=False,
                    base_lr=3e-4, min_lr=1e-6, warmup_lr=1e-5, warmup_ratio=0.05,
                    weight_decay=1e-4, layer_decay=0.75, ema_decay=0.999)
    for key, value in expected.items():
        require(config.get(key) == value, "Unexpected training recipe: " + key)
    require(config.get("train_ratio") is None, "Few-shot training is not this benchmark")


def normalized_state(state):
    from source.utils.runtime import strip_runtime_prefixes
    result = {}
    for name, tensor in state.items():
        key = strip_runtime_prefixes(name)
        require(key not in result, "Colliding checkpoint keys after wrapper removal")
        result[key] = tensor
    return result


def load_candidate(model, directory, entry, protocol_digest, training_config):
    import torch
    path = safe_child(directory, entry["path"])
    require(sha256_file(path) == entry["sha256"], "Candidate bytes changed before load")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("epoch") == entry["epoch"] and payload.get("model_variant") == "EMA"
            and payload.get("protocol_sha256") == protocol_digest
            and payload.get("training_config") == training_config, "Candidate payload identity mismatch")
    state = normalized_state(payload["model_state_dict"])
    require(state and all(isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all())
                          for tensor in state.values()), "Invalid/nonfinite model tensors")
    model.load_state_dict(state, strict=True)
    model.eval()


def build_model(config, existing_snapshot):
    import torch
    from models.UniME.networks import UniMEModel
    from source.benchmark_metrics import OfflineForwardFP32
    from source.utils.runtime import set_model_is_training
    # mode=scratch suppresses the wrapper's pretraining load. The existing path
    # is resolved only; every parameter is replaced by the strict full-state load.
    model = UniMEModel(scale="Original", num_modals=4, num_classes=4, mode="scratch",
                       pretrained_path=str(existing_snapshot), layer_decay=config["layer_decay"],
                       checkpoint_config={"encoder": False, "regularizer": False, "decoder": False})
    model.uni_encoder.set_mode("finetune")
    model.eval()
    set_model_is_training(model, False)
    model.to("cuda")
    forward = torch.compile(model, mode="default") if config.get("compile") else model
    forward = OfflineForwardFP32(forward, dtype=torch.float16, enabled=True)
    forward.eval()
    set_model_is_training(forward, False)
    return model, forward


def native_summary(rows, case_ids):
    """Mirror native FP32 patient means followed by FP64 configuration means."""
    import numpy as np
    # Native BaseDataset sorts mapped names; common-grid rows instead follow the
    # frozen V11 case order. Preserve native reduction order independently.
    if rows and all("processed_id" in row for row in rows):
        rows = sorted(rows, key=lambda row: row["processed_id"])
    native_rows = [{"case_id": row["case_id"], "mask_value": row["mask_value"],
                    **{region: row["native_" + region] for region in REGIONS},
                    "ET_postpro": row["ET_postpro"]} for row in rows]
    validate_rows(native_rows, case_ids)
    fields = (*REGIONS, "ET_postpro")
    configurations = {}
    for _, mask in CONFIGURATIONS:
        values = np.asarray([[row[field] for field in fields] for row in native_rows
                             if row["mask_value"] == mask], dtype=np.float32)
        configurations[str(mask)] = dict(zip(fields, map(float, values.mean(axis=0))))
    averages = np.asarray([[configurations[str(mask)][field] for field in fields]
                           for _, mask in CONFIGURATIONS], dtype=np.float64).mean(axis=0)
    return {**dict(zip(fields, map(float, averages))), "by_configuration": configurations,
            "scoring_domain": "native_processed_grid", "dice_epsilon": 1e-8}


def case_file_identity(protocol, cid, pid):
    """Lightweight immutable-input guard; not a substitute for content validation."""
    raw = Path(protocol["raw_data_root"]) / cid
    data = Path(protocol["data_root"])
    paths = [raw / f"{cid}-{suffix}.nii.gz" for suffix in ("t2f", "t1c", "t1n", "t2w", "seg")]
    paths += [data / "vol" / f"{pid}_vol.npy", data / "seg" / f"{pid}_seg.npy"]
    return {str(path): dict(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in paths}


def evaluate_partition(forward, protocol, partition, destination, geometry_cache, *, evaluation_identity):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from source.dataset.dataset import BratsValidationDataset
    from source.core.inference import sliding_window_inference
    from source.benchmark_metrics import (canonical_mask, native_region_metrics, raw_geometry,
                                           restore_full_grid, avmur_region_dice)
    require(partition in ("selection80", "test"), "No other partition may be evaluated")
    require(set(evaluation_identity) == {"protocol_sha256", "checkpoint_sha256", "epoch"},
            "Evaluation must identify one protocol and one checkpoint")
    destination = Path(destination)
    require(not destination.exists(), "Refusing to overwrite evaluation results")
    ids = protocol["case_ids"][partition]
    mapped = protocol["mapped_ids"][partition]
    list_file = destination.parent / (partition + ".txt")
    contents = "\n".join(mapped) + "\n"
    if list_file.exists():
        require(list_file.read_text() == contents, "Evaluation case list changed")
    else:
        with list_file.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(contents)
    data_root = Path(protocol["data_root"])
    dataset = BratsValidationDataset(str(data_root), str(list_file), num_classes=4)
    # This is a separate OS process after training; its generator is never shared
    # with the native training/validation loaders.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(40))
    rows = []
    started = time.monotonic()
    bar = tqdm(total=len(ids) * 15, desc=destination.stem, ncols=120, mininterval=10)
    try:
        with torch.inference_mode():
            for inputs, targets, _, names in loader:
                pid = names[0]
                cid = protocol["processed_to_case"][pid]
                require(cid in ids, "Loader returned a case outside the declared partition")
                file_identity = case_file_identity(protocol, cid, pid)
                if cid not in geometry_cache:
                    geometry_cache[cid] = raw_geometry(
                        Path(protocol["raw_data_root"]) / cid,
                        data_root / "seg" / f"{pid}_seg.npy",
                        processed_vol_path=data_root / "vol" / f"{pid}_vol.npy")
                    geometry_cache[cid]["input_file_stats"] = file_identity
                geometry = geometry_cache[cid]
                require(geometry["input_file_stats"] == file_identity, "Case files changed between candidates")
                full_target = torch.from_numpy(np.asarray(geometry["target"])).to("cuda")
                inputs = inputs.to(device="cuda", dtype=torch.float32)
                native_target = targets.argmax(dim=1).to("cuda")
                for name, mask_value in CONFIGURATIONS:
                    mask = torch.tensor([canonical_mask(mask_value)], device="cuda", dtype=torch.bool)
                    prediction = sliding_window_inference(forward, inputs, 4, 96, overlap=0.5, masks=mask)
                    native = native_region_metrics(prediction, native_target)
                    restored = restore_full_grid(prediction[0], geometry["source_shape"],
                                                  geometry["crop_bounds"], geometry["padding"])
                    scores = avmur_region_dice(restored, full_target)
                    rows.append(dict(case_id=cid, processed_id=pid, configuration=name, mask_value=mask_value,
                                     **scores, **{"native_" + region: native[region] for region in REGIONS},
                                     ET_postpro=native["ET_postpro"]))
                    bar.update(1)
                require(case_file_identity(protocol, cid, pid) == file_identity, "Case files changed during inference")
                atomic_json(dict(state="running", partition=partition, completed_cases=len(rows) // 15,
                                 total_cases=len(ids), elapsed_seconds=time.monotonic() - started),
                            destination.with_suffix(".progress.json"))
                del full_target, inputs, targets, native_target, prediction, restored
    finally:
        bar.close()
        dataset.close()
    keyed = {(row["case_id"], row["mask_value"]): row for row in rows}
    require(len(keyed) == len(rows), "Duplicate inference rows")
    rows = [keyed[(cid, mask)] for cid in ids for _, mask in CONFIGURATIONS]
    result = dict(state="complete", partition=partition, protocol_id=PROTOCOL_ID,
                  **evaluation_identity,
                  scoring_domain="original_mri_grid", dice_epsilon=1e-6,
                  ET_postpro_domain="native_processed_grid_diagnostic_only",
                  summary=validate_rows(rows, ids), native_summary=native_summary(rows, ids), rows=rows,
                  geometry={cid: {key: value for key, value in geometry_cache[cid].items() if key != "target"}
                            for cid in ids}, elapsed_seconds=time.monotonic() - started)
    write_json_exclusive(result, destination)
    return result


def validate_evaluation_result(payload, protocol, partition, entry, protocol_digest):
    require(payload.get("state") == "complete" and payload.get("partition") == partition,
            "Incomplete or wrong-partition selection result")
    require(payload.get("protocol_id") == PROTOCOL_ID
            and payload.get("protocol_sha256") == protocol_digest
            and payload.get("checkpoint_sha256") == entry["sha256"]
            and payload.get("epoch") == entry["epoch"]
            and payload.get("scoring_domain") == "original_mri_grid"
            and payload.get("dice_epsilon") == 1e-6
            and payload.get("ET_postpro_domain") == "native_processed_grid_diagnostic_only",
            "Evaluation identity or scoring protocol changed")
    ids = protocol["case_ids"][partition]
    summary = validate_rows(payload["rows"], ids)
    require([(row["case_id"], row["mask_value"]) for row in payload["rows"]]
            == [(cid, mask) for cid in ids for _, mask in CONFIGURATIONS],
            "Common-grid rows must follow frozen case/configuration order")
    require(summary == payload["summary"], "Stored selection means differ from per-case rows")
    require(native_summary(payload["rows"], ids) == payload.get("native_summary"),
            "Stored native means differ from per-case rows")
    geometry = payload.get("geometry", {})
    require(set(geometry) == set(ids) and all(
        item.get("raw_headers_verified") is True and item.get("processed_volume_verified") is True
        for item in geometry.values()), "Missing or unverified full-grid geometry")
    return summary


def records_from_results(output, entries, protocol, *, protocol_digest):
    records = []
    for entry in entries:
        name = f"validation/epoch_{entry['epoch']:04d}.json"
        path = safe_child(output, name)
        payload = read_json(path)
        summary = validate_evaluation_result(payload, protocol, "selection80", entry, protocol_digest)
        records.append(dict(epoch=entry["epoch"], checkpoint_path=entry["path"], checkpoint_sha256=entry["sha256"],
                            result_path=name, result_sha256=sha256_file(path), **summary))
    return records


def validate_frozen_selection(output, directory, protocol, entries):
    """Recompute the entire validation ranking before ever constructing test data."""
    output = Path(output)
    frozen = read_json(output / "SELECTION_FROZEN.json")
    require(frozen.get("state") == "frozen" and frozen.get("test_data_loaded") is False
            and frozen.get("protocol_id") == PROTOCOL_ID and frozen.get("configuration_count") == 15
            and frozen.get("scoring_domain") == "original_mri_grid"
            and frozen.get("selection_metric") == "mean_raw45"
            and frozen.get("selection_cases") == 80 and frozen.get("test_used_for_selection") is False
            and frozen.get("protocol_sha256") == sha256_file(Path(directory) / "protocol.json")
            and frozen.get("capture_manifest_sha256") == sha256_file(Path(directory) / "candidates.json"),
            "Frozen selection identity changed")
    ranking = rank_records(records_from_results(output, entries, protocol,
                           protocol_digest=sha256_file(Path(directory) / "protocol.json")), protocol["expected_epochs"])
    require(frozen.get("ranking") == ranking and frozen.get("selected") == ranking[0],
            "Frozen winner differs from validation-only ranking")
    return frozen


def write_tables(result, output):
    for filename, summary in (("test_common_45.csv", result["summary"]),
                               ("test_native_reference.csv", result["native_summary"])):
        fields = (*REGIONS, "ET_postpro") if "native" in filename else REGIONS
        with (Path(output) / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["configuration", *[field + "_dice_percent" for field in fields]])
            for name, mask in CONFIGURATIONS:
                writer.writerow([name, *[format(100 * summary["by_configuration"][str(mask)][field], ".17g")
                                        for field in fields]])
            writer.writerow(["Mean", *[format(100 * summary[field], ".17g") for field in fields]])


def runtime_provenance():
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), seed=40, precision="FP16_forward_FP32_softmax_accumulation",
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
                slurm_job_id=os.environ.get("SLURM_JOB_ID"), created_utc=utc_now())


def run(mode, directory, output):
    import torch
    from source.config.setup_seed import setup_seed
    directory, output = Path(directory).resolve(), Path(output).resolve()
    protocol, entries, started = verify_capture(directory)
    digest = sha256_file(directory / "protocol.json")
    if mode == "verify":
        frozen = validate_frozen_selection(output, directory, protocol, entries)
        if (output / "TEST_COMPLETE.json").exists():
            complete = read_json(output / "TEST_COMPLETE.json")
            test_started = read_json(output / "TEST_STARTED.json")
            test = read_json(output / "test_selected.json")
            require(complete.get("state") == "complete" and complete.get("test_cases") == 251
                    and complete.get("tested_checkpoints") == 1 and complete.get("test_is_historical_reused") is True
                    and complete.get("protocol_sha256") == digest
                    and complete.get("selected_epoch") == frozen["selected"]["epoch"]
                    and complete["selection_sha256"] == sha256_file(output / "SELECTION_FROZEN.json")
                    and complete["test_result_sha256"] == sha256_file(output / "test_selected.json")
                    and complete["checkpoint_sha256"] == frozen["selected"]["checkpoint_sha256"]
                    and complete.get("test_started_sha256") == sha256_file(output / "TEST_STARTED.json")
                    and test_started.get("state") == "started"
                    and test_started.get("checkpoint_sha256") == complete["checkpoint_sha256"]
                    and test_started.get("selection_sha256") == complete["selection_sha256"],
                    "Test provenance changed")
            entry = next(item for item in entries if item["epoch"] == frozen["selected"]["epoch"])
            validate_evaluation_result(test, protocol, "test", entry, digest)
        return {"state": "verified", "selected_epoch": frozen["selected"]["epoch"]}
    require(mode in ("select", "test"), "Unknown benchmark phase")
    require(os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Inference requires one visible GPU inside an authorized Slurm allocation")
    if mode == "select":
        output.mkdir(parents=True, exist_ok=False)
        (output / "validation").mkdir()
        selected_entries = entries
    else:
        frozen = validate_frozen_selection(output, directory, protocol, entries)
        selected_entries = [entry for entry in entries if entry["epoch"] == frozen["selected"]["epoch"]]
        require(len(selected_entries) == 1, "Exactly one validation-selected checkpoint is required")
        write_json_exclusive(dict(state="started", selection_sha256=sha256_file(output / "SELECTION_FROZEN.json"),
                                  checkpoint_sha256=selected_entries[0]["sha256"], started_utc=utc_now()),
                             output / "TEST_STARTED.json")
    setup_seed(40)
    torch.set_num_threads(max(1, min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))))
    write_json_exclusive(runtime_provenance(), output / f"{mode}_runtime.json")
    model, forward = build_model(started["training_config"], directory / selected_entries[0]["path"])
    geometry_cache = {}
    for entry in selected_entries:
        load_candidate(model, directory, entry, digest, started["training_config"])
        identity = dict(protocol_sha256=digest, checkpoint_sha256=entry["sha256"], epoch=entry["epoch"])
        if mode == "select":
            result_path = output / "validation" / f"epoch_{entry['epoch']:04d}.json"
            evaluate_partition(forward, protocol, "selection80", result_path, geometry_cache,
                               evaluation_identity=identity)
        else:
            result = evaluate_partition(forward, protocol, "test", output / "test_selected.json", geometry_cache,
                                        evaluation_identity=identity)
    if mode == "select":
        ranking = rank_records(records_from_results(output, entries, protocol, protocol_digest=digest),
                               protocol["expected_epochs"])
        frozen = dict(state="frozen", protocol_id=PROTOCOL_ID, protocol_sha256=digest,
                      capture_manifest_sha256=sha256_file(directory / "candidates.json"),
                      frozen_utc=utc_now(), selection_cases=80, configuration_count=15,
                      scoring_domain="original_mri_grid", selection_metric="mean_raw45",
                      test_data_loaded=False, test_used_for_selection=False,
                      ranking=ranking, selected=ranking[0])
        write_json_exclusive(frozen, output / "SELECTION_FROZEN.json")
        return {"state": "selection_frozen", "selected_epoch": ranking[0]["epoch"]}
    validate_evaluation_result(result, protocol, "test", selected_entries[0], digest)
    write_tables(result, output)
    complete = dict(state="complete", completed_utc=utc_now(), selected_epoch=selected_entries[0]["epoch"],
                    checkpoint_sha256=selected_entries[0]["sha256"], test_cases=251, tested_checkpoints=1,
                    selection_sha256=sha256_file(output / "SELECTION_FROZEN.json"),
                    test_result_sha256=sha256_file(output / "test_selected.json"),
                    test_started_sha256=sha256_file(output / "TEST_STARTED.json"),
                    protocol_sha256=digest, test_is_historical_reused=True)
    write_json_exclusive(complete, output / "TEST_COMPLETE.json")
    return complete
