"""Pure stdlib protocol checks; fixtures contain no MRI data or model weights."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("standalone_benchmark_protocol", PACKAGE / "source/benchmark_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def synthetic_split():
    cases = [f"BraTS-GLI-{index:05d}-000" for index in range(1251)]
    return {"train": cases[:875], "val": cases[875:1000], "test": cases[1000:]}


def partition_for(split):
    return {"tune_cases": split["val"][:80], "lock_cases": split["val"][80:], "validation_cases": 125}


class MembershipTests(unittest.TestCase):
    def test_exact_split_and_partition(self):
        split = synthetic_split()
        self.assertEqual(protocol.validate_split(split), split)
        subsets = protocol.validate_partition(partition_for(split), split)
        self.assertEqual([len(subsets[key]) for key in ("selection80", "development45")], [80, 45])

    def test_duplicates_wrong_counts_and_patient_variants_rejected(self):
        for mode in ("duplicate", "missing", "cross_split", "patient_variant", "malicious_id"):
            split = synthetic_split()
            if mode == "duplicate":
                split["train"][1] = split["train"][0]
            elif mode == "missing":
                split["val"].pop()
            elif mode == "cross_split":
                split["test"][0] = split["train"][0]
            elif mode == "patient_variant":
                split["test"][0] = split["train"][0][:-3] + "001"
            else:
                split["test"][0] = "../../unexpected"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                protocol.validate_split(split)

    def test_distinct_scans_of_same_patient_allowed_within_partition(self):
        split = synthetic_split()
        split["train"][1] = split["train"][0][:-3] + "001"
        split["val"][1] = split["val"][0][:-3] + "001"
        self.assertEqual(protocol.validate_split(split), split)
        self.assertEqual(protocol.validate_partition(partition_for(split), split)["selection80"], split["val"][:80])
        mapping = {f"BraTS2023_{index:05d}": cid for index, cid in enumerate(sum(split.values(), []))}
        self.assertEqual(len(protocol.validate_mapping(mapping, split)), 1251)

    def test_same_patient_cannot_cross_selection_development(self):
        split = synthetic_split()
        split["val"][80] = split["val"][0][:-3] + "001"
        protocol.validate_split(split)
        with self.assertRaises(ValueError):
            protocol.validate_partition(partition_for(split), split)

    def test_selection_cannot_contain_test_case_or_duplicate_lock_case(self):
        split = synthetic_split()
        for replacement in (split["test"][0], split["val"][0]):
            partition = partition_for(split)
            partition["lock_cases"][0] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                protocol.validate_partition(partition, split)

    def test_native_mapping_and_path_ids(self):
        split = synthetic_split()
        mapping = {f"BraTS2023_{index:05d}": cid
                   for index, cid in enumerate(sum(split.values(), []))}
        inverse = protocol.validate_mapping(mapping, split)
        self.assertEqual(inverse[split["val"][0]], "BraTS2023_00875")
        duplicate = dict(mapping)
        duplicate["BraTS2023_00001"] = duplicate["BraTS2023_00000"]
        with self.assertRaises(ValueError):
            protocol.validate_mapping(duplicate, split)
        unsafe = dict(mapping)
        unsafe["../escape"] = unsafe.pop("BraTS2023_00000")
        with self.assertRaises(ValueError):
            protocol.validate_mapping(unsafe, split)

    def test_canonical_to_native_channel_masks(self):
        for value, expected in ((8, [True, False, False, False]), (2, [False, True, False, False]),
                                (1, [False, False, True, False]), (4, [False, False, False, True]),
                                (15, [True, True, True, True])):
            self.assertEqual(protocol.native_mask(value), expected)
        for value in (0, 16, True, "8"):
            with self.assertRaises(ValueError):
                protocol.native_mask(value)


class RowTests(unittest.TestCase):
    def rows(self, count=80):
        ids = [f"case-{index}" for index in range(count)]
        rows = [{"case_id": cid, "mask_value": mask, "WT": .9, "TC": .8, "ET": .7, "ET_postpro": 0.0}
                for cid in ids for mask in range(1, 16)]
        return ids, rows

    def test_complete_raw45_excludes_postprocessed_et(self):
        ids, rows = self.rows()
        result = protocol.validate_rows(rows, ids)
        self.assertEqual(result["row_count"], 1200)
        self.assertEqual(result["case_count"], 80)
        self.assertEqual(len(result["by_configuration"]), 15)
        self.assertAlmostEqual(result["mean_raw45"], .8)
        for row in rows:
            row["ET_postpro"] = 1.0
        changed = protocol.validate_rows(rows, ids)
        self.assertEqual(changed["mean_raw45"], result["mean_raw45"])
        self.assertEqual(changed["ET_postpro"], 1.0)

    def test_full_test_coverage(self):
        ids, rows = self.rows(251)
        result = protocol.validate_rows(rows, ids)
        self.assertEqual(result["row_count"], 3765)
        self.assertEqual(result["case_count"], 251)

    def test_exact_v11_supplied_order_python_sum_not_fsum(self):
        ids, rows = self.rows()
        # V11 emits case-first then CONFIGURATIONS order. Its summarize_rows
        # deliberately uses sum of supplied rows, not fsum or rounded means.
        by_key = {(row["case_id"], row["mask_value"]): row for row in rows}
        rows = [by_key[(cid, mask)] for cid in ids for _, mask in protocol.CONFIGURATIONS]
        for index, row in enumerate(rows):
            row.update(WT=1.0 if index == 0 else 1e-16, TC=.123456789012345,
                       ET=.987654321098765, ET_postpro=.4,
                       native_WT=.01, native_TC=.02, native_ET=.03)
        with patch.object(protocol.math, "fsum", side_effect=AssertionError("V11 uses built-in sum")):
            result = protocol.validate_rows(rows, ids)
        expected_regions = {region: sum(row[region] for row in rows) / len(rows)
                            for region in protocol.REGIONS}
        for region in protocol.REGIONS:
            self.assertEqual(result[region], expected_regions[region])
        self.assertEqual(result["mean_raw45"], sum(expected_regions.values()) / 3)
        for _, mask in protocol.CONFIGURATIONS:
            selected = [row for row in rows if row["mask_value"] == mask]
            for region in protocol.REGIONS:
                self.assertEqual(result["by_configuration"][str(mask)][region],
                                 sum(row[region] for row in selected) / len(selected))
        for row in rows:
            row.update(native_WT=.91, native_TC=.92, native_ET=.93, ET_postpro=.99)
        self.assertEqual(protocol.validate_rows(rows, ids)["mean_raw45"], result["mean_raw45"])

    def test_missing_duplicate_wrong_masks_and_unexpected_patient(self):
        ids, rows = self.rows()
        for trial in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                protocol.validate_rows(trial, ids)
        for key, value in (("mask_value", 0), ("mask_value", 16), ("mask_value", True),
                           ("case_id", "unexpected-test-patient")):
            trial = copy.deepcopy(rows)
            trial[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                protocol.validate_rows(trial, ids)
        with self.assertRaises(ValueError):
            protocol.validate_rows(rows, ids + [ids[0]])

    def test_invalid_values_including_diagnostic(self):
        ids, rows = self.rows()
        for key in protocol.SCORES:
            for value in (float("nan"), float("inf"), -.01, 1.01, True, None):
                trial = copy.deepcopy(rows)
                trial[0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    protocol.validate_rows(trial, ids)


class RankingTests(unittest.TestCase):
    def records(self):
        return [{"epoch": epoch, "case_count": 80, "row_count": 1200, "mean_raw45": .5}
                for epoch in protocol.EXPECTED_EPOCHS]

    def test_native24_epochs_not_new20_epoch_schedule(self):
        expected = sorted(set(range(460, 601, 10)) | set(range(591, 601)))
        self.assertEqual(protocol.EXPECTED_EPOCHS, expected)
        self.assertEqual(len(expected), 24)

    def test_full_precision_and_earliest_tie(self):
        rows = self.records()
        rows[0]["mean_raw45"] = .8
        rows[1]["mean_raw45"] = .800000000001
        rows[2]["mean_raw45"] = .8
        ranked = protocol.rank_records(list(reversed(rows)), protocol.EXPECTED_EPOCHS)
        self.assertEqual([row["epoch"] for row in ranked[:3]], [470, 460, 480])

    def test_missing_duplicate_invalid_score_or_population(self):
        rows = self.records()
        for trial in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                protocol.rank_records(trial, protocol.EXPECTED_EPOCHS)
        for key, value in (("case_count", 125), ("row_count", 1199), ("epoch", 20),
                           ("mean_raw45", float("nan")), ("mean_raw45", True)):
            trial = copy.deepcopy(rows)
            trial[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.rank_records(trial)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unime_protocol_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "BRATS2023"
        self.data.mkdir()
        self.raw = self.root / "original_MRI"
        self.raw.mkdir()
        self.split = synthetic_split()
        self.partition = partition_for(self.split)
        self.split_path = self.root / "split.json"
        self.partition_path = self.root / "partition.json"
        protocol.write_json_exclusive(self.split, self.split_path)
        protocol.write_json_exclusive(self.partition, self.partition_path)
        self.mapping = {f"BraTS2023_{index:05d}": cid for index, cid in enumerate(sum(self.split.values(), []))}
        protocol.write_json_exclusive(self.mapping, self.data / "mapping.json")
        inverse = {cid: pid for pid, cid in self.mapping.items()}
        for name, ids in self.split.items():
            (self.data / f"{name}.txt").write_text("\n".join(inverse[cid] for cid in ids) + "\n", encoding="utf-8")
        # Synthetic fixtures exercise exact-hash enforcement without changing production constants.
        for name, path in (("SPLIT_SHA256", self.split_path), ("PARTITION_SHA256", self.partition_path)):
            patcher = patch.object(protocol, name, protocol.sha256_file(path))
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self):
        return protocol.prepare_protocol(self.split_path, self.partition_path, self.data,
                                         raw_data_root=self.raw, verify_arrays=False)

    def test_manifest_fixture_needs_no_npy_and_provenance_is_explicit(self):
        result = self.prepare()
        protocol.validate_protocol(result)
        self.assertEqual(result["case_ids"]["selection80"], self.partition["tune_cases"])
        self.assertEqual(result["array_audit"]["arrays_checked"], 0)
        self.assertEqual(result["scoring_domain"], "original_mri_grid")
        self.assertEqual(result["dice_epsilon"], 1e-6)
        self.assertTrue(result["common_full_grid_scoring_approved"])
        self.assertEqual(result["raw_data_root"], str(self.raw.resolve()))
        self.assertEqual(result["native_diagnostic_grid"], "native_processed_grid")
        self.assertEqual(result["native_dice_epsilon"], 1e-8)
        self.assertEqual(result["native_score_fields"], ["native_WT", "native_TC", "native_ET", "ET_postpro"])
        self.assertEqual(result["ET_postpro_diagnostic_grid"], "native_processed_grid")
        self.assertNotIn("ET_postpro", result["selection_score_fields"])
        self.assertEqual(result["native_mapping_provenance"]["channel_order"], ["FLAIR", "T1c", "T1", "T2"])
        self.assertEqual(result["native_mapping_provenance"]["label_mapping"]["ET"], 3)

    def test_array_audit_is_required_by_default(self):
        with self.assertRaises(ValueError):
            protocol.prepare_protocol(self.split_path, self.partition_path, self.data, raw_data_root=self.raw)

    def test_original_grid_requires_existing_raw_directory(self):
        for raw in (None, self.root / "missing", self.split_path):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                protocol.prepare_protocol(self.split_path, self.partition_path, self.data,
                                          raw_data_root=raw, verify_arrays=False)
        result = self.prepare()
        result["raw_data_root"] = str(self.root / "missing")
        with self.assertRaises(ValueError):
            protocol.validate_protocol(result)

    def test_primary_and_native_diagnostic_domains_cannot_be_silently_swapped(self):
        result = self.prepare()
        for field, changed in (("scoring_domain", "native_processed_grid"), ("dice_epsilon", 1e-8),
                               ("common_full_grid_scoring_approved", False),
                               ("native_diagnostic_grid", "original_mri_grid"),
                               ("native_dice_epsilon", 1e-6),
                               ("ET_postpro_diagnostic_grid", "original_mri_grid"),
                               ("raw_data_root", "relative/path")):
            trial = copy.deepcopy(result)
            trial[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_protocol(trial, verify_input_files=False)

    def test_reordered_manifest_or_changed_original_partition_rejected(self):
        path = self.data / "val.txt"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(reversed(lines)) + "\n")
        with self.assertRaises(ValueError):
            self.prepare()
        path.write_text("\n".join(lines) + "\n")
        self.partition_path.write_text(self.partition_path.read_text() + " ")
        with self.assertRaises(ValueError):
            self.prepare()

    def test_frozen_protocol_not_replaced_and_input_change_detected(self):
        result = self.prepare()
        target = self.root / "protocol.json"
        protocol.write_json_exclusive(result, target)
        before = protocol.sha256_file(target)
        with self.assertRaises(FileExistsError):
            protocol.write_json_exclusive({"invalid": True}, target)
        (self.data / "test.txt").write_text("invalid\n")
        with self.assertRaises(ValueError):
            protocol.validate_protocol(result)
        self.assertEqual(protocol.sha256_file(target), before)

    def test_embedded_selection_swap_is_not_hidden_by_correct_file_hashes(self):
        result = self.prepare()
        selection = result["case_ids"]["selection80"]
        development = result["case_ids"]["development45"]
        selection[0], development[0] = development[0], selection[0]
        inverse = {cid: pid for pid, cid in self.mapping.items()}
        for name in ("selection80", "development45"):
            result["mapped_ids"][name] = [inverse[cid] for cid in result["case_ids"][name]]
        with self.assertRaises(ValueError):
            protocol.validate_protocol(result)

    def test_invalid_json_duplicate_keys_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"train":[],"train":[]}', encoding="utf-8")
        with self.assertRaises(ValueError):
            protocol.read_json(path)

    def test_atomic_update_is_explicit_and_rejects_nan(self):
        path = self.root / "progress.json"
        protocol.atomic_json({"count": 1}, path)
        protocol.atomic_json({"count": 2}, path)
        self.assertEqual(protocol.read_json(path), {"count": 2})
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            protocol.atomic_json({"bad": float("nan")}, path)
        self.assertEqual(path.read_bytes(), before)


class HeaderTests(unittest.TestCase):
    def test_official_prior_bounds_are_applied_to_training_arrays(self):
        self.assertEqual(protocol.TRAINING_PRIOR_SHAPE, (160, 180, 210))
        # No image payload is allocated. Header and file checks are mocked;
        # the production array-audit branch itself must reject each oversized axis.
        with tempfile.TemporaryDirectory(prefix="unime_prior_header_") as folder:
            root = Path(folder)
            for name in ("vol", "seg"):
                (root / name).mkdir()
                (root / name / f"case_{name}.npy").touch()
            mapped = {"train": ["case"], "val": [], "test": []}
            for shape in ((161, 180, 210), (160, 181, 210), (160, 180, 211)):
                d, h, w = shape
                headers = [{"shape": (h, w, d, 4), "descr": "<f4"},
                           {"shape": (h, w, d), "descr": "|u1"}]
                with self.subTest(shape=shape), patch.object(protocol, "read_npy_header", side_effect=headers):
                    with self.assertRaisesRegex(ValueError, "exceeds learned prior"):
                        protocol.audit_array_headers(root, mapped)

    def test_header_only_validation_and_truncation(self):
        with tempfile.TemporaryDirectory(prefix="unime_npy_header_") as folder:
            path = Path(folder) / "synthetic.npy"
            header = repr({"descr": "<f4", "fortran_order": False, "shape": (2, 2)}).encode("latin1")
            header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
            data = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 16
            path.write_bytes(data)
            self.assertEqual(protocol.read_npy_header(path)["shape"], (2, 2))
            path.write_bytes(data[:-1])
            with self.assertRaises(ValueError):
                protocol.read_npy_header(path)


if __name__ == "__main__":
    unittest.main()
