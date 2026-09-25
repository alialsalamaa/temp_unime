"""CPU-only provenance, selection/test-gate, and scoring integration checks."""
from contextlib import ExitStack
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from source import benchmark_runner as runner
from source.benchmark_protocol import CONFIGURATIONS, PROTOCOL_ID, REGIONS, rank_records, validate_rows


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def training_config():
    return dict(model_name="UniME", dataset_name="BRATS2023", split_type="Normal",
                num_classes=4, crop_size=96, batch_size=4, num_epochs=600,
                iter_per_epoch=250, use_ema=True, amp=True, bfloat16=False,
                base_lr=3e-4, min_lr=1e-6, warmup_lr=1e-5, warmup_ratio=.05,
                weight_decay=1e-4, layer_decay=.75, ema_decay=.999, train_ratio=None,
                uni_encoder_name="UniEncoder", seed=40, compile=True)


def case_rows(ids, score=.7):
    return [dict(case_id=cid, configuration=name, mask_value=mask,
                 WT=score, TC=score, ET=score, native_WT=.8, native_TC=.7,
                 native_ET=.6, ET_postpro=.99)
            for cid in ids for name, mask in CONFIGURATIONS]


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.capture = self.root / "capture"
        self.output = self.root / "output"
        self.capture.mkdir()
        self.output.mkdir()
        self.config = training_config()
        self.protocol = dict(protocol_id=PROTOCOL_ID,
                             array_audit={"arrays_checked": 2502, "mode": "all_headers_and_exact_payload_sizes"},
                             expected_epochs=[460, 470],
                             case_ids={"selection80": [f"val-{i:03d}" for i in range(80)],
                                       "test": [f"test-{i:03d}" for i in range(251)]})
        save_json(self.capture / "protocol.json", self.protocol)
        self.digest = runner.sha256_file(self.capture / "protocol.json")
        self.identity = dict(protocol_id=PROTOCOL_ID, protocol_sha256=self.digest,
                             model_variant="EMA", expected_epochs=self.protocol["expected_epochs"])
        self.entries = []
        for epoch in self.protocol["expected_epochs"]:
            path = self.capture / f"epoch_{epoch:04d}_ema.pth"
            path.write_bytes(f"synthetic weights for {epoch}".encode())
            entry = dict(epoch=epoch, path=path.name, sha256=runner.sha256_file(path), model_variant="EMA")
            self.entries.append(entry)
            save_json(path.with_suffix(".json"), entry)
        self.source_hashes = {"source/example.py": "a" * 64}
        pretrain_path = self.root / "pretrain.pth"
        pretrain_path.write_bytes(b"synthetic pretrained checkpoint identity")
        self.config["uni_encoder_checkpoint"] = str(pretrain_path)
        self.started = dict(self.identity, training_config=self.config, source_sha256=self.source_hashes,
                            pretrained_checkpoint=dict(path=str(pretrain_path), sha256=runner.sha256_file(pretrain_path)))
        self.manifest = dict(self.identity, entries=self.entries)
        self.complete = dict(self.identity, state="complete", completed_epochs=600,
                             automatic_test_performed=False, candidate_count=len(self.entries))
        self.write_capture()

    def write_capture(self):
        save_json(self.capture / "capture_started.json", self.started)
        save_json(self.capture / "candidates.json", self.manifest)
        self.complete["manifest_sha256"] = runner.sha256_file(self.capture / "candidates.json")
        save_json(self.capture / "TRAINING_COMPLETE.json", self.complete)

    def verify(self, current_sources=None):
        # Membership/source parsing is independently tested in protocol tests;
        # only those two outside boundaries are mocked, not hashes or sidecars.
        with patch.object(runner, "validate_protocol", side_effect=lambda value: value), \
                patch("source.benchmark_capture.source_fingerprints",
                      return_value=self.source_hashes if current_sources is None else current_sources):
            return runner.verify_capture(self.capture)

    def result(self, entry, partition="selection80", score=.7):
        ids = self.protocol["case_ids"][partition]
        rows = case_rows(ids, score)
        return dict(state="complete", partition=partition, protocol_id=PROTOCOL_ID,
                    protocol_sha256=self.digest, checkpoint_sha256=entry["sha256"], epoch=entry["epoch"],
                    scoring_domain="original_mri_grid", dice_epsilon=1e-6,
                    ET_postpro_domain="native_processed_grid_diagnostic_only",
                    summary=validate_rows(rows, ids), native_summary=runner.native_summary(rows, ids), rows=rows,
                    geometry={cid: dict(raw_headers_verified=True, processed_volume_verified=True)
                              for cid in ids})

    def write_validation(self):
        for index, entry in enumerate(self.entries):
            save_json(self.output / "validation" / f"epoch_{entry['epoch']:04d}.json",
                      self.result(entry, score=.6 + .1 * index))

    def freeze(self):
        self.write_validation()
        ranking = rank_records(runner.records_from_results(self.output, self.entries, self.protocol,
                                                           protocol_digest=self.digest),
                               self.protocol["expected_epochs"])
        frozen = dict(state="frozen", protocol_id=PROTOCOL_ID, protocol_sha256=self.digest,
                      capture_manifest_sha256=runner.sha256_file(self.capture / "candidates.json"),
                      selection_cases=80, configuration_count=15, scoring_domain="original_mri_grid",
                      selection_metric="mean_raw45", test_data_loaded=False,
                      test_used_for_selection=False, ranking=ranking, selected=ranking[0])
        save_json(self.output / "SELECTION_FROZEN.json", frozen)
        return frozen

    def cpu_run_context(self, stack):
        stack.enter_context(patch.object(runner, "verify_capture", return_value=(self.protocol, self.entries, self.started)))
        stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": "synthetic-no-submission"}))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(patch("source.config.setup_seed.setup_seed"))
        stack.enter_context(patch.object(torch, "set_num_threads"))
        stack.enter_context(patch.object(runner, "runtime_provenance", return_value={"test_fixture": True}))


class CaptureProvenanceTests(RunnerFixture):
    def test_complete_capture_validates_real_hashes_and_sidecars(self):
        protocol, entries, started = self.verify()
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(entries, self.entries)
        self.assertEqual(started["training_config"], self.config)

    def test_changed_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Code/dependencies changed"):
            self.verify({"source/example.py": "b" * 64})

    def test_pretrained_checkpoint_identity_is_not_silently_changed(self):
        Path(self.started["pretrained_checkpoint"]["path"]).write_bytes(b"different pretraining")
        with self.assertRaisesRegex(ValueError, "Pretrained checkpoint provenance changed"):
            self.verify()

    def test_incomplete_training_or_native_test_access_is_rejected(self):
        original = copy.deepcopy(self.complete)
        for changes in ({"state": "running"}, {"completed_epochs": 599}, {"automatic_test_performed": True}):
            self.complete = dict(original, **changes)
            self.write_capture()
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "incomplete or native testing"):
                self.verify()

    def test_identity_mismatch_or_manifest_only_audit_is_rejected(self):
        self.started["protocol_sha256"] = "0" * 64
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.verify()
        self.protocol["array_audit"]["arrays_checked"] = 0
        save_json(self.capture / "protocol.json", self.protocol)
        with self.assertRaisesRegex(ValueError, "Manifest-only"):
            self.verify()

    def test_weights_and_sidecar_mutation_are_rejected(self):
        path = self.capture / self.entries[0]["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "weights changed"):
            self.verify()
        path.write_bytes(original)
        save_json(path.with_suffix(".json"), dict(self.entries[0], epoch=999))
        with self.assertRaisesRegex(ValueError, "sidecar changed"):
            self.verify()

    def test_manifest_hash_and_candidate_schedule_are_enforced(self):
        self.manifest["entries"] = self.entries[:1]
        save_json(self.capture / "candidates.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.verify()
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "candidate schedule"):
            self.verify()

    def test_training_recipe_cannot_change(self):
        self.started["training_config"]["crop_size"] = 128
        self.write_capture()
        with self.assertRaisesRegex(ValueError, "training recipe: crop_size"):
            self.verify()

    def test_runtime_prefix_normalization_rejects_collisions(self):
        tensor = torch.ones(1)
        result = runner.normalized_state({"module._orig_mod.encoder.weight": tensor})
        self.assertIs(result["encoder.weight"], tensor)
        with self.assertRaisesRegex(ValueError, "Colliding checkpoint"):
            runner.normalized_state({"module.encoder.weight": tensor, "encoder.weight": tensor})

    def test_candidate_load_is_strict_and_checks_tensor_finiteness(self):
        entry = self.entries[0]
        payload = dict(epoch=entry["epoch"], model_variant="EMA", protocol_sha256=self.digest,
                       training_config=self.config, model_state_dict={"module.weight": torch.ones(1)})
        model = MagicMock()
        with patch.object(torch, "load", return_value=payload) as loader:
            runner.load_candidate(model, self.capture, entry, self.digest, self.config)
        self.assertTrue(loader.call_args.kwargs["weights_only"])
        self.assertTrue(model.load_state_dict.call_args.kwargs["strict"])
        self.assertEqual(list(model.load_state_dict.call_args.args[0]), ["weight"])
        model.eval.assert_called_once()
        payload["model_state_dict"]["module.weight"] = torch.tensor([float("nan")])
        with patch.object(torch, "load", return_value=payload), self.assertRaisesRegex(ValueError, "nonfinite"):
            runner.load_candidate(model, self.capture, entry, self.digest, self.config)


class SelectionGateTests(RunnerFixture):
    def test_every_candidate_requires_same80_all15_coverage(self):
        self.write_validation()
        records = runner.records_from_results(self.output, self.entries, self.protocol, protocol_digest=self.digest)
        self.assertEqual([record["case_count"] for record in records], [80, 80])
        self.assertEqual([record["row_count"] for record in records], [1200, 1200])
        path = self.output / "validation/epoch_0470.json"
        original = runner.read_json(path)
        for mutate in (lambda rows: rows.pop(), lambda rows: rows[0].update(case_id="test-000")):
            payload = copy.deepcopy(original)
            mutate(payload["rows"])
            save_json(path, payload)
            with self.assertRaisesRegex(ValueError, "coverage"):
                runner.records_from_results(self.output, self.entries, self.protocol, protocol_digest=self.digest)

    def test_selection_results_reject_wrong_domain_epsilon_and_identity(self):
        self.write_validation()
        path = self.output / "validation/epoch_0460.json"
        original = runner.read_json(path)
        for key, value in (("scoring_domain", "native_processed_grid"), ("dice_epsilon", 1e-8),
                           ("protocol_id", "unrelated"), ("protocol_sha256", "0" * 64),
                           ("checkpoint_sha256", "0" * 64), ("epoch", 470)):
            save_json(path, dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.records_from_results(self.output, self.entries, self.protocol, protocol_digest=self.digest)

    def test_unverified_geometry_and_native_summary_tampering_are_rejected(self):
        self.write_validation()
        path = self.output / "validation/epoch_0460.json"
        original = runner.read_json(path)
        for kind in ("geometry", "native_summary"):
            payload = copy.deepcopy(original)
            if kind == "geometry":
                payload["geometry"][self.protocol["case_ids"]["selection80"][0]]["processed_volume_verified"] = False
            else:
                payload["native_summary"]["WT"] = 0.
            save_json(path, payload)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.records_from_results(self.output, self.entries, self.protocol, protocol_digest=self.digest)

    def test_common_rows_must_retain_exact_frozen_case_and_configuration_order(self):
        self.write_validation()
        path = self.output / "validation/epoch_0460.json"
        payload = runner.read_json(path)
        payload["rows"] = list(reversed(payload["rows"]))
        payload["summary"] = validate_rows(payload["rows"], self.protocol["case_ids"]["selection80"])
        payload["native_summary"] = runner.native_summary(payload["rows"], self.protocol["case_ids"]["selection80"])
        save_json(path, payload)
        with self.assertRaisesRegex(ValueError, "frozen case/configuration order"):
            runner.records_from_results(self.output, self.entries, self.protocol, protocol_digest=self.digest)

    def test_frozen_winner_recomputed_from_validation_only(self):
        frozen = self.freeze()
        actual = runner.validate_frozen_selection(self.output, self.capture, self.protocol, self.entries)
        self.assertEqual(actual["selected"]["epoch"], 470)
        frozen["selected"] = frozen["ranking"][1]
        save_json(self.output / "SELECTION_FROZEN.json", frozen)
        with self.assertRaisesRegex(ValueError, "winner differs"):
            runner.validate_frozen_selection(self.output, self.capture, self.protocol, self.entries)

    def test_frozen_provenance_fields_cannot_be_relabeled(self):
        original = self.freeze()
        for key, value in (("protocol_id", "unrelated"), ("configuration_count", 14),
                           ("scoring_domain", "native_processed_grid"),
                           ("selection_metric", "ET_postpro"), ("test_data_loaded", True)):
            save_json(self.output / "SELECTION_FROZEN.json", dict(original, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_frozen_selection(self.output, self.capture, self.protocol, self.entries)

    def test_test_cannot_load_model_or_data_before_all_validation_is_complete(self):
        self.freeze()
        (self.output / "validation/epoch_0470.json").unlink()
        with ExitStack() as stack:
            self.cpu_run_context(stack)
            build = stack.enter_context(patch.object(runner, "build_model"))
            evaluate = stack.enter_context(patch.object(runner, "evaluate_partition"))
            with self.assertRaises((ValueError, FileNotFoundError)):
                runner.run("test", self.capture, self.output)
            build.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse((self.output / "TEST_STARTED.json").exists())

    def test_select_phase_evaluates_every_candidate_only_on_selection80(self):
        destination = self.root / "fresh_selection"
        seen = []

        def evaluate(_forward, protocol, partition, output, _cache, *, evaluation_identity):
            self.assertEqual(protocol["case_ids"]["selection80"], self.protocol["case_ids"]["selection80"])
            self.assertEqual(partition, "selection80")
            entry = next(item for item in self.entries if item["epoch"] == evaluation_identity["epoch"])
            self.assertEqual(evaluation_identity["checkpoint_sha256"], entry["sha256"])
            self.assertEqual(evaluation_identity["protocol_sha256"], self.digest)
            seen.append(entry["epoch"])
            result = self.result(entry, score=.6 if entry["epoch"] == 460 else .7)
            save_json(output, result)
            return result

        with ExitStack() as stack:
            self.cpu_run_context(stack)
            stack.enter_context(patch.object(runner, "build_model", return_value=(MagicMock(), MagicMock())))
            load = stack.enter_context(patch.object(runner, "load_candidate"))
            stack.enter_context(patch.object(runner, "evaluate_partition", side_effect=evaluate))
            result = runner.run("select", self.capture, destination)
        self.assertEqual(seen, [460, 470])
        self.assertEqual(load.call_count, 2)
        self.assertEqual(result["selected_epoch"], 470)
        frozen = runner.read_json(destination / "SELECTION_FROZEN.json")
        self.assertIs(frozen["test_data_loaded"], False)
        self.assertFalse((destination / "TEST_STARTED.json").exists())

    def test_test_phase_evaluates_exactly_one_frozen_winner(self):
        self.freeze()
        observed = []

        def evaluate(_forward, _protocol, partition, output, _cache, *, evaluation_identity):
            observed.append((partition, evaluation_identity["epoch"]))
            result = self.result(self.entries[1], partition="test")
            save_json(output, result)
            return result

        with ExitStack() as stack:
            self.cpu_run_context(stack)
            stack.enter_context(patch.object(runner, "build_model", return_value=(MagicMock(), MagicMock())))
            load = stack.enter_context(patch.object(runner, "load_candidate"))
            stack.enter_context(patch.object(runner, "evaluate_partition", side_effect=evaluate))
            result = runner.run("test", self.capture, self.output)
        self.assertEqual(observed, [("test", 470)])
        self.assertEqual(load.call_count, 1)
        self.assertEqual(result["tested_checkpoints"], 1)
        self.assertEqual(result["test_cases"], 251)
        self.assertTrue(result["test_is_historical_reused"])

    def test_complete_test_verification_binds_the_test_start_receipt(self):
        frozen = self.freeze()
        selected = self.entries[1]
        test_path = self.output / "test_selected.json"
        save_json(test_path, self.result(selected, partition="test"))
        selection_digest = runner.sha256_file(self.output / "SELECTION_FROZEN.json")
        receipt_path = self.output / "TEST_STARTED.json"
        receipt = dict(state="started", selection_sha256=selection_digest,
                       checkpoint_sha256=selected["sha256"])
        save_json(receipt_path, receipt)
        complete = dict(state="complete", selected_epoch=frozen["selected"]["epoch"],
                        checkpoint_sha256=selected["sha256"], test_cases=251, tested_checkpoints=1,
                        selection_sha256=selection_digest, test_result_sha256=runner.sha256_file(test_path),
                        test_started_sha256=runner.sha256_file(receipt_path),
                        protocol_sha256=self.digest, test_is_historical_reused=True)
        save_json(self.output / "TEST_COMPLETE.json", complete)
        with patch.object(runner, "verify_capture", return_value=(self.protocol, self.entries, self.started)), \
                patch.object(runner, "build_model") as build, patch.object(runner, "evaluate_partition") as evaluate:
            self.assertEqual(runner.run("verify", self.capture, self.output),
                             {"state": "verified", "selected_epoch": 470})
            save_json(receipt_path, dict(receipt, state="changed"))
            with self.assertRaisesRegex(ValueError, "Test provenance changed"):
                runner.run("verify", self.capture, self.output)
            build.assert_not_called()
            evaluate.assert_not_called()


class ReportingTests(RunnerFixture):
    def test_native_summary_restores_sorted_processed_patient_reduction_order(self):
        ids = [f"case-{index:03d}" for index in range(80)]
        rows = case_rows(ids)
        for row in rows:
            index = ids.index(row["case_id"])
            row["processed_id"] = f"BraTS2023_{79-index:05d}"
            row["native_WT"] = float(np.float32(index / 80))
            row["native_TC"] = float(np.float32((index % 17) / 17))
            row["native_ET"] = float(np.float32((index % 23) / 23))
        expected_order = sorted(rows, key=lambda row: row["processed_id"])
        fields = ("native_WT", "native_TC", "native_ET", "ET_postpro")
        first_mask = CONFIGURATIONS[0][1]
        expected_values = [[row[field] for field in fields] for row in expected_order
                           if row["mask_value"] == first_mask]
        expected_mean = np.asarray(expected_values, dtype=np.float32).mean(axis=0)
        with patch.object(np, "asarray", wraps=np.asarray) as arrays:
            result = runner.native_summary(rows, ids)
        # Check actual reduction input, not just a rounded score that could agree
        # accidentally despite a different float32 summation order.
        self.assertEqual(arrays.call_args_list[0].args[0], expected_values)
        self.assertEqual([result["by_configuration"][str(first_mask)][field]
                          for field in (*REGIONS, "ET_postpro")], expected_mean.tolist())

    def test_native_summary_keeps_float32_patient_then_float64_configuration_means(self):
        ids = [f"case-{index}" for index in range(80)]
        rows = case_rows(ids)
        for index, row in enumerate(rows):
            row.update(native_WT=float(np.float32((index % 71) / 71)),
                       native_TC=float(np.float32((index % 43) / 43)),
                       native_ET=float(np.float32((index % 29) / 29)))
        actual = runner.native_summary(rows, ids)
        configurations = []
        for _, mask in CONFIGURATIONS:
            native_cases = np.asarray([[row["native_WT"], row["native_TC"], row["native_ET"], row["ET_postpro"]]
                                       for row in rows if row["mask_value"] == mask], dtype=np.float32)
            expected = native_cases.mean(axis=0)
            configurations.append(expected)
            for index, field in enumerate((*REGIONS, "ET_postpro")):
                self.assertEqual(actual["by_configuration"][str(mask)][field], float(expected[index]))
        expected_overall = np.asarray(configurations, dtype=np.float64).mean(axis=0)
        self.assertEqual([actual[field] for field in (*REGIONS, "ET_postpro")], expected_overall.tolist())

    def test_primary_table_contains45_scores_without_postprocessed_et(self):
        result = self.result(self.entries[0], partition="test")
        runner.write_tables(result, self.output)
        with (self.output / "test_common_45.csv").open(newline="") as stream:
            primary = list(csv.reader(stream))
        with (self.output / "test_native_reference.csv").open(newline="") as stream:
            native = list(csv.reader(stream))
        self.assertEqual(primary[0], ["configuration", "WT_dice_percent", "TC_dice_percent", "ET_dice_percent"])
        self.assertEqual(len(primary), 17)  # Header, 15 combinations, mean.
        self.assertEqual(sum(len(row) - 1 for row in primary[1:16]), 45)
        self.assertEqual([row[0] for row in primary[1:16]], [name for name, _ in CONFIGURATIONS])
        self.assertEqual(primary[-1][0], "Mean")
        self.assertIn("ET_postpro_dice_percent", native[0])
        with self.assertRaises(FileExistsError):
            runner.write_tables(result, self.output)


class FullGridIntegrationTests(unittest.TestCase):
    def test_case_identity_covers_all_seven_inputs_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cid, pid = "BraTS-GLI-00001-000", "BraTS2023_00001"
            raw = root / "raw" / cid
            raw.mkdir(parents=True)
            (root / "data/vol").mkdir(parents=True)
            (root / "data/seg").mkdir()
            paths = [raw / f"{cid}-{suffix}.nii.gz" for suffix in ("t2f", "t1c", "t1n", "t2w", "seg")]
            paths += [root / "data/vol" / f"{pid}_vol.npy", root / "data/seg" / f"{pid}_seg.npy"]
            for path in paths:
                path.write_bytes(b"synthetic file identity only")
            protocol = dict(raw_data_root=str(root / "raw"), data_root=str(root / "data"))
            original = runner.case_file_identity(protocol, cid, pid)
            self.assertEqual(set(original), {str(path) for path in paths})
            self.assertTrue(all(set(item) == {"size", "mtime_ns"} for item in original.values()))
            paths[0].write_bytes(b"changed and longer synthetic file identity only")
            self.assertNotEqual(runner.case_file_identity(protocol, cid, pid), original)

    def test_native_case_loader_inference_and_full_grid_scoring_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vol").mkdir()
            (root / "seg").mkdir()
            # Frozen protocol order is deliberately opposite the native dataset's
            # lexical processed-ID sorting order.
            pid, cid = "BraTS2023_00002", "BraTS-GLI-00002-000"
            other_pid, other_cid = "BraTS2023_00001", "BraTS-GLI-00001-000"
            volume = np.ones((2, 2, 2, 4), dtype=np.float32)
            native_target = np.zeros((2, 2, 2), dtype=np.uint8)
            native_target[0, 0, 0] = 3
            for processed_id in (pid, other_pid):
                np.save(root / "vol" / f"{processed_id}_vol.npy", volume)
                np.save(root / "seg" / f"{processed_id}_seg.npy", native_target)
            full_target = np.zeros((4, 4, 4), dtype=np.uint8)
            full_target[1:3, 1:3, 1:3] = native_target
            full_target[0, 0, 0] = 3  # Native crop excludes this genuine target voxel.
            geometry = dict(source_shape=(4, 4, 4), crop_bounds=((1, 3),) * 3,
                            padding=((0, 0),) * 3, target=full_target,
                            raw_headers_verified=True, processed_volume_verified=True,
                            outside_crop_tumor_voxels=1)
            protocol = dict(data_root=str(root), raw_data_root=str(root / "raw"),
                            case_ids={"selection80": [cid, other_cid]},
                            mapped_ids={"selection80": [pid, other_pid]},
                            processed_to_case={pid: cid, other_pid: other_cid})
            identity = dict(protocol_sha256="a" * 64, checkpoint_sha256="b" * 64, epoch=460)
            original_to, original_tensor = torch.Tensor.to, torch.tensor
            masks_seen = []

            def to_cpu(tensor, *args, **kwargs):
                args = tuple("cpu" if isinstance(arg, str) and arg == "cuda" else arg for arg in args)
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return original_to(tensor, *args, **kwargs)

            def cpu_tensor(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return original_tensor(*args, **kwargs)

            def infer(_model, inputs, classes, crop_size, *, overlap, masks):
                self.assertEqual(tuple(inputs.shape), (1, 4, 2, 2, 2))
                self.assertEqual(inputs.dtype, torch.float32)
                self.assertEqual((classes, crop_size, overlap), (4, 96, .5))
                masks_seen.append(tuple(masks[0].tolist()))
                return torch.from_numpy(native_target.copy()).unsqueeze(0).long()

            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.Tensor, "to", to_cpu))
                stack.enter_context(patch.object(torch, "tensor", cpu_tensor))
                files = stack.enter_context(patch.object(runner, "case_file_identity",
                                            return_value={"synthetic": {"size": 1, "mtime_ns": 1}}))
                raw = stack.enter_context(patch("source.benchmark_metrics.raw_geometry",
                                               side_effect=lambda *args, **kwargs: copy.deepcopy(geometry)))
                inference = stack.enter_context(patch("source.core.inference.sliding_window_inference", side_effect=infer))
                stack.enter_context(patch("tqdm.tqdm"))
                cache = {}
                result = runner.evaluate_partition(MagicMock(), protocol, "selection80", root / "result.json", cache,
                                                   evaluation_identity=identity)
                self.assertEqual(raw.call_count, 2)
                self.assertEqual([call.args[0].name for call in raw.call_args_list], [other_cid, cid])
                self.assertEqual(raw.call_args.kwargs["processed_vol_path"], root / "vol" / f"{pid}_vol.npy")
                files.return_value = {"synthetic": {"size": 1, "mtime_ns": 2}}
                with self.assertRaisesRegex(ValueError, "Case files changed between candidates"):
                    runner.evaluate_partition(MagicMock(), protocol, "selection80", root / "second.json", cache,
                                              evaluation_identity=identity)
                self.assertEqual(inference.call_count, 30)
                self.assertFalse((root / "second.json").exists())
            self.assertEqual(len(result["rows"]), 30)
            self.assertEqual([(row["case_id"], row["mask_value"]) for row in result["rows"]],
                             [(case, mask) for case in (cid, other_cid) for _, mask in CONFIGURATIONS])
            self.assertEqual(len(set(masks_seen)), 15)
            self.assertEqual(result["native_summary"]["ET"], 1.)
            self.assertLess(result["summary"]["ET"], 1.)
            self.assertAlmostEqual(result["summary"]["ET"], 2 / 3, places=6)
            self.assertEqual(result["scoring_domain"], "original_mri_grid")
            self.assertEqual(result["checkpoint_sha256"], identity["checkpoint_sha256"])
            self.assertNotIn("target", result["geometry"][cid])
            self.assertTrue((root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
