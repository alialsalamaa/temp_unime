"""CPU-only capture contract tests; no training, dataset reads, CUDA or Slurm."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Dict, Iterator
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unime_benchmark_capture_test", ROOT / "source/benchmark_capture.py")
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def extract_train_definitions(names):
    tree = ast.parse((ROOT / "source/train.py").read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"torch": torch, "Dict": Dict, "Iterator": Iterator, "contextmanager": contextmanager}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "source/train.py", "exec"), namespace)
    return namespace


def native_epochs():
    rule = extract_train_definitions({"_valid_strategy"})["_valid_strategy"]
    return [epoch for epoch in range(1, 601) if rule(epoch, 600)]


class Arguments:
    def __init__(self, protocol):
        self.benchmark_protocol = str(protocol)
        self.model_name = "UniME"
        self.uni_encoder_name = "UniEncoder"
        self.use_ema = True
        self.num_epochs = 600
        self.seed = 40

    def to_dict(self):
        return vars(self).copy()


class BenchmarkCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol_path = self.root / "protocol.json"
        self.protocol = {"protocol_id": capture.PROTOCOL_ID, "expected_epochs": native_epochs(),
                         "array_audit": {"mode": "all_headers_and_exact_payload_sizes", "arrays_checked": 2502}}
        self.write_protocol()
        self.args = Arguments(self.protocol_path)
        self.pretrained = self.root / "pretrained.pth"
        self.pretrained.write_bytes(b"synthetic checkpoint bytes; no model loading in capture")
        self.args.uni_encoder_checkpoint = str(self.pretrained)
        self.source_root = self.root / "source_tree"
        self.source_root.mkdir()
        for name in ("main.py", "pretrain.py", "requirements.txt", "requirements-paper.txt", "environment-paper.yml"):
            (self.source_root / name).write_text("# synthetic source\n", encoding="utf-8")
        for name in ("models", "pretrain_models", "source", "scripts"):
            directory = self.source_root / name
            directory.mkdir()
            (directory / "example.py").write_text("value = 1\n", encoding="utf-8")
        (self.source_root / "scripts/run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        source_patch = patch.object(capture, "SOURCE_ROOT", self.source_root)
        source_patch.start()
        self.addCleanup(source_patch.stop)
        # Membership/mapping/hash verification has its own full protocol tests.
        # Capture unit fixtures intentionally contain no real dataset or split.
        validator_patch = patch.object(capture, "validate_protocol", return_value=None)
        self.validator = validator_patch.start()
        self.addCleanup(validator_patch.stop)

    def write_protocol(self):
        self.protocol_path.write_text(json.dumps(self.protocol), encoding="utf-8")

    def context(self):
        return capture.load_benchmark_protocol(self.args, native_epochs())

    def start(self):
        return capture.BenchmarkCapture(self.root / "run", self.args, self.context())

    @staticmethod
    def weights():
        return {"weight": torch.tensor([1., 2.]), "counter": torch.tensor(7, dtype=torch.int64)}

    def save_all(self, writer):
        for epoch in native_epochs():
            writer.save(epoch, self.weights())

    def test_native_schedule_is_unchanged_and_explicit(self):
        expected = list(range(460, 591, 10)) + list(range(591, 601))
        self.assertEqual(native_epochs(), expected)
        self.assertEqual(len(expected), 24)
        context = self.context()
        self.assertEqual(context["expected_epochs"], expected)
        self.assertEqual(context["protocol_sha256"], capture.sha256_file(self.protocol_path))
        self.validator.assert_called_once_with(self.protocol)

    def test_disabled_mode_has_no_output_or_protocol_read(self):
        self.args.benchmark_protocol = None
        self.protocol_path.unlink()
        self.assertIsNone(capture.load_benchmark_protocol(self.args, native_epochs()))
        self.assertFalse((self.root / "run").exists())
        self.validator.assert_not_called()

    def test_strict_protocol_rejections_propagate_before_output(self):
        for error in ("Protocol input bytes changed: mapping", "Mapped partition IDs changed",
                      "Protocol field changed: configuration_count"):
            with self.subTest(error=error):
                self.validator.reset_mock()
                self.validator.side_effect = ValueError(error)
                with self.assertRaisesRegex(ValueError, error):
                    self.start()
                self.validator.assert_called_once_with(self.protocol)
                self.assertFalse((self.root / "run").exists())

    def test_duplicate_protocol_json_keys_fail_closed(self):
        raw = self.protocol_path.read_text(encoding="utf-8")
        self.protocol_path.write_text(raw[:-1] + ', "expected_epochs": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate benchmark JSON key"):
            self.context()

    def test_missing_or_manifest_only_array_audit_is_not_training_ready(self):
        for audit in (None, {}, {"mode": "manifest_only", "arrays_checked": 0},
                      {"mode": "manifest_only", "arrays_checked": 2502},
                      {"mode": "all_headers_and_exact_payload_sizes", "arrays_checked": 2500},
                      {"mode": "all_headers_and_exact_payload_sizes", "arrays_checked": 2502.0}):
            with self.subTest(audit=audit):
                self.protocol["array_audit"] = audit
                self.write_protocol()
                with self.assertRaisesRegex(ValueError, "2502-array"):
                    self.start()
                self.assertFalse((self.root / "run").exists())

    def test_pretrained_checkpoint_is_required_and_fingerprinted(self):
        for filename in (None, "", str(self.root / "missing.pth")):
            with self.subTest(filename=filename):
                self.args.uni_encoder_checkpoint = filename
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.start()
        self.args.uni_encoder_checkpoint = str(self.pretrained)
        writer = self.start()
        self.assertEqual(writer.started["pretrained_checkpoint"], {
            "path": str(self.pretrained.resolve()), "sha256": capture.sha256_file(self.pretrained)})

    def test_pretrained_relative_resolution_matches_native_repo_first_order(self):
        checkpoint = self.source_root / "pretrained.pth"
        checkpoint.write_bytes(b"repo relative checkpoint")
        self.args.uni_encoder_checkpoint = checkpoint.name
        context = self.context()
        self.assertEqual(context["pretrained_checkpoint_path"], str(checkpoint.resolve()))

    def test_source_fingerprints_cover_only_required_source_inventory(self):
        for relative in ("models/ignored.pth", "scripts/progress.json", "tests/ignored.py",
                         "source/__pycache__/ignored.py"):
            path = self.source_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not source provenance")
        fingerprints = capture.source_fingerprints(self.source_root)
        self.assertEqual(set(fingerprints), {"main.py", "pretrain.py", "requirements.txt",
            "requirements-paper.txt", "environment-paper.yml", "models/example.py",
            "pretrain_models/example.py", "source/example.py", "scripts/example.py", "scripts/run.sh"})
        self.assertEqual(fingerprints, {name: capture.sha256_file(self.source_root / name)
                                        for name in fingerprints})
        self.assertEqual(self.start().started["source_sha256"], fingerprints)

    def test_source_change_addition_or_deletion_blocks_completion(self):
        original = self.source_root / "models/example.py"
        added = self.source_root / "scripts/new.py"
        for mutation in ("change", "add", "delete"):
            with self.subTest(mutation=mutation):
                writer = capture.BenchmarkCapture(self.root / ("run-" + mutation), self.args, self.context())
                self.save_all(writer)
                if mutation == "change":
                    original.write_text("value = 2\n", encoding="utf-8")
                elif mutation == "add":
                    added.write_text("# new source\n", encoding="utf-8")
                else:
                    original.unlink()
                try:
                    with self.assertRaisesRegex(RuntimeError, "source inventory changed"):
                        writer.complete(600)
                    self.assertFalse((writer.directory / "candidates.json").exists())
                    self.assertFalse((writer.directory / "TRAINING_COMPLETE.json").exists())
                finally:
                    original.write_text("value = 1\n", encoding="utf-8")
                    if added.exists():
                        added.unlink()

    def test_pretrained_change_blocks_completion(self):
        writer = self.start()
        self.save_all(writer)
        self.pretrained.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(RuntimeError, "pretrained Uni-Encoder checkpoint changed"):
            writer.complete(600)
        self.assertFalse((writer.directory / "candidates.json").exists())
        self.assertFalse((writer.directory / "TRAINING_COMPLETE.json").exists())

    def test_wrong_architecture_ema_or_budget_fail_early(self):
        for key, value in (("model_name", "UniMEBase"), ("uni_encoder_name", "UniEncoderBase"),
                           ("use_ema", False), ("use_ema", 1), ("num_epochs", 1000),
                           ("num_epochs", "600"), ("benchmark_protocol", "")):
            original = getattr(self.args, key)
            with self.subTest(key=key, value=value):
                setattr(self.args, key, value)
                with self.assertRaises(ValueError):
                    self.context()
                setattr(self.args, key, original)
        self.assertFalse((self.root / "run").exists())

    def test_wrong_protocol_or_schedule_is_rejected(self):
        cases = [dict(self.protocol, protocol_id="other"),
                 dict(self.protocol, expected_epochs=native_epochs()[:-1]),
                 dict(self.protocol, expected_epochs=list(reversed(native_epochs()))),
                 dict(self.protocol, expected_epochs=[True] + native_epochs()[1:]),
                 dict(self.protocol, expected_epochs=native_epochs() + [600])]
        for protocol in cases:
            self.protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                self.context()

    def test_snapshot_clones_ema_not_later_mutated_student_and_is_weights_only(self):
        definitions = extract_train_definitions({"_swap_named_tensors", "_EMAWeights"})
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(3)
        ema = definitions["_EMAWeights"](model, .5)
        with torch.no_grad():
            model.weight.fill_(10)
            model.bias.fill_(11)
        writer = self.start()
        with ema.use_ema_weights(model):
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        entry = writer.save(460, state)
        for value in state.values():
            value.fill_(99)
        saved = torch.load(writer.directory / entry["path"], map_location="cpu", weights_only=True)
        self.assertEqual(saved["model_variant"], "EMA")
        self.assertEqual(saved["epoch"], 460)
        self.assertEqual(saved["protocol_sha256"], self.context()["protocol_sha256"])
        self.assertEqual(saved["training_config"], self.args.to_dict())
        torch.testing.assert_close(saved["model_state_dict"]["weight"], torch.full((1, 2), 2.))
        torch.testing.assert_close(saved["model_state_dict"]["bias"], torch.full((1,), 3.))
        torch.testing.assert_close(model.weight, torch.full((1, 2), 10.))
        self.assertEqual(set(saved), {"epoch", "model_state_dict", "model_variant", "protocol_sha256", "training_config"})
        self.assertEqual(entry["sha256"], capture.sha256_file(writer.directory / entry["path"]))

    def test_preexisting_output_duplicate_and_out_of_order_epochs_fail_closed(self):
        writer = self.start()
        with self.assertRaises(FileExistsError):
            self.start()
        for epoch in (450, 470, True):
            with self.subTest(epoch=epoch), self.assertRaises(RuntimeError):
                writer.save(epoch, self.weights())
        entry = writer.save(460, self.weights())
        original_hash = entry["sha256"]
        with self.assertRaises(RuntimeError):
            writer.save(460, self.weights())
        self.assertEqual(capture.sha256_file(writer.directory / entry["path"]), original_hash)
        with self.assertRaises(RuntimeError):
            writer.complete(600)
        self.assertFalse((writer.directory / "TRAINING_COMPLETE.json").exists())

    def test_complete_manifest_covers_all_candidates_and_is_immutable(self):
        writer = self.start()
        self.save_all(writer)
        self.assertFalse((writer.directory / "candidates.json").exists())
        with self.assertRaises(RuntimeError):
            writer.complete(599)
        completion = writer.complete(600)
        manifest_path = writer.directory / "candidates.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual([entry["epoch"] for entry in manifest["entries"]], native_epochs())
        self.assertEqual(completion["candidate_count"], 24)
        self.assertEqual(completion["manifest_sha256"], capture.sha256_file(manifest_path))
        self.assertEqual(manifest["protocol_sha256"], self.context()["protocol_sha256"])
        self.assertEqual((writer.directory / "protocol.json").read_bytes(), self.protocol_path.read_bytes())
        for entry in manifest["entries"]:
            self.assertFalse(Path(entry["path"]).is_absolute())
            self.assertEqual(entry["model_variant"], "EMA")
        with self.assertRaises(RuntimeError):
            writer.complete(600)
        with self.assertRaises(RuntimeError):
            writer.save(600, self.weights())

    def test_changed_checkpoint_blocks_completion(self):
        writer = self.start()
        self.save_all(writer)
        path = writer.directory / writer.entries[0]["path"]
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "bytes changed"):
            writer.complete(600)
        self.assertFalse((writer.directory / "TRAINING_COMPLETE.json").exists())

    def test_changed_protocol_blocks_capture(self):
        writer = self.start()
        (writer.directory / "protocol.json").write_bytes(b"{}")
        with self.assertRaisesRegex(RuntimeError, "protocol changed"):
            writer.save(460, self.weights())

    def test_failed_torch_save_does_not_publish_entry_or_completion(self):
        writer = self.start()
        with patch.object(torch, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                writer.save(460, self.weights())
        self.assertEqual(writer.entries, [])
        self.assertFalse((writer.directory / "epoch_0460_ema.json").exists())
        self.assertFalse((writer.directory / "TRAINING_COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            writer.save(460, self.weights())

    def test_capture_does_not_consume_python_numpy_or_torch_rng(self):
        python_state = random.getstate()
        numpy_state = copy.deepcopy(np.random.get_state())
        torch_state = torch.random.get_rng_state().clone()
        writer = self.start()
        self.save_all(writer)
        writer.complete(600)
        self.assertEqual(random.getstate(), python_state)
        actual_numpy = np.random.get_state()
        self.assertEqual(actual_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
        self.assertEqual(actual_numpy[2:], numpy_state[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_cli_flag_defaults_to_none_and_round_trips(self):
        parse_spec = importlib.util.spec_from_file_location("unime_capture_parse_test", ROOT / "source/config/parse.py")
        module = importlib.util.module_from_spec(parse_spec)
        with patch.dict(sys.modules, {parse_spec.name: module}):
            parse_spec.loader.exec_module(module)
            with patch.object(sys, "argv", ["main.py"]):
                self.assertIsNone(module.TrainingConfig.from_args().benchmark_protocol)
            with patch.object(sys, "argv", ["main.py", "--benchmark_protocol", str(self.protocol_path)]):
                args = module.TrainingConfig.from_args()
                self.assertEqual(args.to_dict()["benchmark_protocol"], str(self.protocol_path))

    def test_capture_precedes_native_pruning_and_test_is_bypassed_only_in_benchmark_mode(self):
        train_source = (ROOT / "source/train.py").read_text(encoding="utf-8")
        self.assertLess(train_source.index("benchmark_capture.save(epoch, checkpoint_model_state)"),
                        train_source.index("os.remove(worst_checkpoint[1])"))
        self.assertGreater(train_source.index("benchmark_capture.complete(args.num_epochs)"),
                           train_source.index("# Training complete"))
        main_tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
        guard = [node for node in ast.walk(main_tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "benchmark_context is not None"
                 and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                         and child.func.id == "test_top_k_models" for child in ast.walk(node))]
        self.assertEqual(len(guard), 1)
        self.assertFalse(any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                             and child.func.id == "test_top_k_models"
                             for statement in guard[0].body for child in ast.walk(statement)))


if __name__ == "__main__":
    unittest.main()
