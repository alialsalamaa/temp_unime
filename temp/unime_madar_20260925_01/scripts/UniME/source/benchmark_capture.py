"""Optional immutable EMA capture; does not evaluate or alter native training."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path


PROTOCOL_ID = "unime_original80_all15_rawdice_v1"
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprints(root):
    """Hash the training/evaluation source inventory, never caches or outputs."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Source root must be an existing non-symlink directory")
    root = root.resolve()
    paths = {root / "main.py", root / "pretrain.py"}
    for name in ("models", "pretrain_models", "source", "scripts"):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Missing or unsafe source directory: " + name)
        for parent, directories, filenames in os.walk(directory, followlinks=False):
            directories[:] = sorted(entry for entry in directories if entry != "__pycache__")
            if any((Path(parent) / entry).is_symlink() for entry in directories):
                raise ValueError("Source directory symlinks are not supported")
            paths.update(Path(parent) / filename for filename in filenames
                         if Path(filename).suffix in (".py", ".sh"))
    for pattern in ("requirements*", "environment*.yml"):
        paths.update(root.glob(pattern))
    result = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing or unsafe source file: " + str(path))
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def validate_protocol(protocol):
    # Lazy import keeps disabled capture and standalone CPU helper imports cheap.
    # No override of verify_input_files: frozen membership and hashes are mandatory.
    from source.benchmark_protocol import validate_protocol as strict_validate_protocol
    return strict_validate_protocol(protocol)


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate benchmark JSON key: " + key)
        result[key] = value
    return result


def _pretrained_checkpoint(args):
    """Match native repo-first resolution without importing model dependencies."""
    filename = getattr(args, "uni_encoder_checkpoint", None)
    if not isinstance(filename, str) or not filename:
        raise ValueError("Benchmark capture requires an explicit existing Uni-Encoder checkpoint")
    path = Path(filename).expanduser()
    candidates = [path] if path.is_absolute() else [SOURCE_ROOT / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Benchmark Uni-Encoder checkpoint is missing: " + filename)


def _exclusive_bytes(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _exclusive_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _exclusive_bytes(path, raw)


def load_benchmark_protocol(args, expected_epochs):
    """Validate before model allocation; caller supplies the native epoch list."""
    filename = getattr(args, "benchmark_protocol", None)
    if filename is None:
        return None
    if (not isinstance(filename, str) or not filename
            or args.model_name != "UniME"
            or getattr(args, "uni_encoder_name", None) not in (None, "UniEncoder")
            or getattr(args, "use_ema", False) is not True
            or type(args.num_epochs) is not int or args.num_epochs != 600):
        raise ValueError("Benchmark capture requires original UniME/UniEncoder, EMA and 600 epochs")
    expected = list(expected_epochs)
    if (not expected or any(type(epoch) is not int for epoch in expected)
            or expected != sorted(set(expected)) or expected[-1] != 600):
        raise ValueError("Invalid native validation epoch schedule")
    raw = Path(filename).read_bytes()
    protocol = json.loads(raw, object_pairs_hook=_unique_json_pairs)
    if (not isinstance(protocol, dict) or protocol.get("protocol_id") != PROTOCOL_ID
            or not isinstance(protocol.get("expected_epochs"), list)
            or any(type(epoch) is not int for epoch in protocol["expected_epochs"])
            or protocol["expected_epochs"] != expected):
        raise ValueError("Benchmark protocol must exactly match the native validation epochs")
    validate_protocol(protocol)
    audit = protocol.get("array_audit")
    if (not isinstance(audit, dict)
            or audit.get("mode") != "all_headers_and_exact_payload_sizes"
            or type(audit.get("arrays_checked")) is not int or audit["arrays_checked"] != 2502):
        raise ValueError("Benchmark training requires a complete 2502-array header/payload audit")
    pretrained_checkpoint = _pretrained_checkpoint(args)
    return {"protocol": protocol, "raw": raw, "protocol_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_epochs": expected, "pretrained_checkpoint_path": str(pretrained_checkpoint)}


def assert_fresh_capture_output(log_dir):
    path = Path(log_dir) / "benchmark_candidates"
    if path.exists() or path.is_symlink():
        raise FileExistsError("Benchmark capture output already exists; choose a fresh run directory")


class BenchmarkCapture:
    """Main-rank-only, append-only files; final manifest and completion are write-once."""

    def __init__(self, log_dir, args, context):
        if context is None:
            raise ValueError("A validated benchmark protocol is required")
        assert_fresh_capture_output(log_dir)
        self.directory = Path(log_dir) / "benchmark_candidates"
        self.expected_epochs = list(context["expected_epochs"])
        self.protocol_sha256 = context["protocol_sha256"]
        self.training_config = copy.deepcopy(args.to_dict())
        # Fail before creating output if config cannot be safely recorded.
        json.dumps(self.training_config, allow_nan=False)
        self.source_root = SOURCE_ROOT.resolve()
        source_sha256 = source_fingerprints(self.source_root)
        pretrained_path = _pretrained_checkpoint(args)
        if str(pretrained_path) != context["pretrained_checkpoint_path"]:
            raise RuntimeError("Resolved Uni-Encoder checkpoint changed after preflight")
        pretrained_checkpoint = dict(path=str(pretrained_path), sha256=sha256_file(pretrained_path))
        self.entries = []
        self.finished = False
        self.directory.mkdir(parents=True, exist_ok=False)
        _exclusive_bytes(self.directory / "protocol.json", context["raw"])
        self.started = dict(schema_version=1, protocol_id=PROTOCOL_ID,
                            protocol_sha256=self.protocol_sha256, model_variant="EMA",
                            expected_epochs=self.expected_epochs,
                            training_config=self.training_config,
                            source_sha256=source_sha256, pretrained_checkpoint=pretrained_checkpoint)
        _exclusive_json(self.directory / "capture_started.json", self.started)

    def _verify_header(self):
        if self.directory.is_symlink():
            raise RuntimeError("Capture directory must not become a symlink")
        if sha256_file(self.directory / "protocol.json") != self.protocol_sha256:
            raise RuntimeError("Frozen capture protocol changed")
        if json.loads((self.directory / "capture_started.json").read_text()) != self.started:
            raise RuntimeError("Frozen capture metadata changed")

    def save(self, epoch, model_state_dict):
        """Snapshot the already-cloned EMA state before native top-k pruning."""
        import torch
        self._verify_header()
        if (self.finished or type(epoch) is not int or len(self.entries) >= len(self.expected_epochs)
                or epoch != self.expected_epochs[len(self.entries)]):
            raise RuntimeError("Duplicate, missing, out-of-order or unexpected candidate epoch")
        if not isinstance(model_state_dict, dict) or not model_state_dict:
            raise ValueError("A nonempty EMA model state dictionary is required")
        if any(not isinstance(name, str) or not isinstance(value, torch.Tensor)
               for name, value in model_state_dict.items()):
            raise ValueError("Only named model tensors may be captured")
        state = {name: value.detach().cpu().clone() for name, value in model_state_dict.items()}
        payload = dict(epoch=epoch, model_state_dict=state, model_variant="EMA",
                       protocol_sha256=self.protocol_sha256,
                       training_config=copy.deepcopy(self.training_config))
        path = self.directory / f"epoch_{epoch:04d}_ema.pth"
        with path.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        entry = dict(epoch=epoch, path=path.name, sha256=sha256_file(path), model_variant="EMA")
        _exclusive_json(path.with_suffix(".json"), entry)
        self.entries.append(entry)
        return dict(entry)

    def complete(self, completed_epochs):
        self._verify_header()
        if (self.finished or type(completed_epochs) is not int or completed_epochs != 600
                or [entry["epoch"] for entry in self.entries] != self.expected_epochs):
            raise RuntimeError("Cannot complete capture without all native candidates and 600 epochs")
        if source_fingerprints(self.source_root) != self.started["source_sha256"]:
            raise RuntimeError("Frozen training source inventory changed")
        pretrained = self.started["pretrained_checkpoint"]
        if sha256_file(pretrained["path"]) != pretrained["sha256"]:
            raise RuntimeError("Frozen pretrained Uni-Encoder checkpoint changed")
        for entry in self.entries:
            path = self.directory / entry["path"]
            if path.is_symlink() or sha256_file(path) != entry["sha256"]:
                raise RuntimeError("Captured candidate bytes changed")
            if json.loads(path.with_suffix(".json").read_text()) != entry:
                raise RuntimeError("Captured candidate descriptor changed")
        manifest = dict(schema_version=1, protocol_id=PROTOCOL_ID,
                        protocol_sha256=self.protocol_sha256, model_variant="EMA",
                        expected_epochs=self.expected_epochs, entries=self.entries)
        manifest_path = self.directory / "candidates.json"
        _exclusive_json(manifest_path, manifest)
        complete = dict(schema_version=1, state="complete", protocol_id=PROTOCOL_ID,
                        protocol_sha256=self.protocol_sha256, model_variant="EMA",
                        completed_epochs=completed_epochs, candidate_count=len(self.entries),
                        expected_epochs=self.expected_epochs, manifest_path="candidates.json",
                        manifest_sha256=sha256_file(manifest_path),
                        automatic_test_performed=False)
        _exclusive_json(self.directory / "TRAINING_COMPLETE.json", complete)
        self.finished = True
        return complete
