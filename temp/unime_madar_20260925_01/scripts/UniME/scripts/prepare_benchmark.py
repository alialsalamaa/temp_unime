"""Prepare original80/all15 original-grid raw-Dice metadata; never train.

Native-grid scores remain separately labelled diagnostics. No training,
architecture, preprocessing, or inference recipe is modified by this command.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def load_protocol_module():
    # Importing source.__init__ would import Torch through the native dataloader.
    path = Path(__file__).resolve().parents[1] / "source/benchmark_protocol.py"
    spec = importlib.util.spec_from_file_location("unime_benchmark_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--validation-partition", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Processed BRATS2023 directory")
    parser.add_argument("--raw-data-root", type=Path, required=True,
                        help="Original MRI directory for approved full-grid evaluation only")
    parser.add_argument("--output", type=Path, required=True, help="New protocol.json; existing files are never replaced")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Skip2502NPYheader checks; record this limitation explicitly")
    args = parser.parse_args(argv)
    protocol = load_protocol_module()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refusing to replace an existing protocol: " + str(args.output))
    result = protocol.prepare_protocol(args.split_json, args.validation_partition, args.data_root,
                                       raw_data_root=args.raw_data_root,
                                       verify_arrays=not args.manifest_only)
    protocol.write_json_exclusive(result, args.output)
    print(json.dumps({"state": "prepared", "protocol_id": result["protocol_id"],
                      "output": str(args.output.resolve()), "sha256": protocol.sha256_file(args.output),
                      "case_counts": {name: len(ids) for name, ids in result["case_ids"].items()},
                      "expected_epochs": result["expected_epochs"], "array_audit": result["array_audit"],
                      "scoring_domain": result["scoring_domain"], "dice_epsilon": result["dice_epsilon"],
                      "native_diagnostic_grid": result["native_diagnostic_grid"],
                      "training_started": False, "gpu_used": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
