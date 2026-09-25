"""Separate-process validation selection, then one frozen checkpoint's test."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "test", "verify"))
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from source.benchmark_runner import run
    print(json.dumps(run(args.mode, args.candidates, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
