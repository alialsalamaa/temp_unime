#!/usr/bin/env bash
# Invoke only inside a separately authorized, existing one-GPU allocation.
# This file intentionally contains no scheduler submission or cancellation.
set -euo pipefail
umask 077

die() { printf '%s\n' "$*" >&2; exit 1; }
[[ $# -eq 0 ]] || die 'No extra arguments are accepted; choose a fresh RUN_DIR.'
[[ -n "${SLURM_JOB_ID:-}" ]] || die 'An existing authorized Slurm allocation is required.'
[[ "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]] || die 'Unexpected Slurm job ID.'
[[ "${SLURM_NTASKS:-1}" == 1 && "${WORLD_SIZE:-1}" == 1 ]] || die 'Exactly one task/process is required.'
[[ -z "${LOCAL_RANK+x}" && -z "${ACCELERATE_PROCESS_INDEX+x}" ]] || die 'Do not launch this wrapper through DDP/Accelerate.'
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || die 'Exactly one scheduler-visible GPU token is required.'
[[ -n "${RUN_DIR:-}" ]] || die 'Set RUN_DIR to a new direct child of artifacts/unime/benchmark_runs.'

base=/adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines
repo="$base/UniME"
python="$base/.venv-unime/bin/python"
run_parent="$base/artifacts/unime/benchmark_runs"
raw="$base/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
processed="$base/artifacts/unime/data/BRATS2023"
split="$base/artifacts/splits/official_imfuse_split.json"
partition="$base/artifacts/unime/reference/validation_partition.json"
preprocessor="$base/baseline_adapters/unime/preprocess_full.py"
[[ -x "$python" ]] || die "Missing prepared environment: $python"
[[ "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" == "$repo" ]] || die 'Launcher is not in the declared remote UniME source directory.'
[[ -d "$raw" && -f "$split" && -f "$partition" && -f "$preprocessor" ]] || die 'Required existing raw-data link, split, partition, or preprocessor is missing.'

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export UNIME_PYTHON="$python"
export UNIME_DATA_PATH="$base/artifacts/unime/data"
export UNIME_SEED=40
export UNIME_WANDB_MODE=disabled
export WANDB_MODE=disabled
# Both native entrypoints copy --gpu_ids into CUDA_VISIBLE_DEVICES. Preserve
# Slurm's physical index/UUID verbatim; the sole visible device is logical cuda:0.
export UNIME_GPU_IDS="$CUDA_VISIBLE_DEVICES"
unset UNIME_PRETRAIN_CHECKPOINT

# Check the assigned device before any data preparation, run outputs, or model.
"$python" -B -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one usable allocated GPU is required"; print("Allocated logical cuda:0:", torch.cuda.get_device_name(0), flush=True)'

# Resolve and validate the exact new target before creating it. No overwrite or
# resume is permitted, including an existing empty directory or symlink.
RUN_DIR="$("$python" -B -c '
import pathlib, sys
target, parent = map(pathlib.Path, sys.argv[1:])
if not target.is_absolute() or target.exists() or target.is_symlink():
    raise SystemExit("RUN_DIR must be a new absolute non-symlink path")
if target.parent.resolve() != parent.resolve() or not target.name or len(target.name) > 64:
    raise SystemExit("RUN_DIR must be a new direct child of " + str(parent) + "; name <=64 characters")
if parent.is_symlink() or parent.resolve() != parent:
    raise SystemExit("Run parent must not traverse symlinks")
parent.mkdir(parents=True, exist_ok=True)
target.mkdir(exist_ok=False)
print(target.resolve())
' "$RUN_DIR" "$run_parent")"
export RUN_DIR
export UNIME_PRETRAIN_LOG_ROOT="$RUN_DIR/pretrain"
export UNIME_FINETUNE_LOG_ROOT="$RUN_DIR/finetune"
mkdir -- "$RUN_DIR/logs"
# Keep multiprocessing socket names short. This is private node-local runtime
# scratch, not dataset storage. Nonempty scratch is preserved on failure.
run_tmp="$(mktemp -d "/tmp/unime.${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$run_tmp" TMP="$run_tmp" TEMP="$run_tmp"
trap 'rmdir -- "$run_tmp" 2>/dev/null || true' EXIT
protocol="$RUN_DIR/protocol.json"
candidates="$UNIME_FINETUNE_LOG_ROOT/BRATS2023-40-Normal/UniME/benchmark_candidates"
results="$RUN_DIR/benchmark"
cd -- "$repo"

freeze_or_check() {
    "$python" -B -c '
import importlib.util, json, os, pathlib, sys
mode, repo, base, run = sys.argv[1], *map(pathlib.Path, sys.argv[2:])
def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, repo / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
capture = load("standalone_capture", "source/benchmark_capture.py")
protocol = load("standalone_protocol", "source/benchmark_protocol.py")
extras = ("baseline_adapters/unime/preprocess_full.py",
          "artifacts/splits/official_imfuse_split.json",
          "artifacts/unime/reference/validation_partition.json",
          "UniME/BENCHMARK_PROTOCOL.md")
record = dict(source_sha256=capture.source_fingerprints(repo),
              input_sha256={name: protocol.sha256_file(base / name) for name in extras},
              environment={key: os.environ.get(key) for key in (
                  "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "UNIME_PYTHON", "UNIME_DATA_PATH",
                  "UNIME_GPU_IDS", "UNIME_SEED", "UNIME_WANDB_MODE", "WANDB_MODE",
                  "UNIME_PRETRAIN_LOG_ROOT", "UNIME_FINETUNE_LOG_ROOT", "TMPDIR")},
              commands=[["bash", "scripts/pretrain.sh"],
                        ["bash", "scripts/finetune.sh", "--benchmark_protocol", str(run / "protocol.json")]],
              stage_epochs=600, updates_per_epoch=250, seed=40,
              source_status="existing_local_paper_configuration_not_certified_author_reproduction")
path = run / "SOURCE_CONFIG_FROZEN.json"
if mode == "freeze":
    protocol.write_json_exclusive(record, path)
elif mode == "check":
    if protocol.read_json(path) != record:
        raise SystemExit("Source, frozen inputs, command recipe, or runtime selectors changed")
else:
    raise SystemExit("Unknown provenance mode")
' "$1" "$repo" "$base" "$RUN_DIR"
}

freeze_or_check freeze
# The existing native preparation helper maintains shared, mutable preparation
# records. Archive its prior small JSON records before its validated reuse/pass.
"$python" -B -c '
import hashlib, json, pathlib, sys
base, run = map(pathlib.Path, sys.argv[1:])
records = {}
for name in ("preprocessing_report.json", "run_manifest.json", "status.json"):
    path = base / "artifacts/unime/preprocessing" / name
    if path.is_file():
        raw = path.read_bytes()
        records[name] = dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(), content=json.loads(raw))
with (run / "PREPROCESSING_PREVIOUS_RECORDS.json").open("x") as handle:
    json.dump(records, handle, indent=2, allow_nan=False)
    handle.write("\n")
' "$base" "$RUN_DIR"

"$python" -B "$preprocessor" --workers 4 2>&1 | tee "$RUN_DIR/logs/preprocessing.log"
freeze_or_check check
"$python" -B scripts/prepare_benchmark.py \
    --split-json "$split" --validation-partition "$partition" \
    --data-root "$processed" --raw-data-root "$raw" --output "$protocol" \
    2>&1 | tee "$RUN_DIR/logs/protocol.log"

"$python" -B -c '
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
with (path.parent / "PROTOCOL_FROZEN.json").open("x") as handle:
    json.dump(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()), handle)
    handle.write("\n")
' "$protocol"

check_protocol() {
    "$python" -B -c '
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected = json.loads((path.parent / "PROTOCOL_FROZEN.json").read_text())
if expected != dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()):
    raise SystemExit("Frozen benchmark protocol changed")
' "$protocol"
}

freeze_or_check check
check_protocol
bash scripts/pretrain.sh 2>&1 | tee "$RUN_DIR/logs/pretrain.log"
freeze_or_check check
check_protocol
bash scripts/finetune.sh --benchmark_protocol "$protocol" 2>&1 | tee "$RUN_DIR/logs/finetune.log"
freeze_or_check check
check_protocol
"$python" -B scripts/run_benchmark.py select --candidates "$candidates" --output "$results" \
    2>&1 | tee "$RUN_DIR/logs/select.log"
freeze_or_check check
check_protocol
"$python" -B scripts/run_benchmark.py test --candidates "$candidates" --output "$results" \
    2>&1 | tee "$RUN_DIR/logs/test.log"
freeze_or_check check
check_protocol
"$python" -B scripts/run_benchmark.py verify --candidates "$candidates" --output "$results" \
    2>&1 | tee "$RUN_DIR/logs/verify.log"
[[ -f "$results/TEST_COMPLETE.json" ]] || die 'No complete one-checkpoint test receipt was produced.'
printf 'UniME benchmark completed and verified: %s\n' "$results"
