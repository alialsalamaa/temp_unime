"""Fail-closed, single-use smoke -> afterok full-run launch; no retries/cancels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

BASE = Path('/adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines')
ROOT = BASE / 'setup/unime_launch_20260923_01'
REPO = BASE / 'UniME'
PYTHON = BASE / '.venv-unime/bin/python'
TAG = 'unime_20260923_01'
SMOKE = BASE / 'artifacts/unime/smoke_runs' / TAG
FULL = BASE / 'artifacts/unime/benchmark_runs' / TAG
DATA = BASE / 'artifacts/unime/data/BRATS2023'
PROOF = BASE / 'artifacts/unime/preprocessing'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def submission_receipt(path):
    # A very fast scheduler can start smoke before sbatch's caller has recorded
    # its response. Wait for that one receipt; never trigger a retry/submission.
    for _ in range(60):
        try:
            return read(path)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.5)
    raise RuntimeError('Submission receipt was not committed: ' + str(path))


def exclusive(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def progress(stage, **extra):
    value = dict(stage=stage, updated_utc=utc(), job_id=os.environ.get('SLURM_JOB_ID'), **extra)
    target = ROOT / 'progress.json'
    temporary = ROOT / ('progress.' + str(os.getpid()) + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, target)
    print(json.dumps(value), flush=True)


def verify_source():
    if Path(__file__).resolve().parent != ROOT or ROOT.is_symlink():
        raise RuntimeError('Run only from the pinned remote launch directory')
    manifest = read(ROOT / 'manifest.json')
    for relative, expected in manifest['files'].items():
        path = BASE / relative
        if path.is_symlink() or not path.resolve().is_relative_to(BASE) or sha(path) != expected:
            raise RuntimeError('Frozen source/configuration changed: ' + relative)
    return sha(ROOT / 'manifest.json')


def allocation():
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdigit() or os.environ.get('SLURM_ARRAY_JOB_ID'):
        raise RuntimeError('A single non-array Slurm allocation is required')
    if os.environ.get('SLURM_RESTART_COUNT', '0') != '0':
        raise RuntimeError('This isolated launch must not be requeued')
    if os.environ.get('SLURM_NTASKS', '1') != '1' or os.environ.get('WORLD_SIZE', '1') != '1':
        raise RuntimeError('Exactly one process/task is required')
    token = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not token or ',' in token:
        raise RuntimeError('Exactly one Slurm GPU must be visible')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one usable GPU is required')
    return dict(job_id=job, gpu=torch.cuda.get_device_name(0), cuda_token=token,
                torch=torch.__version__, cuda=torch.version.cuda)


def run(command, log):
    print('COMMAND ' + json.dumps(list(map(str, command))), flush=True)
    with Path(log).open('x') as output:
        completed = subprocess.run(list(map(str, command)), cwd=REPO, stdout=output,
                                   stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        print(Path(log).read_text(errors='replace')[-16000:], flush=True)
        raise RuntimeError(f'Command failed ({completed.returncode}); see {log}')
    print(Path(log).read_text(errors='replace')[-2500:], flush=True)


def latest_pass(log, stage):
    rows = []
    for line in Path(log).read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('stage') == stage:
            rows.append(value)
    steps = [r for r in rows if r.get('status') == 'step_passed']
    final = [r for r in rows if r.get('status') == 'passed']
    if len(steps) != 6 or [r['step'] for r in steps] != list(range(1, 7)) or len(final) != 1:
        raise RuntimeError('Incomplete six-step smoke: ' + stage)
    return final[0]


def assert_only_this_unime_job(current=None):
    response = subprocess.run(['squeue', '-h', '-u', 'khaled', '-o', '%i|%j|%T'],
                              capture_output=True, text=True, check=True)
    rows = [line.split('|') for line in response.stdout.splitlines() if line.strip()]
    extra = [row for row in rows if row[1].startswith('unime-') and row[0] != current]
    if extra:
        raise RuntimeError('Another UniME job already exists: ' + repr(extra))
    return rows


def submit(mode, dependency=None):
    verify_source()
    assert_only_this_unime_job(os.environ.get('SLURM_JOB_ID') if mode == 'full' else None)
    intent = ROOT / (mode.upper() + '_SUBMIT_INTENT.json')
    command = ['sbatch', '--parsable', '--chdir=' + str(ROOT), '--comment=' + TAG + ':' + mode,
               '--partition=main', '--nodes=1', '--ntasks=1', '--gres=gpu:nvidia_h200:1',
               '--cpus-per-task=16', '--mem=192G', '--no-requeue']
    if dependency is not None:
        if not dependency.isdigit() or mode != 'full':
            raise ValueError('Invalid smoke dependency')
        command.append('--dependency=afterok:' + dependency)
    command.append(str(ROOT / (mode + '.slurm')))
    exclusive(intent, dict(command=command, created_utc=utc(), dependency=dependency,
                           warning='Single-use intent. Never resubmit blindly after ambiguous response.'))
    environment = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    response = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    exclusive(ROOT / (mode.upper() + '_SBATCH_RESPONSE.json'),
              dict(returncode=response.returncode, stdout=response.stdout, stderr=response.stderr, utc=utc()))
    if response.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s]+)?\s*', response.stdout):
        raise RuntimeError('sbatch failed or returned ambiguous output; inspect intent/queue before any retry')
    job = response.stdout.strip().split(';')[0]
    exclusive(ROOT / (mode.upper() + '_SUBMITTED.json'),
              dict(job_id=job, mode=mode, dependency=dependency, created_utc=utc(), command=command))
    print(json.dumps(dict(submitted=mode, job_id=job, dependency=dependency)), flush=True)
    return job


def configure_runtime():
    cache = ROOT / 'cache'
    for key, folder in dict(XDG_CACHE_HOME='xdg', TORCH_HOME='torch', TRITON_CACHE_DIR='triton',
                            TORCHINDUCTOR_CACHE_DIR='inductor', CUDA_CACHE_PATH='cuda',
                            TORCH_EXTENSIONS_DIR='extensions', WANDB_DIR='wandb').items():
        path = cache / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    scratch = Path(tempfile.mkdtemp(prefix='unime.' + os.environ['SLURM_JOB_ID'] + '.', dir='/tmp'))
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(scratch)
    return scratch


def smoke():
    manifest_sha = verify_source()
    gpu = allocation()
    expected = submission_receipt(ROOT / 'SMOKE_SUBMITTED.json')['job_id']
    if gpu['job_id'] != expected or SMOKE.exists() or SMOKE.is_symlink():
        raise RuntimeError('Unexpected job or reused smoke directory')
    SMOKE.mkdir(parents=True, exist_ok=False)
    exclusive(SMOKE / 'REQUEST.json', dict(created_utc=utc(), manifest_sha256=manifest_sha, **gpu,
              task='One extra GPU smoke, then one full job on success; t1 remains untouched',
              pretrain_epochs=600, finetune_epochs=600, global_batch=4, crop=96,
              synthetic_smoke_does_not_estimate_accuracy=True))
    checks = {}
    for stage in ('pretrain', 'finetune'):
        progress('smoke_' + stage, output=str(SMOKE / (stage + '.log')))
        log = SMOKE / (stage + '.log')
        run([PYTHON, '-B', REPO / 'scripts/smoke_training_gpu.py', '--stage', stage, '--compile', '--steps', '6'], log)
        checks[stage] = latest_pass(log, stage)
        verify_source()
    previous = {}
    for name in ('preprocessing_report.json', 'run_manifest.json', 'status.json'):
        path = PROOF / name
        if path.is_file():
            previous[name] = dict(sha256=sha(path), content=read(path))
    exclusive(SMOKE / 'PREPROCESSING_PREVIOUS_RECORDS.json', previous)
    progress('preprocessing', output=str(SMOKE / 'preprocessing.log'))
    run([PYTHON, '-B', BASE / 'baseline_adapters/unime/preprocess_full.py', '--workers', '4'], SMOKE / 'preprocessing.log')
    report = read(PROOF / 'preprocessing_report.json')
    if not (report.get('status') == 'passed' and report.get('arrays_validated') == 2502
            and report.get('cases_validated') == 1251 and report.get('full_value_scan_passed') is True):
        raise RuntimeError('Complete data preparation/value scan did not pass')
    exclusive(SMOKE / 'PREPROCESSING_PASSED.json', report)
    progress('protocol')
    protocol = SMOKE / 'protocol.json'
    run([PYTHON, '-B', REPO / 'scripts/prepare_benchmark.py',
         '--split-json', BASE / 'artifacts/splits/official_imfuse_split.json',
         '--validation-partition', BASE / 'artifacts/unime/reference/validation_partition.json',
         '--data-root', DATA, '--raw-data-root', BASE / 'data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData',
         '--output', protocol], SMOKE / 'protocol.log')
    progress('inference_parity', output=str(SMOKE / 'inference_parity.log'))
    run([PYTHON, '-B', REPO / 'scripts/smoke_inference_gpu.py', '--protocol', protocol,
         '--output', SMOKE / 'INFERENCE_PARITY.json'], SMOKE / 'inference_parity.log')
    parity = read(SMOKE / 'INFERENCE_PARITY.json')
    if not (parity.get('state') == 'passed'
            and parity.get('check') == 'full_unime_native_test_offline_inference_parity'
            and parity.get('protocol_sha256') == sha(protocol)
            and parity.get('partition') == 'selection80'
            and parity.get('configuration_count') == 15
            and parity.get('inference_cases') == 1
            and parity.get('test_payloads_loaded') is False
            and parity.get('runtime', {}).get('slurm_job_id') == gpu['job_id']):
        raise RuntimeError('Full-model inference parity did not pass')
    verify_source()
    exclusive(SMOKE / 'SMOKE_PASSED.json', dict(status='passed', completed_utc=utc(), **gpu,
              manifest_sha256=manifest_sha, protocol_sha256=sha(protocol), synthetic_steps=checks,
              evidence_sha256={name: sha(SMOKE / name) for name in (
                  'pretrain.log', 'finetune.log', 'PREPROCESSING_PASSED.json', 'INFERENCE_PARITY.json')},
              test_cases_used_for_smoke_inference=0, full_run=str(FULL)))
    progress('smoke_passed_submitting_full', smoke=str(SMOKE))
    # afterok prevents the replacement allocation from starting before this
    # smoke job exits successfully. A failing smoke never reaches submission.
    full_job = submit('full', dependency=gpu['job_id'])
    progress('full_submitted_after_smoke', full_job_id=full_job, dependency='afterok:' + gpu['job_id'])


def full():
    manifest_sha = verify_source()
    gpu = allocation()
    submitted = read(ROOT / 'FULL_SUBMITTED.json')
    passed = read(SMOKE / 'SMOKE_PASSED.json')
    if (gpu['job_id'] != submitted['job_id'] or submitted['dependency'] != passed['job_id']
            or passed['status'] != 'passed' or passed['manifest_sha256'] != manifest_sha
            or passed['protocol_sha256'] != sha(SMOKE / 'protocol.json')):
        raise RuntimeError('Full training requires this launch\'s completed smoke')
    for name, expected in passed['evidence_sha256'].items():
        if sha(SMOKE / name) != expected:
            raise RuntimeError('Smoke evidence changed: ' + name)
    assert_only_this_unime_job(gpu['job_id'])
    smoke_complete = False
    # afterok is the scheduler safety barrier. Accounting may lag its release.
    for _ in range(6):
        accounted = subprocess.run(['sacct', '-X', '-n', '-P', '-j', passed['job_id'],
                                   '--format=JobIDRaw,State,ExitCode'], capture_output=True, text=True, check=True)
        rows = [line.split('|') for line in accounted.stdout.splitlines() if line.strip()]
        if any(row[:3] == [passed['job_id'], 'COMPLETED', '0:0'] for row in rows):
            smoke_complete = True
            break
        time.sleep(5)
    if not smoke_complete:
        raise RuntimeError('Smoke allocation is not recorded COMPLETED/0:0; no training launched')
    os.environ['RUN_DIR'] = str(FULL)
    progress('full_training', run_dir=str(FULL))
    result = subprocess.run(['bash', str(REPO / 'scripts/run_nebius_benchmark.sh')], cwd=REPO)
    if result.returncode:
        raise RuntimeError(f'Full benchmark exited with {result.returncode}; no automatic retry')
    progress('full_complete', run_dir=str(FULL))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('submit-smoke', 'smoke', 'full', 'verify'))
    args = parser.parse_args()
    if args.mode == 'verify':
        print(json.dumps(dict(status='source_verified', manifest_sha256=verify_source())))
        return
    if args.mode == 'submit-smoke':
        submit('smoke')
        return
    scratch = None
    try:
        scratch = configure_runtime()
        (smoke if args.mode == 'smoke' else full)()
    except BaseException:
        progress('failed', mode=args.mode, traceback=traceback.format_exc())
        raise
    finally:
        if scratch is not None:
            try:
                scratch.rmdir()
            except OSError:
                pass


if __name__ == '__main__':
    main()
