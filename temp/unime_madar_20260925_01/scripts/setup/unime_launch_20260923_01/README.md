# UniME one-extra-GPU launch — 2026-09-23

Authorization: leave V11 `t1` alone and temporarily allow one extra GPU allocation.
Run smoke first, then full training only if all checks pass. No architecture or
training recipe changes. Both jobs use one H200, one process, 16 CPU cores and
192 GiB host memory. Limits: six hours smoke, 14 days full (ceilings, not estimates).
No requeue, arrays, automatic retries or cancellations.

Smoke: compiled FP16 six-step pretraining and finetuning, batch4/crop96; full
1,251-case preparation and value/split/prior checks; native-vs-offline inference
parity on one selection80 case across all15 masks. No test-performance smoke
evaluation. Test data receives only the same case-local preprocessing/value checks.

Only after all checks pass, submit `unime-full` with `afterok` on the smoke job,
so the two UniME GPU allocations never run concurrently. The full job checks
source and smoke receipts, then runs the unchanged benchmark launcher:
600 pretraining +600 finetuning epochs (250 updates/epoch), original80/all15/raw
checkpoint selection, then one frozen winner's251-case historical test.
Native125-case diagnostics remain separate from the80-case final selector.

Artifacts:

- `artifacts/unime/smoke_runs/unime_20260923_01`
- `artifacts/unime/benchmark_runs/unime_20260923_01`
- This directory: submission receipts, source manifest, logs, private caches.

Single-use submit intents prevent duplicate launches. After an ambiguous SSH
response inspect receipts and scheduler comment `unime_20260923_01:smoke` or
`:full`; do not resubmit blindly. Synthetic success is not accuracy evidence.
