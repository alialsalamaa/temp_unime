# UniME common-split benchmark

This is a prepared, no-submit workflow. It is not evidence that preprocessing,
GPU inference parity, training, or the benchmark has completed. Run it only in a
separately authorized single-GPU Slurm allocation, after the current authorized
job has finished. The launcher neither requests resources nor submits another job.

## Current recipe and what stays unchanged

The 23 September six-item restoration changes the stem LR multiplier, loss
epsilon, scheduler clock, prior dimensions and short-axis cropping. Paper LR
values, modality masking probability0.5, symmetric padding128, segmentation
architecture, augmentation, precision, EMA and native inference are retained.
The independent patch sampler is unchanged because it was not in that approved
list. See PAPER_REPRODUCTION.md for exact values and remaining implementation
choices. Pretraining and segmentation fine-tuning run sequentially on the
same assigned GPU: each is 600 epochs of 250 updates (150,000 updates per stage),
with seed 40, batch size 4 and 96-cubed training crops. The wrappers remain the
source of the exact recipe; they and the source inventory are hashed before
pretraining. Native entrypoints also save their resolved configurations.

The native pretraining transfer remains its existing best EMA reconstruction
checkpoint, `ema_best_checkpoint.pth`. That reconstruction-selection pass uses
training cases, not the held-out test. Internal fine-tuning validation on all 125
validation cases is retained at its native cadence, with its original native-grid
metric and top-k bookkeeping. The 45 development cases therefore still appear in
native diagnostics, but cannot choose the final benchmark checkpoint. They are
not an untouched independent lockbox. Extra all-mask evaluation runs in a separate
process after training, so it cannot consume the training loader random generator.

The recipe is the approved **paper-first, code-fallback configuration**, with
the scoped restorations above, not byte-identical author code. See
[PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md), especially its upstream-versus-local
parameter table and paper/code-discrepancy caveat. Its recorded base is upstream
commit `88ef06d`; remaining differences include modality and patch masking,
paper learning-rate values, transformer-block LR decay and compiled grouping,
raw L2 regularization, minimum-size padding and the normalization mask. The
official scheduler mechanics, prior dimensions and stem multiplier are now
restored. The written paper and released
implementation do not fully agree or specify every choice. Preserving this local
interpretation does not establish which exact recipe generated the authors'
published scores, nor that these are empirically optimal hyperparameters on our
split. The authorized restoration is not a new architecture search or a
test-result-driven hyperparameter experiment. Report the differences and frozen
source/configuration alongside results.

`PAPER_REPRODUCTION.md` describes the ordinary native selection/testing path;
the explicitly enabled benchmark mode changes final selection and testing as
documented below. Its earlier environment/readiness observations are historical,
not proof of the current environment or completed full-model GPU parity. The
new run's runtime records provide its actual dependency/device provenance.

### Missing-modality preprocessing disclosure

The preserved local preprocessing loads all four acquired modalities first.
Both crop bounds and the normalization brain mask use their union of nonzero
voxels (`scripts/brats23_process.py`, `process_case` and `normalize_channels`).
Modality availability masks are applied later to those already-preprocessed
arrays. Consequently, even a nominally single-modality prediction uses geometry
and a normalization mask derived with all four channels available. This is
simulated missing-modality evaluation after shared preprocessing, not a claim
that the complete pipeline was tested with genuinely absent raw modalities.
The crop mask is derived from image intensities, not tumor labels; nevertheless,
the all-four-channel preprocessing assumption must be disclosed. It is preserved
here to avoid silently changing the existing recipe. Supporting truly absent
raw modalities would require a separately specified and validated preprocessing
protocol, outside this benchmark change.

## Final selection and reporting

Use the frozen 875/125/251 case split and the original 80/45 subdivision of the
125 validation cases. The frozen split audit found **875 training case/scan IDs
from 757 unique patient IDs, including 118 repeated-scan patient groups**;
validation has 125 case IDs from 125 patients, and test has 251 case IDs from 251
patients. Patient IDs are the case IDs without their final scan suffix. There is
no patient-ID overlap across those three partitions. Therefore report training
as 875 cases/scans, not 875 unique patients. Distinct scans of the same patient
are allowed within a partition; duplicate case IDs and patient overlap across
train/validation/test or the 80/45 subdivision remain prohibited.

Capture the already-existing EMA weights at the 24 native fine-tuning validation
epochs: 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
591, 592, 593, 594, 595, 596, 597, 598, 599 and 600. There is no extra training
validation schedule and no extra EMA variant. Preserve these snapshots before
native top-k pruning; capture does not change which native weights are trained.

After all training completes, evaluate every snapshot on the original 80 selection
cases, all 15 nonempty modality combinations, and raw WT/TC/ET Dice. The selection
score is the equally weighted mean of those 45 configuration/region values,
without score rounding; exact ties choose the earliest epoch. Freeze the complete
ranking and one winner before constructing test data for inference. Native
automatic multi-checkpoint testing is disabled only in this benchmark mode.

The approved primary score uses the original registered MRI grid and AVMUR's
`1e-6` Dice epsilon. Native crop/pad preprocessing and prediction stay unchanged:
remove padding, restore the predicted crop to its original geometry, and score
the restored segmentation. This is evaluation-only geometry/scoring adaptation,
not new training, interpolation, calibration, ensembling or postprocessing.
Separately preserve the native processed-grid WT/TC/ET and native ET postprocessing
diagnostic with native epsilon `1e-8`. `ET_postpro` is explicitly a native-grid
diagnostic and is never part of the primary raw-45 selector.

Test the one frozen validation winner on all 251 test cases and all 15 combinations.
This test split has been evaluated historically in the wider AVMUR study: describe
it as a reused historical test, not a newly untouched/blind test. Do not choose a
different checkpoint using these test results. Output the 45 primary values and
the separately labelled native reference values, with full per-case records and
checkpoint/protocol/source hashes.

This is a common-split, common-final-selection benchmark, not an equal-compute
comparison or a guarantee of identical author reproduction. UniME's updates,
pretraining cost and candidate cadence differ from AVMUR's 1,500-epoch recipe.
Use original V11's 80-case-selected model as the matched primary comparator;
label the later 125-case retrospective t1 selection separately.

## Remote layout and invocation

All existing inputs are below
`/adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines`:

- Repository: `UniME`; existing environment: `.venv-unime/bin/python`.
- Raw data: `data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` (existing link).
- Processed data: `artifacts/unime/data/BRATS2023`.
- Split: `artifacts/splits/official_imfuse_split.json`.
- Original subdivision: `artifacts/unime/reference/validation_partition.json`.

Inside the authorized allocation only, choose a never-used run name:

```bash
RUN_DIR=/adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/artifacts/unime/benchmark_runs/CHOOSE_A_NEW_RUN_NAME \
  bash /adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/UniME/scripts/run_nebius_benchmark.sh
```

No submission command is provided or executed here. The launcher rejects missing
Slurm allocation, multiple visible GPUs/tasks, distributed launch wrappers, and
existing run directories, including empty directories and symlinks. It preserves
Slurm's exact GPU visibility token; the assigned device is logical `cuda:0`.
W&B is disabled. Model logs, protocol, source/configuration provenance, candidate
snapshots and results belong to the new run. Short private `/tmp/unime.JOB.*`
scratch avoids multiprocessing Unix-socket path-length failures; nonempty scratch
is not recursively deleted.

The stages are corrected preprocessing/verification, immutable protocol preparation
with all 2,502 array headers/payload sizes checked, native pretraining, native
fine-tuning with benchmark capture, separate selection, one-checkpoint testing,
and artifact verification. Heavy preprocessing and GPU work occur only inside
that future allocation, never as a login-node preparation step.

The restored learned prior is `(160,180,210)` in D/H/W order. Every training
array must fit before pretraining. Old preprocessing/protocol receipts do not
certify this revision; regenerate affected arrays and create a new protocol with
the updated reviewed preprocessing hash. Do not reuse an old pretraining state
with a differently shaped prior or loosen bounds to force a run through.

The existing preprocessing helper has shared mutable output/provenance locations:
it fully validates case pairs before reusing them, otherwise regenerates/replaces
processed pairs, and rewrites mapping/split/progress/completion records. It never
changes the raw MRI files. The launcher archives the previous small preparation
JSON records in the new run before invoking it. It does not claim those shared
preparation files are immutable. Do not run another consumer/preprocessor against
these processed outputs concurrently; the helper's lock protects preparation,
not every downstream reader. The helper itself has no Slurm guard; the launcher
provides that guard and must be the entrypoint for this workflow.

Important artifacts under the new run include `SOURCE_CONFIG_FROZEN.json`,
`protocol.json`, `PROTOCOL_FROZEN.json`, `logs/`, and
`finetune/BRATS2023-40-Normal/UniME/benchmark_candidates/`.
Final outputs are `benchmark/SELECTION_FROZEN.json`, `test_selected.json`,
`test_common_45.csv`, `test_native_reference.csv`, and `TEST_COMPLETE.json`.
This launcher deliberately has no resume or overwrite mode. A failure requires
inspection and a newly authorized recovery plan; do not silently reuse its run
directory or launch a replacement job.
