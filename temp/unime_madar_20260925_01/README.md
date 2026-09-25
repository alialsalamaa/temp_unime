# UniME: scripts-only handoff for training from scratch

This folder contains no model weights, dataset payload, authentication tokens,
or active transfer helpers. Cleanup and source verification completed on
2026-09-25. The frozen scientific code was not modified.

## Required contents

- `scripts/`: 118 verified source, configuration, split and provenance files.
- `unime_source_and_provenance.tar`: the same source tree packaged for transfer
  (1,167,360 bytes; no weights or dataset).
- `TRANSFER_MANIFEST.json`: original file/archive hashes. Its dataset/checkpoint
  entries are metadata only, not instructions to transfer those excluded assets.
- `MADAR_TRAINING_PLAN.md`: current from-scratch training and evaluation plan.
- `README.md`: this handoff guide.

For a compact private-repository upload, include the source TAR, transfer
manifest, and training plan. The extracted `scripts/` copy is provided for direct
inspection; uploading both it and the TAR is optional.

All 118 source files and the TAR were rechecked against the unchanged manifest
after cleanup. Historical source manifests, smoke receipts and launcher files
inside `scripts/` are retained as provenance. They do not mean the Nebius
launchers are ready to run on Madar or authorize reusing old weights.

## Training plan and current readiness

Run 600 pretraining epochs from random initialization, then 600 fine-tuning
epochs using the newly trained checkpoint. Preserve the approved architecture,
hyperparameters, preprocessing, case split and selection/test protocol. Do not
load the old Nebius pretraining or interrupted fine-tuning checkpoints.

Before training: import code to Madar, prepare the training environment and
machine-specific orchestration, verify storage and data preparation, and pass a
fresh Madar smoke/runtime test. No training job has been submitted.

## Dataset already on Madar

Source: Synapse `syn51514132`, version 1.
Archive: `/work/vxc/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`.
Size: 13,172,616,844 bytes.
SHA-256: `e1f6b0ff1320dcf2fc1d472cdf823190446783701f2dd705ac3b4c14a2a97d3f`.
CPU-only download job 1210742 completed successfully in 3 minutes 21 seconds;
size/hash matched the original Nebius archive. Extraction is still required.

## Cleanup

At the user's explicit request, the temporary cleanup backup was permanently
deleted, including the old local weights, logs, probes, and download helpers:
45 files totaling 2,069,796,852 bytes. No local recovery copy was retained.
The 118 required source files and source archive were verified again afterward.
No files on Nebius or Madar were changed by this local cleanup.
