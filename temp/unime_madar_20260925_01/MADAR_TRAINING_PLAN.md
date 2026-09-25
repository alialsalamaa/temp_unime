# UniME Madar training plan: from scratch

Latest explicit user decision: retrain from scratch, transferring scripts only.
This supersedes the earlier cached-pretraining/fresh-fine-tuning migration plan.

## Scientific workflow

1. Initialize UniME's pretraining stage from scratch and run 600 epochs.
2. Initialize the native fine-tuning stage from the newly produced pretraining
   checkpoint, with fresh optimizer/scheduler/EMA state, and run 600 epochs.
3. Preserve all 24 fine-tuning EMA selection candidates at the approved native
   validation epochs.
4. Select one checkpoint on the original 80 validation cases and all 15 nonempty
   modality combinations using the approved raw WT/TC/ET Dice selection rule.
5. Freeze that checkpoint and evaluate it on the 251 test cases and all 15
   combinations, recording all 45 region/combination results.

Preserve the approved paper-first architecture and hyperparameters, seed40,
batch4, crop96, 250 updates/epoch, preprocessing, labels, modality order, and
875 training /125 validation /251 test cases. The 45 remaining validation cases
remain part of native diagnostic validation, not an untouched lockbox. The
historical test set has previously been inspected and is not a new blind holdout.

Do not load the old Nebius pretrained weights or interrupted fine-tuning state.
At the user's explicit request, the old local checkpoint and temporary cleanup
backup were permanently deleted. They are not part of this scripts-only transfer
or from-scratch run. The original files on Nebius were not changed.

## Transfer and readiness

Transfer `unime_source_and_provenance.tar`, `TRANSFER_MANIFEST.json`, and this
training plan. No old weights or dataset should be uploaded from this folder.
The manifest contains weight/dataset metadata but no weight or dataset payload.
The source archive contains 118 code/configuration/provenance files, including
the frozen scientific source and historical evidence. Historical checkpoints and
smoke receipts are not authorization to reuse weights or skip new Madar checks.

The dataset ZIP is already downloaded and SHA-256 verified at `/work/vxc` on
Madar; it has not been extracted. Scripts have not yet been imported to Madar.
No training job has been submitted for this from-scratch plan.

Before submission: import/verify code, extract/verify cases, install a separate
training environment, resolve prepared-data storage, adapt only machine-specific
orchestration, and pass new Madar smoke/runtime checks. Keep the original frozen
source intact. Do not run the old Nebius cached-pretraining launcher unchanged.
Respect Madar's 72-hour GPU-job limit and the current one-GPU-job-at-a-time
authorization. The later permission to use extra resources was for downloading,
not an unrestricted change to the scientific training methodology.
