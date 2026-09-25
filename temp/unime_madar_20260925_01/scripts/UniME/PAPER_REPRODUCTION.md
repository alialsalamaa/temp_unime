# Local UniME paper-first configuration

Current source revision: user-approved six-item restoration, 23 September 2026.
Explicit paper settings take priority; released original-pipeline mechanics fill
specified gaps. This is not a byte-identical author reproduction. No GPU job is
submitted by these changes. The independent patch sampler was NOT included in
this restoration authorization and remains unchanged; it is not claimed to be
uniquely required by the paper's Bernoulli notation.

Based on upstream commit `88ef06d`. Changes target the original
`UniEncoder` → `UniME` pipeline. They implement the written settings in the
[CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Song_Uni-Encoder_Meets_Multi-Encoders_Representation_Before_Fusion_for_Brain_Tumor_Segmentation_CVPR_2026_paper.pdf)
and [supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Song_Uni-Encoder_Meets_Multi-Encoders_CVPR_2026_supplemental.pdf).
The paper/code discrepancies mean this is an implementation of the written
protocol, not confirmation of the exact recipe that produced the authors' scores.

## Environment

Current Nebius interpreter:
`/adialab/usr/khaled/SAMTOK_DIR/AVMUR/baselines/.venv-unime/bin/python`, with
PyTorch 2.8.0+cu126 and Python 3.11.16. The old-server instructions and observations
below are historical; they are not the current Nebius environment or a new
GPU-readiness result. Do not reinstall the old cluster environment on Nebius.

```bash
conda activate unime
```

Recreate with `conda env create -f environment-cluster.yml` from this directory.
The installed `unime` environment uses Python 3.11, PyTorch 2.6.0, torchvision
0.21.0 and CUDA 12.4. **PyTorch 2.6 is a compatibility exception to the paper's
2.8.0:** this server has glibc 2.26, while the official 2.8 CUDA wheels require
2.28. A conda-forge 2.8 CUDA install was also rejected because its CUDA library
dependencies require glibc 2.28. The operating system was not modified.

On a system with glibc >= 2.28, recreate the paper's framework version with
`conda env create -f environment-paper.yml` and `conda activate unime-paper`.
That specification pins PyTorch 2.8.0/torchvision 0.23.0 with CUDA 12.6 and has
not been runtime-tested here. Both environments share `requirements-common.txt`
and the same training parameters. CUDA runtimes come with the wheels; GPU nodes
must provide a compatible NVIDIA driver. Python 3.11 is a local choice.
Existing AVMUR environments are unchanged. The upstream `requirements.txt`
is retained for provenance and should not be installed over these environments.
`requirements-cluster-lock.txt` records the installed Python packages, including
transitive dependencies, for replay with `python -m pip install -r
requirements-cluster-lock.txt` after creating a Python 3.11 environment.

## Parameter and implementation changes

| Setting | Upstream original pipeline | Local paper configuration |
| --- | --- | --- |
| Pretraining peak LR | `5e-4` | `3e-4` |
| Pretraining minimum LR | `1e-5` | `1e-6` |
| Fine-tuning minimum LR | `1e-5` | `1e-6` |
| Warmup/cosine clock | Cosine clock includes warmup | Restored official clock: `t_initial=150000`, `warmup_prefix=False`, warmup 7500; paper LR values retained |
| Whole-modality masking | Uniformly drop 0 or 1 | Bernoulli `p=0.5`, conditioned on at least one available modality |
| Patch masking | Fixed number of visible patches | Existing independent draws, `q=0.75`; unchanged by this six-item restoration |
| Transformer layer LR | Extra block decay factor; compiled names could bypass grouping | Paper block `lr × 0.75^(L-l)`; official stem `0.75^(L+1)` and final norm 1; unwrap before grouping |
| Mask regularizer | Spatial-mean-centered L2 plus epsilon | Raw spatial L2, coefficient `0.005`; official `(norm + 1e-6).mean()` reduction restored |
| Short source dimensions | Foreground-interval crop; no padding | Official crop restored, followed by retained symmetric zero-padding to at least 128 |
| Effective learned prior | CLI `(160,180,210)` D/H/W | Restored `(160,180,210)`, explicit in original pretraining wrapper |
| Normalization brain mask | Sum of channels greater than zero | Union of nonzero channels |

The original architecture remains depth 16, embedding 864, 12 attention heads,
4 register tokens, patch size 8³ and CNN widths 16/32/64/128. Both stages retain
600 epochs, 250 iterations per epoch, batch size 4, crop size 96³, AdamW with
weight decay `1e-4`, and 5% linear warmup from `1e-5` before cosine decay.

The retained modality sampler enumerates allowed subsets using conditional
independent-Bernoulli probabilities. It implements probability `p=0.5` and a
nonempty constraint, without reverting to the conflicting released cap of one
masked modality. All 15 nonempty combinations are equally likely under this
implementation. Independence and enumeration are implementation choices, not
literal paper instructions. The legacy cap remains available; the original
pretraining wrapper sets it to 3. Lowering it changes the protocol.

The pretraining prior now uses the official effective CLI dimensions
`(160, 180, 210)` in D/H/W order. This is a code-derived value, not a paper-stated
hyperparameter. Training headers and the benchmark/split checks enforce these
bounds before training; a failure must not silently enlarge the prior. The old
server's full-dataset report measured maximum processed dimensions `(155,177,208)`,
which fit, but regenerated Nebius inputs still require verification. The
restored short-axis crop changes content/placement, not the minimum-128 shape.
Use a fresh pretraining run; a prior tensor of a different size is not compatible
with resuming the old full pretraining/optimizer state. Only encoder weights,
not this prior, transfer into the unchanged segmentation architecture.

The raw mask penalty retains upstream averaging of per-sample/per-modality
spatial norms and restores the added epsilon `1e-6`; the paper does not specify
these details. The norm is over the input crop, not the complete prior grid.
The pretraining
parser now defaults to LR `3e-4` and crop 96, while the launchers explicitly
declare all central training values.

The released scheduler clock is intentionally retained with paper LR values.
Its discrete transition is slightly below the nominal peak at update 7500;
neither an exact peak at that update nor exact minimum on the final optimizer
update is promised. Padding remains constant-zero after normalization, with an
odd surplus voxel at the end; these retained mechanics are not specified by the
supplement. The crop branch itself now matches the released source.

## Preserve the study split

Run preparation once, from this directory, before training:

```bash
python scripts/brats23_process.py \
  --raw-dir ../data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --split-json ../artifacts/splits/official_imfuse_split.json \
  --write-mapping
```

This preserves the existing 875/125/251 memberships through the generated
case-name mapping. It validates coverage, duplicates, and patient overlap
before processing. Default upstream random-split generation is still available
when no split JSON is supplied; it is not the command for this comparison.
The processed NumPy data use channel order FLAIR/T1c/T1/T2. Existing processed
arrays from the old preprocessing require `--overwrite` to receive the padding
and normalization changes.

## Launch after data preparation

```bash
conda activate unime
bash scripts/pipeline/train_original.sh
```

The pipeline calls the same `scripts/pretrain.sh` and `scripts/finetune.sh`
used for separate stages. Shell errors stop the pipeline, and missing pretrained
weights fail explicitly. Stage outputs default to
`log_pretrain_paper/` and `log_paper/`, keeping them separate from upstream runs.
Both stages use seed 40 and a consistent checkpoint path. The paper does not
specify this seed. W&B is disabled by default.

Optional shell settings: `UNIME_PYTHON`, `UNIME_GPU_IDS`, `UNIME_DATA_PATH`,
`UNIME_SEED`, `UNIME_PRETRAIN_LOG_ROOT`, `UNIME_FINETUNE_LOG_ROOT`,
`UNIME_PRETRAIN_CHECKPOINT`, and `UNIME_WANDB_MODE`. Use new output roots for
independent seeds or experiments. The standalone scripts accept additional CLI
arguments. Keep effective global batch size 4 when adapting to distributed
execution; the repository's batch argument is per process.

EMA (`0.999`), the existing augmentation choices, and full-modality validation
selection are retained because the paper does not fully specify alternatives.
Validation still averages WT/TC/ET/postprocessed-ET as in upstream code. During a
short pretraining smoke run, set `--ema_validate_start_epoch 1`; its normal
value of 500 will not produce an EMA checkpoint in a short run.

## Metrics and checks

The ordinary native evaluation path (without opt-in benchmark mode) retains
released behavior. The default `--top_k 3` keeps the three best checkpoints according to
the full-modality validation composite score, the mean Dice of WT, TC, raw ET,
and postprocessed ET. The final-epoch checkpoint is also tested if it is not
already among those three, so the usual test queue contains three or four
distinct checkpoints.

Each queued checkpoint is evaluated separately across all 15 nonempty modality
combinations. Dice tables report WT, TC, raw ET, and postprocessed ET, including
the average across modality combinations. Postprocessing sets predicted ET to
empty when it contains fewer than 500 voxels. Inference retains 50% overlap
and probability averaging. No HD95 calculation or HD95 CSV output is included.

For the approved Nebius comparison, use [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md):
original80/all15 raw-Dice selection and one frozen winner on the251 historical
test cases. Native multi-checkpoint testing is disabled in that mode. Source
and preprocessing changes invalidate old protocol receipts; create fresh
processed-data provenance and a new protocol after verification, never edit a
frozen old protocol to bypass its hash checks.

Run the focused checks in the new environment:

```bash
python -m unittest discover -s tests -p 'test_paper_*.py' -v
python -m pip check
python pretrain.py --help
python main.py --help
```

Historical old-server verification: all 32 then-retained focused tests passed, `pip check` found no broken
requirements, and a small UniEncoder completed a CPU forward/backward/AdamW
step (two 16³ inputs, embedding 72, depth 2). This smoke configuration only
checks model/library integration; the launchers retain the full paper model.
The login node has no usable CUDA device. GPU execution, full data preparation,
and full training have not been run.
