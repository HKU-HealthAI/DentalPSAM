# DentalPSAM

DentalPSAM performs dental-plaque segmentation by combining three rendered
2D views with per-face 3D mesh features. This repository contains the main
model, TSGCNet feature generator, training and validation entry points,
fixed-split prediction export, and the original equal-triangle MICCAI evaluator
with participant-clustered 95% confidence intervals. Physical-area evaluation
is a separate analysis, not the default paper-reproduction metric.

This code snapshot deliberately excludes the later stitching model, patient
data, checkpoints, saved predictions, and result artifacts.

## What is included

```text
dentalpsam/
  model.py                 checkpoint-compatible DentalPSAM architecture
  data.py                  three-view image/mesh-patch loading
  checkpoints.py           strict SAM and DentalPSAM checkpoint loading
  mesh_targets.py          padded-row-safe mesh supervision
  prediction_io.py         face-order reconstruction helpers
  evaluation.py            original equal-triangle metric protocol
  uv_projection.py         historical planar/cylindrical UV mathematics
  segment_anything/        modified vendored SAM implementation
tsgcnet/
  model.py                 colour-aware TSGCNet architecture
  data.py                  PLY features and 16,000-face input contract
  utils.py                 graph-neighbour and compatibility utilities
  augmentation.py          optional geometric augmentation
prepare_uv_views.py        PLY -> three PNG views + info NPZ
train_tsgcnet.py           patient-disjoint TSGCNet training
validate_tsgcnet.py        frozen-checkpoint area-weighted validation
export_tsgcnet_features.py TSGCNet -> SOTA_mesh + label_mesh NPZ
train_dentalpsam.py        DentalPSAM training (default 50 epochs)
validate_dentalpsam.py     frozen-checkpoint validation diagnostics
predict_dentalpsam.py      fixed-list 2D and 3D probability export
evaluate_dentalpsam.py     original MICCAI evaluation (default protocol)
tools/preflight_split.py   read-only manual_2D contract check
tools/evaluate_mesh.py     face/area/vertex metrics, CIs, paired tests
tests/                     synthetic tests; no patient data
docs/                      pipeline, provenance, and result boundaries
```

The old `SAMoral/train.py`, `SAMoral/pred2D_jhn.py`, TSGCNet
`pred_to_2d_new.py`, and hard-coded evaluation scripts are not public entry
points. Their inspected SHA-256 values and the deliberate changes are recorded
in [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md).

## Installation

Server checks use Python 3.9.18 in the `py39_rt2` environment. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for tested package versions.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the official SAM ViT-B checkpoint separately. This repository does
not redistribute it or any trained DentalPSAM/TSGCNet weights.

## End-to-end data flow

```text
origin/label PLY
      |
      | prepare_uv_views.py
      v
three-view origin/label PNG + info NPZ
      |
      | export_tsgcnet_features.py + frozen TSGCNet checkpoint
      v
SOTA_mesh NPZ [XYZ + P(plaque)] + label_mesh NPZ [XYZ + target]
      |
      | train/validate/predict DentalPSAM
      v
2D probability maps + 3D per-face probabilities
      |
      | evaluate_dentalpsam.py
      v
mesh-macro equal-triangle metrics + participant-clustered 95% CI
```

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for exact array keys,
shapes, label direction, face ordering, and padding semantics.

## 1. Prepare three UV views

Start from topology-matched, already preprocessed PLY files and a fixed list
with one mesh ID per line. Write to a new derived directory; the script never
edits the input PLYs.

```bash
python prepare_uv_views.py \
  --origin-dir /data/split/origin \
  --label-dir /data/split/label \
  --mesh-list /protocol/split_mesh_ids.txt \
  --output-dir ./derived/split_uv
```

This generates `origin/*.png`, `label/*.png`, and `info/*.npz`. Each mesh has
22 patches: 6 upper, 8 inner, and 8 outer. The command hashes all inputs and
writes `projection_manifest.json`.

The server PLY files were already gum-removed and reduced to at most 16,000
faces. That upstream preprocessing code is not included here; this command
does not downsample or uniformly remesh a mesh.

## 2. Train or validate TSGCNet

Training is not required when a frozen compatible checkpoint already exists.
The clean trainer accepts only train and validation directories and selects by
validation participant-macro area-weighted plaque IoU.

```bash
python train_tsgcnet.py \
  --train-dir /data/internal/train \
  --val-dir /data/internal/val \
  --out-dir ./outputs/tsgcnet_train \
  --epochs 50 --patience 12 \
  --learning-rate 1e-4 --weight-decay 1e-5 \
  --plaque-class-weight 1.0 --seed 42 --k 12 --device cuda:0
```

Validate a selected checkpoint without changing it:

```bash
python validate_tsgcnet.py \
  --checkpoint /checkpoints/tsgcnet.pth \
  --val-dir /data/internal/val \
  --output ./outputs/tsgcnet_validation.json \
  --device cuda:0
```

The original server trainer evaluated the external cohort during training and
selected its best model using external label-1 IoU. That behavior is not used
here because it cannot support an independent external-validation claim.

## 3. Generate `SOTA_mesh` and `label_mesh`

Use the PLY dataset root together with the UV `info` directory. The output is
always a fresh derived directory.

```bash
python export_tsgcnet_features.py \
  --checkpoint /checkpoints/tsgcnet.pth \
  --data-dir /data/split \
  --info-dir ./derived/split_uv/info \
  --mesh-list /protocol/split_mesh_ids.txt \
  --out-dir ./derived/split_tsgcnet \
  --device cuda:0 --k 12
```

The exporter creates:

- `SOTA_mesh/<mesh>.npz`: columns 0--8 are the three triangle vertices; column
  9 is TSGCNet class-1 `P(plaque)`;
- `label_mesh/<mesh>.npz`: identical geometry; column 9 is 1 for a plaque face
  and 0 otherwise;
- `scores/<mesh>.npy`: one class-1 probability per original PLY face;
- `SOTA_pred/<mesh>_<view>.png`: compatibility raster only;
- `export_manifest.json`: checkpoint, list, input, source, and output hashes.

Unlike the historical `pred_to_2d_new.py`, this command never replaces model
scores with ground truth, never writes into raw data, and never silently skips
a requested mesh. It verifies that the three UV views form an exact partition
of all original PLY faces.

Create a fresh `manual_2D` directory using derived files (copies or read-only
symlinks are both acceptable):

```text
manual_2D/origin     -> derived/split_uv/origin
manual_2D/label      -> derived/split_uv/label
manual_2D/info       -> derived/split_uv/info
manual_2D/SOTA_mesh  -> derived/split_tsgcnet/SOTA_mesh
manual_2D/label_mesh -> derived/split_tsgcnet/label_mesh
```

Validate it before model use:

```bash
python tools/preflight_split.py \
  --data-dir ./derived/manual_2D \
  --participant-id-prefix-length 4 \
  --out ./outputs/split_contract.json
```

## 4. Train DentalPSAM

The defaults implement the cleaned 50-epoch baseline protocol: batch size 4,
Adam at `1e-4`, no weight decay, seed 42, 2D Dice-CE weight 2, valid-face 3D
BCE weight 1, and checkpoint selection by validation mesh BCE. Training and
validation participants must be disjoint.

`--mesh-fusion gated` is the historical 397-key architecture (default);
`--mesh-fusion concat` is the newer 403-key server architecture. Inference
selects the architecture from checkpoint keys and loads it strictly. These
variants must retain separate experiment identities.

```bash
python train_dentalpsam.py \
  --train-dir /derived/internal_train/manual_2D \
  --val-dir /derived/internal_val/manual_2D \
  --sam-checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --save-dir ./outputs/dentalpsam_train \
  --epochs 50 --batch-size 4 --learning-rate 1e-4 \
  --image-loss-weight 2 --mesh-loss-weight 1 \
  --target-threshold 0.5 --seed 42 --deterministic --device cuda:0
```

All-zero padded mesh rows are excluded from loss and IoU. The historical
positive-patch filter is retained (`--min-positive-pixels 50`) and both
training and prediction binarize 2D labels with the same `gray > 127` rule.

Validate a selected checkpoint:

```bash
python validate_dentalpsam.py \
  --checkpoint /checkpoints/dentalpsam.pth \
  --sam-checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --val-dir /derived/internal_val/manual_2D \
  --output ./outputs/dentalpsam_validation.json \
  --device cuda:0
```

Validation output is a checkpoint-selection diagnostic. It is not a held-out
participant-level result.

## 5. Export fixed-split predictions

```bash
python predict_dentalpsam.py \
  --checkpoint /checkpoints/dentalpsam.pth \
  --sam-checkpoint /checkpoints/sam_vit_b_01ec64.pth \
  --data-dir /derived/external/manual_2D \
  --mesh-list /protocol/external_mesh_ids.txt \
  --expected-count 120 \
  --output-dir ./outputs/dentalpsam_external \
  --target-threshold 0.5 --device cuda:0
```

The command produces continuous 2D probability maps at the output root and
continuous per-face 3D probabilities in `3Dpred/`. It records checkpoint and
split hashes in `prediction_run_manifest.json`. The printed branch metrics are
diagnostics only.

## 6. Evaluate using the original MICCAI protocol

The MICCAI manuscript uses 220/60/120 meshes for training/validation/testing.
This scope excludes stitching: the corresponding ablation row reports plaque
IoU **0.547**, Dice **0.700**, and OA **0.830**. These are reference targets,
not an assertion that the packaged code has already reproduced them. The full
model row includes stitching and is not the current reproduction target.

Evaluate saved predictions without retraining:

```bash
python evaluate_dentalpsam.py \
  --data-dir /data/external \
  --prediction-dir ./outputs/dentalpsam_external \
  --mesh-list /protocol/external_mesh_ids.txt \
  --expected-count 120 \
  --output-dir ./outputs/external_original_evaluation \
  --fusion-weight-2d 0.5 --threshold 0.5 \
  --bootstrap-reps 10000 --seed 42
```

This command preserves the inspected fixed-fusion evaluator's semantics:

- each triangle has equal weight; no triangle-area weighting;
- three views reproject to faces using UV-centre truncation, without resizing;
- the 2D scores come from saved 8-bit PNGs, as in the historical script;
- plaque decisions use strict `score > 0.5`;
- fixed fusion is `0.5 * score_2d + 0.5 * score_3d`;
- metrics are computed per mesh, then averaged over meshes;
- 95% CIs resample participants together with all of their meshes.

The output contains `per_mesh_metrics.csv`, `summary.json` (2D, 3D, and fusion
metrics with CIs), and `evaluation_manifest.json` with input/source hashes.
Missing predictions, face-count mismatches, duplicated faces, and incomplete
UV partitions are errors; cases are never silently dropped or truncated.
Ground-truth-dependent adaptive fusion is not used.

## Optional: physical-area analysis and paired comparisons

Copy and edit `examples/evaluation_spec.json`; every method in one comparison
should be evaluated in the same invocation. This separate evaluator uses its
own recorded projection and aggregation rules. Even its `face` row must not
be substituted for the original protocol above.

```bash
python tools/evaluate_mesh.py \
  --data-dir /data/external \
  --mesh-list /protocol/external_mesh_ids.txt \
  --spec-json examples/evaluation_spec.json \
  --out-dir ./outputs/external_area_evaluation \
  --target-source label_ply_any_exact_black_vertex \
  --target-threshold 0.5 \
  --bootstrap-reps 10000 --permutation-reps 100000 \
  --seed 20260810 --primary-method DentalPSAM
```

Outputs include:

- `per_case_metrics.csv`;
- `per_patient_metrics.csv`;
- `patient_bootstrap_95ci.csv`;
- `paired_sign_flip_tests.csv` with Holm-adjusted p-values;
- `evaluation_manifest.json` binding all inputs and outputs by SHA-256.

Use `unit=area` for area-weighted segmentation metrics and the explicitly
named coverage fields for physical tooth-surface coverage. `unit=face`
assigns equal weight to every triangle and can change with mesh density;
`unit=vertex` is a separate sensitivity analysis. The estimator is the macro
mean over participants after combining their `01` and `02` arches.

## Paper values versus reproducible values

[docs/PAPER_RESULTS.md](docs/PAPER_RESULTS.md) records the MICCAI manuscript tables
as a reference target. Those legacy values are not presented as regenerated by
this commit and must not be combined with confidence intervals from a new
cohort or evaluator. A current result requires the fixed split, exact
checkpoint hashes, prediction manifest, and evaluation manifest from the same
run.

## Verification

See [docs/SERVER_VALIDATION.md](docs/SERVER_VALIDATION.md) for actual server
checks, source/checkpoint hashes, and remaining full-cohort reproduction gaps.

The following checks use only synthetic fixtures:

```bash
python -m compileall -q .
python tests/test_prediction_io.py --require_numpy
python tests/test_preflight_split.py
python tests/test_evaluate_mesh.py
python tests/test_mesh_targets.py
python tests/test_original_evaluation.py
```

For a GPU environment with all runtime dependencies installed, also check the
CLI surfaces:

```bash
python prepare_uv_views.py --help
python train_tsgcnet.py --help
python validate_tsgcnet.py --help
python export_tsgcnet_features.py --help
python train_dentalpsam.py --help
python validate_dentalpsam.py --help
python predict_dentalpsam.py --help
python evaluate_dentalpsam.py --help
python tools/evaluate_mesh.py --help
```

## Release and license boundary

The source code is ready for review, but this commit is not a formal model or
dataset release. Before creating a release tag, the repository owner must add
the final DentalPSAM project license, confirm TSGCNet redistribution terms,
publish checkpoint URLs plus SHA-256 values, and document dataset access and
participant manifests. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The inspected compute and determinism details are in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

For version control, artifact boundaries, and evidence-linked commits, see
[CONTRIBUTING.md](CONTRIBUTING.md).

Run `python tools/check_git_payload.py` after staging to inspect the Git index
for private artifacts, oversized files, and common credential patterns.
