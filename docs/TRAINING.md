# Training

Testing trained checkpoints does not require training. A new task-training run
has two stages: train the 3D branch, then use its frozen checkpoint to prepare
inputs for DentalPSAM. SAM initialization remains pretrained rather than random.

**Status: reference/reconstructed training implementation.** Some settings have
historical source evidence; others are deliberate corrections or new-run choices.
This is not a byte-for-byte replay of the training run underlying the paper.
See the [protocol comparison](#relationship-to-the-paper-experiments) below.

## Prerequisites

Prepare participant-disjoint `data/train` and `data/val` splits using the
[dataset layout](DATASET.md). Each contains preprocessed meshes and matching
annotations. Place the SAM ViT-B initialization at `checkpoints/sam.pth`;
`--sam-checkpoint` can override this location. Weights are not bundled or
automatically downloaded. Use new output paths outside the input data tree.

## 1. Train the 3D branch

```bash
python train.py --stage 3d --data data --output outputs/3d
```

The selected checkpoint is `outputs/3d/best.pth`. The training defaults
are 50 epochs, batch size 1, Adam at `1e-4`, weight decay `1e-5`, NLL loss,
ReduceLROnPlateau, early-stop patience 12, and seed 42. Selection uses the mean
equal-triangle plaque IoU over validation meshes, with strict probability
`> 0.5`. Participants remain separated between training and validation.
Use `--epochs 50 --patience 50` to run all 50 epochs without early stopping.

The model returns `[1, faces, 2]` log probabilities. NLL is evaluated over the
two classes for each original face; repeated padding rows do not contribute.
This is the current training objective, not a replay of the historical
Dice-Focal trainer. The model architecture and input normalization are retained.

`configuration.json` records the declared split and options before training.
`training_log.csv` records each epoch. `best.pth` and `last.pth` contain model,
optimizer, scheduler, and random-generator state; inference loads only the model
state. Automatic training resume is not currently exposed by the CLI.

## 2. Train DentalPSAM

```bash
python train.py --stage dentalpsam --data data \
  --branch-checkpoint outputs/3d/best.pth --output outputs/dentalpsam
```

The command prepares views and frozen 3D features automatically in a separate
`outputs/dentalpsam_inputs` directory, then starts training. It does not overwrite
source data or change the 3D checkpoint. `--branch-checkpoint` is required;
existing prepared inputs are not reused by the public training workflow.

Existing files are not proof that a prepared cache is correct. After changing
the 3D checkpoint, or when label-cache provenance is uncertain, regenerate
both feature and target files in a new directory using the
[mesh export command](../dentalpsam/branch3d/README.md#regenerate-prepared-mesh-files).
Keep the same split lists and the original PLY annotations.

Defaults are 50 epochs, batch size 4, Adam at `1e-4`, zero weight decay, StepLR
at epoch 40 with gamma 0.1, seed 42, 2D Dice-CE weight 2, and mesh BCE weight 1.
There is no early stopping. Mesh BCE uses continuous annotation values from
`mesh_labels`, not the standalone 3D classifier's binary targets. Zero-padded
rows are excluded from mesh loss, and
training and validation patches require at least 50 positive label pixels;
test prediction keeps all patches. Validation mesh
BCE selects `outputs/dentalpsam/best_model.pth`.

The default mesh fusion variant is gated; `--mesh-fusion concat` defines a
different checkpoint architecture. Frozen model loading infers the variant
from checkpoint keys and is strict. Do not select a variant from test outcomes.

## Weight bundles

Use the three files from this training run as one weight bundle:

| Bundle filename | Training output or initialization |
| --- | --- |
| `checkpoints/dentalpsam.pth` | `outputs/dentalpsam/best_model.pth` from stage 2 |
| `checkpoints/branch3d.pth` | `outputs/3d/best.pth` used to prepare stage 2 inputs |
| `checkpoints/sam.pth` | The same SAM ViT-B initialization used in stage 2 |
| `checkpoints/manifest.json` | `outputs/dentalpsam/manifest.json`, generated after stage 2 |

Copy these files into a new bundle directory with the indicated filenames,
then run `python test.py --data data/test --weights checkpoints --output results`.
Keep the original training outputs. Do not substitute a different 3D checkpoint
after DentalPSAM has been trained on its features.

The manifest declares `schema_version: 1`, `model: DentalPSAM`, SHA-256 values
under `dentalpsam_sha256`, `branch3d_sha256`, `sam_sha256`, and `mesh_fusion`.
It also records `provenance.kind` and `provenance.reference`. New runs use
`training_checkpoint`: each saved DentalPSAM checkpoint embeds the input-weight
hashes and preparation-record hashes. The public test command compares all
three files and the checkpoint architecture before preparing any inputs.
An embedded training binding is checked even if a manifest claims a different
provenance kind; changing the manifest cannot override that binding.

For historical weights without an embedded binding, only an author-supplied
`author_verified` manifest with a documented pairing reference is accepted.
The software cannot reconstruct missing training history from filenames, and
does not automatically certify arbitrary historical combinations. These hashes
check consistency with a trusted declaration, not digital authorship or model
quality. Do not create or alter a manifest merely to bypass a mismatch.

Advanced explicit checkpoint paths require `--bundle-manifest`,
`--branch-checkpoint`, and `--sam-checkpoint`; they do not bypass verification.

## Options and scope

```bash
python train.py --stage 3d --help
python train.py --stage dentalpsam --help
```

Use `--device cuda:1` to choose a GPU. The 3D stage enables deterministic
settings; use `--deterministic` for DentalPSAM. Exact bitwise identity across
PyTorch versions or GPUs is not promised.

Validation selects the checkpoint; test data are not used by either trainer.
The 3D validation rule is aligned with the paper evaluator for new training
runs. This does not change predictions from existing checkpoints or the
DentalPSAM stage-2 trainer. See the [package guide](../dentalpsam/README.md)
for model contracts and the tested environment.

## Relationship to the paper experiments

The MICCAI camera-ready manuscript describes 400 scans from 200 participants,
a 220/60/120 train/validation/test split, SAM ViT-B, Adam at `1e-4`, and batch
size 4. It does not specify every branch's training setting or identify an
immutable source revision and weight bundle. The following comparison uses the
retained SAMoral source archive and the saved 3D trainer/log, not comments from
later workspace guides. Source availability does not prove which file revision
produced every historical checkpoint.

| Setting | Historical evidence | Current reference implementation |
| --- | --- | --- |
| 3D optimizer | Saved trainer: Adam, `lr=1e-3`, weight decay `1e-5` | Adam, `lr=1e-4`, weight decay `1e-5`; changed learning rate |
| 3D loss | Saved trainer: equally weighted Dice and focal losses; class-axis mismatch found on real outputs | Per-face NLL on two classes; changed objective |
| 3D duration/schedule | Saved trainer: 200 epochs, StepLR every 20 epochs, gamma 0.5, LR floor `1e-5` | 50 maximum epochs; ReduceLROnPlateau; different schedule |
| 3D selection | Saved trainer/log: highest class-1 IoU on its declared evaluation cohort, without physical-area weighting | Highest validation mesh-macro equal-triangle plaque IoU, strict `>0.5`; participant-disjoint validation |
| 3D early stopping | No early-stop condition in the saved trainer | Patience 12 by default; `--patience 50` keeps a 50-epoch run |
| 3D augmentation | Saved trainer explicitly disables it | Disabled |
| DentalPSAM optimizer | Archived source: Adam, `lr=1e-4`, weight decay 0, batch 4 | Retained defaults; optimizer/lr/batch also stated in the paper |
| DentalPSAM duration/schedule | Archived source: 50 epochs; StepLR at 40, gamma 0.1 | Retained defaults; no early stopping |
| DentalPSAM joint loss | Archived source: `2 * DiceCE_2D + BCE_mesh`; DiceCE uses sigmoid and squared predictions | Retained weighting; mesh BCE now excludes padding |
| DentalPSAM selection | Archived source compares mean validation mesh BCE to its running minimum | Same criterion, but its numerical value changes with the padding correction and validation cohort |
| Patch selection | Archived loader requires at least 50 positive mask pixels in train and validation | Retained default; test inference keeps all patches |
| Random seed | No explicit deterministic seed setup in the archived task trainer | Seed 42; deterministic controls added for new runs |
| Mesh fusion variant | No immutable paper checkpoint-to-source mapping is available from the manuscript | Training defaults to gated; concat is explicit. Inference infers the variant from keys and verifies it against the bundle |
| Splits | Paper: 220/60/120; archived task trainer selects on a directory called External | Caller-declared participant-disjoint train/val; do not equate these automatically with the paper's split |

The earlier public trainer's participant-macro **area-weighted** selection was
a reconstructed setting, not supported by the saved historical 3D trainer.
It was replaced explicitly by equal-triangle validation for new runs. No model
architecture, checkpoint tensor key, normalization, or paper evaluator was
changed as part of that correction.

The archive and currently present server training file are different versions;
the latter already contains the padding correction and configurable paths.
Consequently, agreement with the current server file alone does not establish
agreement with the paper experiment. The exact paper split membership and the
original model/3D/SAM weight pairing still need run-specific evidence before a
historical training-reproduction claim can be made.
