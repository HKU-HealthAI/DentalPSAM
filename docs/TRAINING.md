# Training

Testing trained checkpoints does not require training. A new task-training run
has two stages: train the 3D branch, then use its frozen checkpoint to prepare
inputs for DentalPSAM. SAM initialization remains pretrained rather than random.

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
source data or change the 3D checkpoint. When compatible prepared inputs already
exist, omit `--branch-checkpoint` to reuse them.

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

## Test your trained model

Use the three files from this training run as one weight bundle:

| Bundle filename | Training output or initialization |
| --- | --- |
| `checkpoints/dentalpsam.pth` | `outputs/dentalpsam/best_model.pth` from stage 2 |
| `checkpoints/branch3d.pth` | `outputs/3d/best.pth` used to prepare stage 2 inputs |
| `checkpoints/sam.pth` | The same SAM ViT-B initialization used in stage 2 |

Copy these files into a new bundle directory with the indicated filenames,
then run `python test.py --data data/test --weights checkpoints --output results`.
Keep the original training outputs. Do not substitute a different 3D checkpoint
after DentalPSAM has been trained on its features.

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
