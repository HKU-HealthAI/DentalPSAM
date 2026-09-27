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

The selected checkpoint is `outputs/3d/best.pth`. The retained training defaults
are 50 epochs, batch size 1, Adam at `1e-4`, weight decay `1e-5`, NLL loss,
ReduceLROnPlateau, patience 12, and seed 42. Selection uses participant-macro
area-weighted validation plaque IoU, not the equal-triangle test estimator.

## 2. Train DentalPSAM

```bash
python train.py --stage dentalpsam --data data \
  --branch-checkpoint outputs/3d/best.pth --output outputs/dentalpsam
```

The command prepares views and frozen 3D features automatically in a separate
`outputs/dentalpsam_inputs` directory, then starts training. It does not overwrite
source data or change the 3D checkpoint. When compatible prepared inputs already
exist, omit `--branch-checkpoint` to reuse them.

Defaults are 50 epochs, batch size 4, Adam at `1e-4`, zero weight decay, StepLR
at epoch 40 with gamma 0.1, seed 42, 2D Dice-CE weight 2, and mesh BCE weight 1.
There is no early stopping. Zero-padded rows are excluded from mesh loss, and
training patches require at least 50 positive label pixels. Validation mesh
BCE selects `outputs/dentalpsam/best_model.pth`.

The default mesh fusion variant is gated; `--mesh-fusion concat` defines a
different checkpoint architecture. Frozen model loading infers the variant
from checkpoint keys and is strict. Do not select a variant from test outcomes.

## Options and scope

```bash
python train.py --stage 3d --help
python train.py --stage dentalpsam --help
```

Use `--device cuda:1` to choose a GPU. The 3D stage enables deterministic
settings; use `--deterministic` for DentalPSAM. Exact bitwise identity across
PyTorch versions or GPUs is not promised.

These are the retained validation-only trainers, not a claim of exact historical
training replay. Cleanup does not modify losses, selection rules, normalization,
or the paper evaluator to improve scores. Detailed environment and compatibility
records are kept separately in [reproducibility](../reproducibility/REPRODUCIBILITY.md).
