# Training

Run training only when task weights are unavailable or a new declared
experiment is required. Testing a frozen checkpoint needs no training.

## Stage 1: 3D branch

```bash
python train.py --stage 3d --data data --output outputs/3d_branch \
  --epochs 50 --patience 12 --learning-rate 1e-4 --weight-decay 1e-5 \
  --seed 42 --k 12 --device cuda:0
```

`data/train` and `data/val` each contain `origin/` and `label/` PLY pairs.
The preserved implementation uses batch size 1, NLL loss, Adam, and
ReduceLROnPlateau. Checkpoint selection uses participant-macro area-weighted
validation plaque IoU; `best.pth` is written to the new output directory.
This is a validation-only protocol, not an exact historical training replay.
It must not be confused with the equal-triangle test metric.

## Stage 2: DentalPSAM

```bash
python train.py --stage dentalpsam --data data \
  --3d-checkpoint outputs/3d_branch/best.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --output outputs/dentalpsam --epochs 50 --batch-size 4 \
  --learning-rate 1e-4 --seed 42 --deterministic --device cuda:0
```

With `--3d-checkpoint`, the command creates frozen inputs in a separate sibling
directory ending in `_inputs`; original data stay unchanged. Without this
option, `data/train/manual_2D` and `data/val/manual_2D` must already exist.
Explicit `--train-dir`/`--val-dir` may be used instead of `--data`.

The preserved defaults are Adam, zero weight decay, StepLR at epoch 40 with
gamma 0.1, 2D Dice-CE weight 2, mesh BCE weight 1, and no early stopping.
All-zero mesh padding rows are excluded. Training patches require at least
50 positive label pixels. The best checkpoint is selected by validation mesh
BCE and saved as `best_model.pth`.

`--mesh-fusion gated` is the default; `--mesh-fusion concat` defines a separate
architecture identity. Neither variant should be selected using test scores.

## Validation and boundaries

```bash
python scripts/dentalpsam/validate.py \
  --checkpoint outputs/dentalpsam/best_model.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --val-dir data/val/manual_2D --output outputs/validation.json
```

For regenerated inputs, pass the corresponding prepared validation directory.
Training/validation code is shared through the package, not through executable
scripts. Validation diagnostics retain their existing threshold and aggregation
rules; only the testing workflow is the original equal-triangle paper protocol.

The refactor does not alter optimizer settings, losses, seeds, or selection
rules to improve outcomes. Exact historical training and MICCAI result
reproduction remain unverified. See [Reproducibility](REPRODUCIBILITY.md).
