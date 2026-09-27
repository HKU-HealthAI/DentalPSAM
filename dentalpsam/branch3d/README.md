# 3D branch

This module learns per-triangle plaque scores from dental mesh geometry,
normals, and scan colours. Its checkpoint supplies the frozen 3D inputs used
when training and testing DentalPSAM.

Start with [Training](../../docs/TRAINING.md) for the two-stage workflow or
[Dataset](../../docs/DATASET.md) for the input layout. You do not need to run
any module in this directory as a separate command.

## Train and use the checkpoint

From the repository root:

```bash
# Stage 1: learn the mesh representation.
python train.py --stage 3d --data data --output outputs/3d

# Stage 2: prepare inputs with that frozen checkpoint and train DentalPSAM.
python train.py --stage dentalpsam --data data \
  --branch-checkpoint outputs/3d/best.pth --output outputs/dentalpsam
```

Stage 1 selects `outputs/3d/best.pth` using validation data only. Stage 2 also
requires SAM initialization at `checkpoints/sam.pth` and generates the 3D
inputs automatically. Original meshes and annotations are not overwritten.

For testing a trained model, put its matching weights in the bundle described
in the [main README](../../README.md#test), then run:

```bash
python test.py --data data/test --weights checkpoints --output results
```

`branch3d.pth` must be the checkpoint associated with the DentalPSAM weights;
substituting another checkpoint changes the model inputs.

## Inputs and outputs

One sample is a topology-matched pair of coloured scan and annotation PLY
meshes. The loader reads scan colours from the scan, not from the annotation.
Annotations provide targets: a face's per-channel minimum vertex RGB must be
`[0, 0, 0]` to denote plaque (class 1); otherwise its class is 0. Intermediate
grayscale values are valid annotations. For grayscale annotations, the rule
marks a triangle as plaque when any of its vertices is exactly black.

The loader constructs 33 features per face. Column ranges below are Python
slices, with an exclusive right endpoint.

| Columns | Feature | Used directly by the model |
| --- | --- | --- |
| `0:9` | XYZ coordinates of the triangle's three vertices | Coordinate stream |
| `9:12` | Triangle centre | Loader centering; not a separate model stream |
| `12:21` | Normal vectors at the three vertices | Normal stream |
| `21:24` | Per-channel minimum scan RGB | Retained loader field; not consumed by `forward()` |
| `24:33` | Scan RGB at the three vertices | Colour stream |

The model receives float features `[1, 33, 16000]` on its device and face
indices `[1, 16000, 3]` on the CPU. Short meshes repeat their final face to
16,000 rows, preserving original face order. This is padding, not downsampling
or uniform remeshing; prepare meshes with at most 16,000 faces upstream.
Training loss, exported scores, and evaluation exclude repeated padding rows.

[data.py](data.py) owns the checkpoint-sensitive centering and scaling.
Centering includes the repeated faces, and the retained scalar scale is
computed from the original vertex record, including colour channels. Do not
replace this with a different normalization when loading existing weights.

## What the network does

[model.py](model.py) transforms the coordinate, normal, and colour features
in three streams. It gathers neighbouring faces through mesh connectivity,
aggregates their features, and combines the streams to classify each face.
The default neighbourhood width is 12; inference uses batch size 1.

The output is **log probabilities** with shape `[1, 16000, 2]`.
`output[0, :original_face_count, 1].exp()` gives `P(plaque)` in original face
order. Do not treat the output as logits for a sigmoid, swap the two classes,
or include repeated rows in the score.

The internal class retains the name `TSGCNet` for checkpoint compatibility;
the public workflow calls this component the **3D branch**.

## How DentalPSAM uses it

[features.py](features.py) runs the frozen checkpoint in evaluation mode and
uses UV metadata to place each original face into its image patch. Each
exported row contains nine original triangle-coordinate values followed by
one plaque probability. These `[N, 10]` rows are distinct from the 33-feature
input used by the standalone mesh model.

Feature probabilities and annotation targets are stored separately. The
exporter records face-order indices and checks that the three views cover
every original face exactly once. The DentalPSAM loader then left-pads each
patch to 6,000 rows without changing its order.

Unlike this classifier's binary labels, exported annotation targets preserve
`1 - min(vertex RGB / 255, per channel)[0]`, including intermediate grayscale
values. DentalPSAM uses them directly as continuous targets in its mesh BCE.
The paper evaluator retains its separate binary face-label rule.

### Regenerate prepared mesh files

Training and testing normally prepare these inputs automatically. To rebuild
them separately for an existing split with UV metadata:

```bash
python -m dentalpsam.branch3d.features \
  --data data/train --checkpoint checkpoints/branch3d.pth \
  --output outputs/prepared_train --device cuda:0
```

The default fixed list is `data/train/mesh_ids.txt`; `--mesh-list` overrides it.
Repeat with `data/val` and `data/test`, keeping their original membership.
Both the public dataset layout and the historical `origin/label/manual_2D`
layout are accepted. Existing rendered views and UV metadata are reused;
no annotation or input file is overwritten.

The fresh output contains `label_mesh/*.npz` (annotation-derived continuous
targets), `SOTA_mesh/*.npz` (model-derived plaque probabilities),
`SOTA_pred/*.png` (auxiliary visualizations), and `scores/*.npy` (unpadded scores
in original face order). Each NPZ contains `up/in/out` patch rows and their
`face_order_up/in/out` arrays. Both NPZ files are reopened and checked against
their own source values before the export is accepted. `export_manifest.json`
records the checkpoint, source/input hashes, output hashes and verification.

For source compatibility, patch assignment retains upper-bound-only UV
clipping and Python negative indexing. Auxiliary rasterization truncates
coordinates and pixel intensity. These details affect cached inputs: do not
replace them with lower-bound clipping or rounding for an existing checkpoint.
Predictions use the source exporter's class-1 softmax conversion in evaluation
mode; ground-truth targets never pass through the model. Do not populate both
NPZ files from the same prediction array.

The auxiliary scores generated here are **not** DentalPSAM's final 3D output.
DentalPSAM's [MeshDecoder](../model.py) also uses 2D appearance features and
produces its own mesh logits. The final evaluator fuses those predictions
with back-projected 2D predictions; it does not fuse the auxiliary input scores.

## Training and evaluation

The stage-1 defaults are 50 maximum epochs, batch size 1, Adam with learning
rate `1e-4` and weight decay `1e-5`, NLL loss, and seed 42. Augmentation is
disabled. Validation participant-macro area-weighted plaque IoU selects the
checkpoint; early stopping uses patience 12. These validation scores are not
the equal-triangle test metrics produced by the public `test.py` command.

Training writes `best.pth`, `training_log.csv`, and `manifest.json` to the new
output directory. `python train.py --stage 3d --help` lists the available options.

## Source map

- [data.py](data.py): PLY loading, targets, 33-feature construction, and normalization.
- [model.py](model.py), [utils.py](utils.py): model layers and neighbourhood gathering.
- [training.py](training.py): participant-disjoint training and checkpoint selection.
- [features.py](features.py): frozen-checkpoint scores and aligned DentalPSAM inputs.
- [validation.py](validation.py): validation-only area-weighted diagnostics.
- [reporting.py](reporting.py): standalone fixed-checkpoint equal-triangle evaluation.
