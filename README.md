# DentalPSAM

DentalPSAM combines rendered 2D views and 3D mesh features for dental-plaque
segmentation. This repository contains the main model and its data preparation,
training, prediction, and evaluation code. The stitching module is not included
in this version.

## Dataset

The dataset is being organized and is **not yet publicly available**.
Please reach out to the authors for data access. Access conditions and release
details will be provided when the dataset is ready; this repository does not
currently distribute patient data or a public download link.

The code expects preprocessed, gingiva-removed meshes with matching scan and
annotation geometry. A prepared split has the following structure:

```text
data/
  train/
    origin/                 Colored PLY meshes
    label/                  Plaque annotations on matching meshes
    manual_2D/              Rendered views and aligned mesh features
  val/
    origin/
    label/
    manual_2D/
  test/
    origin/
    label/
    manual_2D/
splits/
  test.txt                  One mesh identifier per line
```

Participants must not overlap between training, validation, and test splits.
The data format and preparation procedure are described in
[Data preparation](docs/DATA_PIPELINE.md). Generated files are stored separately
from original data. Gingival removal, downsampling, and uniform remeshing are
not performed by the code in this repository.

## Code

The reference environment is Linux with Python 3.9.18, PyTorch 2.0.1, and
CUDA 11.7. An NVIDIA GPU with a compatible driver is required for GPU execution.

```bash
git clone https://github.com/HKU-HealthAI/DentalPSAM.git
cd DentalPSAM
conda env create --file environment.yml
conda activate dentalpsam
```

The environment specification pins direct dependencies to the versions used
for verification. Fresh-environment installation has not yet been verified.
Run the examples below from the repository root. To check dependencies and
synthetic regression tests, run `bash tests/run_checks.sh`.

### Testing

Testing uses fixed pretrained weights and prepared 3D branch inputs. No
additional training is required.

Place your compatible weights in `checkpoints/`:

```text
checkpoints/
  sam_vit_b_01ec64.pth       SAM ViT-B initialization
  dentalpsam.pth            Trained DentalPSAM weights
  3d_branch.pth             Only needed when generating mesh features from PLY
```

Public links for the task-trained weights are not yet available. A SAM
checkpoint alone is insufficient. If `manual_2D/` is not already prepared,
follow [Data preparation](docs/DATA_PIPELINE.md#prepare-a-split) first; this
runs the fixed 3D feature branch, without retraining it.

Generate 2D and 3D predictions:

```bash
python predict_dentalpsam.py \
  --checkpoint checkpoints/dentalpsam.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --data-dir data/test/manual_2D \
  --mesh-list splits/test.txt \
  --output-dir outputs/test_predictions \
  --device cuda:0
```

Evaluate the saved predictions on the same mesh list:

```bash
python evaluate_dentalpsam.py \
  --data-dir data/test \
  --prediction-dir outputs/test_predictions \
  --mesh-list splits/test.txt \
  --output-dir outputs/test_metrics \
  --fusion-weight-2d 0.5 --threshold 0.5 \
  --bootstrap-reps 10000 --seed 42
```

Evaluation uses the original **equal-triangle** protocol, not area weighting:
2D scores are reprojected onto faces and combined with 3D scores at fixed
0.5/0.5 weights, with plaque defined by `score > 0.5`. Metrics are calculated
per mesh and averaged; 95% confidence intervals resample participants with
all their meshes. Missing predictions are reported as errors.

Results are saved in `summary.json`, `per_mesh_metrics.csv`, and
`evaluation_manifest.json`. Always use a new output directory. For the
120-mesh MICCAI test list, add `--expected-count 120` to both commands.

The code has passed bounded server-side compatibility and integration checks,
but the MICCAI table values have **not yet been reproduced**. Measured results
and remaining gaps are listed in [Server verification](docs/SERVER_VALIDATION.md);
paper reference values are listed separately in [Paper results](docs/PAPER_RESULTS.md).

### Training

Prepare the training and validation inputs once, including the 3D branch
features. The command below trains DentalPSAM directly; it does not launch
additional upstream training. Use the same fixed feature checkpoint throughout
an experiment.

```bash
python train_dentalpsam.py \
  --train-dir data/train/manual_2D \
  --val-dir data/val/manual_2D \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --save-dir outputs/dentalpsam_train \
  --epochs 50 --batch-size 4 --learning-rate 1e-4 \
  --seed 42 --deterministic --device cuda:0
```

The default uses Adam, 2D Dice-CE loss, padding-masked 3D BCE loss, and
checkpoint selection by validation mesh BCE. Training and validation
participants must be disjoint; test labels are never used for selection.

To validate a saved checkpoint:

```bash
python validate_dentalpsam.py \
  --checkpoint checkpoints/dentalpsam.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --val-dir data/val/manual_2D \
  --output outputs/validation.json --device cuda:0
```

The model implementation is in `dentalpsam/`; the fixed 3D feature generator
is in `tsgcnet/`. Testing infers the gated or concatenation variant from
checkpoint keys and loads it strictly. New training defaults to the gated
variant; `--mesh-fusion concat` selects the alternative. Training settings,
compatibility limitations, and supplementary evaluation protocols are in
[Reproducibility](docs/REPRODUCIBILITY.md).

For development, see [Contributing](CONTRIBUTING.md). Project licensing and
third-party redistribution terms remain to be confirmed; see
[Third-party notices](THIRD_PARTY_NOTICES.md). This source snapshot is not a
formal dataset or model-weight release.
