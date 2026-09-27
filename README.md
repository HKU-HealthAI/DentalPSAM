# DentalPSAM

**DentalPSAM: Lifting SAM to Dental Plaque Segmentation with Hybrid 2D-3D
Knowledge Fusion** — MICCAI 2026.

DentalPSAM combines a geometry-enhanced 2D SAM branch and an appearance-enhanced
3D mesh branch to segment dental plaque in intraoral scans.

![DentalPSAM architecture from Figure 1 of the MICCAI paper](assets/method.png)

Architecture shown in the MICCAI paper. The implementation guide is
[inside the package](dentalpsam/README.md).

## Dataset

The dataset is being organized and is not publicly released yet.
Please reach out to the authors regarding data access.
This repository does not distribute patient scans or annotations.

See [Dataset](docs/DATASET.md) for the input layout and annotations.

## Installation

Use Python 3.9–3.11 and run from the repository root:

```bash
git clone https://github.com/HKU-HealthAI/DentalPSAM.git
cd DentalPSAM
python -m pip install -e .
```

Model execution requires a compatible PyTorch environment; the reference
server uses Python 3.9, PyTorch 2.0, and an NVIDIA GPU.

## Test

Place the authorized test data in `data/test` and compatible weights in:

```text
checkpoints/
  dentalpsam.pth
  branch3d.pth
  sam.pth
```

Task-trained checkpoints are not publicly released yet. Once available to you:

```bash
python test.py --data data/test --weights checkpoints --output results
```

The command prepares inputs as needed, predicts, and evaluates. Read
`results/summary.txt` or `results/metrics.json` for the results.
Testing does not require retraining.

## Train

### 1. Train the 3D branch

```bash
python train.py --stage 3d --data data --output outputs/3d
```

### 2. Train DentalPSAM

```bash
python train.py --stage dentalpsam --data data \
  --branch-checkpoint outputs/3d/best.pth --output outputs/dentalpsam
```

Place SAM initialization at `checkpoints/sam.pth`. The second stage prepares
3D inputs from the first stage's checkpoint automatically.
See [Training](docs/TRAINING.md) for prerequisites and options.

## MICCAI results

Table 1 of the MICCAI paper, PlaqueIOS test set (120 scans). DSC denotes Dice;
OA denotes overall accuracy. Higher is better for every metric.

| Method | Plaque IoU | Non-plaque IoU | Mean IoU | Plaque DSC | Non-plaque DSC | Mean DSC | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet | 0.436 | 0.709 | 0.572 | 0.601 | 0.824 | 0.712 | 0.769 |
| TSGCNet | 0.476 | 0.758 | 0.617 | 0.626 | 0.858 | 0.742 | 0.807 |
| DentalMAE | 0.447 | 0.751 | 0.599 | 0.611 | 0.855 | 0.733 | 0.799 |
| Fine-tuned SAM | 0.463 | 0.745 | 0.604 | 0.623 | 0.851 | 0.737 | 0.800 |
| H-SAM | 0.472 | 0.700 | 0.586 | 0.635 | 0.819 | 0.727 | 0.772 |
| CrossTooth | 0.480 | 0.746 | 0.613 | 0.645 | 0.851 | 0.748 | 0.801 |
| Fine-tuned SAM3 | 0.513 | 0.693 | 0.603 | 0.673 | 0.814 | 0.744 | 0.777 |
| **DentalPSAM** | **0.556** | **0.777** | **0.667** | **0.712** | **0.872** | **0.792** | **0.831** |

![Qualitative comparison from Figure 3 of the MICCAI paper](assets/comparison.png)

Qualitative comparison from Figure 3 of the MICCAI paper. Blue marks plaque.

## Code

The [package guide](dentalpsam/README.md) maps model, data, training, and evaluation
modules, including the 3D branch in `dentalpsam/branch3d/`.
`train.py` and `test.py` are the public commands; both support `--help`.
Developer checks and recorded environments are separate from this workflow.

## Citation

Haoning Jiang et al. *DentalPSAM: Lifting SAM to Dental Plaque Segmentation with
Hybrid 2D-3D Knowledge Fusion.* MICCAI, 2026.
Third-party attribution is retained in [Third-party notices](THIRD_PARTY_NOTICES.md).
