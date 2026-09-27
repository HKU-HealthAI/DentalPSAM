# DentalPSAM

DentalPSAM combines 2D and 3D branches for dental-plaque segmentation.

## Dataset

The dataset is being organized and is not publicly released yet.
Please reach out to the authors regarding data access.
This repository does not contain patient data.

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

## Code

The implementation lives in `dentalpsam/`, including `dentalpsam/branch3d/`.
`train.py` and `test.py` are the public commands; both support `--help`.
Developer checks and recorded environments are separate from this workflow.

## Citation

Publication metadata will be added after author confirmation.
Third-party attribution is retained in [Third-party notices](THIRD_PARTY_NOTICES.md).
