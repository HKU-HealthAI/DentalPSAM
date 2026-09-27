# DentalPSAM

DentalPSAM combines a 2D branch and a 3D branch for dental-plaque segmentation.
This repository provides the code for preparing inputs, testing trained
checkpoints, and training the model.

## Dataset

The dataset used in this work is currently being organized and is not yet
available for public release. Please reach out to the authors regarding data
access. No patient scans, annotations, or participant identifiers are distributed
in this repository.

See [Dataset specification](docs/DATASET.md) for the actual file layout,
annotations, preprocessing assumptions, and participant-level split rules.
The code reads original data without overwriting it.

## Code

Install from the repository root in a Python 3.9–3.11 environment:

```bash
git clone https://github.com/HKU-HealthAI/DentalPSAM.git
cd DentalPSAM
python -m pip install -e .
```

Server compatibility checks use Python 3.9.18, PyTorch 2.0.1,
CUDA 11.7, and an NVIDIA GPU. Dependency ranges are in `pyproject.toml`; the
exact reference stack is in `environment.yml` and `requirements-validated.txt`.
See [Environment setup](docs/REPRODUCIBILITY.md#installation) for details.

### Testing

**Without study data.** Try the CPU-only interface demo:

```bash
python examples/synthetic_sample/run.py --output outputs/synthetic_demo
```

This synthetic example verifies file layout, data loading, model-input
construction, and reporting. It does not run the neural model or reproduce
the paper's experimental results.

**With study data and trained checkpoints.** Prepare the
[test split](docs/DATASET.md) and [weights](docs/MODEL_AND_CHECKPOINTS.md).
Public task-checkpoint download links are not
yet available; the repository does not contain pretrained weights.

```bash
python test.py \
  --data data/test \
  --checkpoint checkpoints/dentalpsam.pth \
  --output outputs/test
```

By default, the command reads `data/test/mesh_ids.txt` and loads
`sam_vit_b_01ec64.pth` beside the DentalPSAM checkpoint. Override these with
`--mesh-list` and `--sam-checkpoint`. If the split contains preprocessed
PLY pairs instead of prepared inputs, add `--3d-checkpoint` to generate the
required inputs automatically in the new output directory.

One command runs prediction, 2D-to-mesh reprojection, fixed 0.5/0.5 fusion, and
the original **equal-triangle** evaluation. It writes:

```text
outputs/test/
  predictions/              Saved 2D and 3D probabilities
  metrics.json              Both classes, branch/fusion metrics, and 95% CIs
  summary.txt               Readable fused-result summary
  evaluation/               Per-mesh results and provenance manifest
  run.log                   Detailed execution log
```

For the fixed MICCAI test list, use `--expected-count 120`. The decision rule
is strict `> 0.5`; confidence intervals resample participants with all their
meshes. Area and vertex analyses are separate protocols, documented in
[Evaluation](docs/EVALUATION.md).

### Training

Testing does not require retraining. For a new training run, use two stages.
The input root contains participant-disjoint `train/` and `val/` splits.

**1. Train the 3D branch.** It learns mesh representations from geometry and
scan colors, using participant-disjoint training and validation splits.

```bash
python train.py --stage 3d \
  --data data --output outputs/3d_branch --epochs 50
```

**2. Train DentalPSAM.** The command below uses the selected 3D checkpoint to
prepare fixed branch inputs automatically, then trains DentalPSAM with SAM
initialization:

```bash
python train.py --stage dentalpsam \
  --data data \
  --3d-checkpoint outputs/3d_branch/best.pth \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --output outputs/dentalpsam --epochs 50 --seed 42 --deterministic
```

If inputs are already prepared, omit `--3d-checkpoint`. See
[Training](docs/TRAINING.md) for validation, defaults, and protocol limitations.
`python train.py --stage dentalpsam --help` lists the available options.

The model, data loader, and shared training logic are in `dentalpsam/`.
`train.py` and `test.py` are the main entry points; `scripts/` contains optional
step-by-step commands. See [Model and checkpoints](docs/MODEL_AND_CHECKPOINTS.md)
for the internal implementation and weight dependencies. `configs/` provides
readable option examples; command-line arguments remain authoritative.

For contributors:

```bash
python -m pip install -e ".[dev]"
pytest -q
bash tests/run_checks.sh
```

See [Repository acceptance](docs/ACCEPTANCE.md) for the current software gate.
Paper-result reproduction is tracked separately and remains unverified; a
readability cleanup does not establish matching paper metrics.

Formal citation metadata and project licensing are awaiting author confirmation.
This is not yet a formal dataset or model-weight release. Existing third-party
attribution remains in [Third-party notices](THIRD_PARTY_NOTICES.md).
