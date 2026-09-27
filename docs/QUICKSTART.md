# Quick start

## Install

The package accepts Python 3.9–3.11 and the dependency ranges in
`pyproject.toml`. These are installation compatibility bounds, not a claim
that every combination was tested. The exact checked environment is Python
3.9.18, PyTorch 2.0.1, torchvision 0.15.2, and CUDA 11.7.

For that reference stack, run from the cloned repository:

```bash
conda env create --file environment.yml
conda activate dentalpsam
```

Alternatively, use an existing compatible environment:

```bash
python -m pip install -e .
```

GPU execution needs an appropriate NVIDIA driver. CPU-only testing of software
contracts uses no CUDA device. Full fresh Conda resolution and exact historical
training reproduction are separate from installing the editable package.

## Public demo

```bash
python examples/synthetic_sample/run.py --output outputs/synthetic_demo
```

This generates artificial geometry and fabricated probabilities, exercises the
real loader, and writes an evaluation report. It does not measure model quality.

## Test with study data and weights

You need authorized access to the [dataset](DATASET.md) and compatible
[weights](MODEL_AND_CHECKPOINTS.md). There are no automatic private downloads.

```bash
python test.py --data data/test --checkpoint checkpoints/dentalpsam.pth \
  --mesh-list data/test/mesh_ids.txt --output outputs/test --expected-count 120
```

`--data` is a single split, not the full train/val/test root. It must contain
`origin/`, `label/`, and prepared `manual_2D/` inputs. The fixed mesh list is
required; the program does not infer or filter a test cohort from model results.

For a preprocessed PLY split without prepared inputs:

```bash
python test.py --data data/test --checkpoint checkpoints/dentalpsam.pth \
  --3d-checkpoint checkpoints/3d_branch.pth --output outputs/test_from_ply
```

This explicitly requests new UV views and 3D features. Supplying
`--3d-checkpoint` regenerates inputs in the run directory even when prepared
inputs already exist. Do not substitute a different upstream checkpoint for
the one associated with the trained DentalPSAM weights.

The run directory must not exist and must be outside the source data tree.
Use `summary.txt` for the fused summary, `metrics.json` for all branch metrics,
and `evaluation/` for the exact inputs and per-mesh report. Any missing mesh or
misaligned face array is an error. Detailed progress is in `run.log`.
