# Reproducibility notes

## Installation

Clone the repository and run from its root. The package accepts Python
3.9–3.11; dependency ranges are defined in `pyproject.toml`. These are
installation bounds, not a claim that every combination was tested.

For the reference Conda stack:

```bash
conda env create --file environment.yml
conda activate dentalpsam
```

For an existing compatible environment:

```bash
python -m pip install -e .
```

GPU execution needs an appropriate NVIDIA driver. The synthetic example and
unit tests need no patient data, external checkpoints, or GPU. Fresh Conda
resolution is separate from installing into the checked server environment.

## Model configuration

DentalPSAM uses the vendored SAM ViT-B image encoder with 1024 x 1024 model
input, a 64 x 64 embedding grid, embedding dimension 256, and eight attention
heads. The mesh path receives padded `[batch, 6000, 10]` patches. Channel 9 of
`SOTA_mesh` is an auxiliary 3D feature score, contractually `P(plaque)`.
The historical decoder multiplies it into learned geometry; the newer decoder
concatenates it with geometry before a fusion MLP. Inference selects the
architecture from checkpoint keys and enforces strict loading. Do not invert
a historical score channel merely because a test-set metric improves: the
original generating checkpoint and score convention must establish its meaning.

The default clean training protocol is 50 epochs, batch size 4, Adam with
learning rate `1e-4` and zero weight decay, a StepLR decay of 0.1 after 40
epochs, 2D Dice-CE weight 2, valid-face mesh BCE weight 1, seed 42, and no
early stopping. The best checkpoint is selected by validation mesh BCE only.
The historical patch filter keeps 256 x 256 label patches containing at least
50 positive pixels; set `--min-positive-pixels 0` only for a separately named
all-patch experiment.

The 3D branch uses 16,000 faces, `k=12`, batch size 1, Adam at `1e-4`, weight decay
`1e-5`, at most 50 epochs, and validation patient-macro area-weighted plaque
IoU for checkpoint selection. Its training command cannot receive a test or
external directory.

## Computing environment

The public environment specification is `environment.yml`. PyTorch,
torchvision, and the CUDA runtime are installed through Conda; direct Python
dependencies are pinned in `requirements-validated.txt`. The package dependency
ranges in `pyproject.toml` are separate installation bounds, not a full tested
version matrix. This is a reference environment,
not a fully locked set of transitive dependencies. Fresh-environment creation
has not yet been verified; the execution checks used the existing environment
below. Installing a newer dependency stack is a separate compatibility test.

Server checks on 2026-09-27 used the existing `py39_rt2` environment:
Python 3.9.18, PyTorch 2.0.1, torchvision 0.15.2, MONAI 1.3.2,
NumPy 1.26.4, OpenCV 4.10.0.84, Open3D 0.18.0, plyfile 1.1.3,
pandas 2.3.3, patchify 0.2.3, scikit-learn 1.6.1, and tqdm 4.66.4.
The inspected host has eight NVIDIA GeForce RTX 3090 GPUs (24,576 MiB each)
and driver 550.54.15. These are observed validation-environment versions,
not proof of the environment used for every historical training run.
For a new result, archive `python --version`, `pip freeze`, `torch.__version__`,
`torch.version.cuda`, GPU model/driver, and the complete command beside the run
manifest.

## Determinism

`train.py --stage dentalpsam --deterministic` seeds Python, NumPy, PyTorch, CUDA, and
DataLoader workers and requests deterministic PyTorch algorithms with warnings
for unsupported kernels. `train.py --stage 3d` enables the same deterministic
policy. Exact cross-version or cross-GPU bitwise identity is not promised.

## Experimental separation

- Select checkpoints only on the declared internal validation participants.
- Do not use internal test or external labels for checkpoint, threshold, or
  fusion-weight selection.
- Export a fixed, participant-complete mesh list to a fresh output directory.
- Use one evaluator invocation for all models in a statistical comparison.
- Keep face, area, and vertex units separate. The MICCAI reproduction default
  is the original equal-triangle metric; area is a separate analysis.

## Training compatibility boundary

The main DentalPSAM training workflow consumes prepared inputs from a fixed
3D feature checkpoint. It does not retrain that generator implicitly.
`train.py --stage 3d` exposes that separate stage for training from scratch;
testing or training DentalPSAM with existing prepared inputs does not require it.

The clean training entry points are explicit, validation-only protocols;
they are not verbatim copies of the historical trainers. In particular, the
clean 3D trainer uses NLL loss and validation-area selection, whereas the
inspected historical trainer used DiceFocal and external label-1 IoU selection.
These changes must not be described as exact reproduction of historical
training. Existing frozen checkpoints should first be tested with matched
inference and the original evaluator; retraining is not required for that check.

## Supplementary analysis

`tools/evaluate_mesh.py` supports face/area/vertex comparisons. Edit
`examples/evaluation_spec.json` to locate each method's saved predictions, then
run the methods together on one declared cohort. Area metrics weight each
triangle by its physical surface area; their participant-level aggregation
differs from the original mesh-macro evaluator. Keep these results separate.

`tools/audit_sota_inputs.py` checks geometry, face order, score direction, and
agreement with independently exported scores. `tools/fingerprint_tsgcnet.py`
compares checkpoint probabilities with cached feature scores without using
test IoU for selection. These are provenance checks, not model-training steps.

For a fresh-data integration check, use `tools/smoke_pipeline.py --help`.
It executes all five stages on one mesh and records input hashes before and
after execution. Such a check cannot establish full-cohort paper performance.

## Known limitations

- Dataset and task weights are not yet public. The synthetic example checks
  software interfaces, not model quality or anatomical UV separation.
- The traced full-cohort results do not reproduce the MICCAI targets. Training
  protocols differ from the inspected historical selection procedure; naming
  the stages does not make them an exact historical replay.
- The generating checkpoint for historical stored 3D scores is not fully
  resolved. Do not substitute or invert scores based on test performance.
- Gingival removal and downsampling precede this pipeline. Uniform remeshing
  is not performed by this code.
- UV separation assumes oriented, preprocessed dental meshes. The renderer
  retains unused fixed-range arguments and a label-map-back diagnostic shortcut
  for at most ten positive predictions; that diagnostic is not the reported
  model evaluator.
- The 3D implementation preserves native reshape, padding-sensitive
  normalization, neighborhood ordering, and discarded dropout return values.
  Altering these is a scientific change, not repository cleanup.
- Project licensing and final citation metadata await author confirmation.
  Third-party licenses and notices are retained independently.

## Archived evidence and separate acceptance

The [Repository Release Gate](ACCEPTANCE.md) concerns public usability and
unchanged software behavior. The [Scientific Reproduction Gate](archive/PAPER_RESULTS.md)
is separate and remains open. Neither a Git push nor a passing import check
completes both gates.

[Source provenance](archive/SOURCE_PROVENANCE.md),
[revision-bound server checks](archive/SERVER_VALIDATION.md), and the aggregate
JSON reports in `verification/` are retained for audit. They document specific
earlier revisions and are not required reading for normal testing or training.
