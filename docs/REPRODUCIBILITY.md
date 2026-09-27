# Reproducibility notes

## Model configuration

DentalPSAM uses the vendored SAM ViT-B image encoder with 1024 x 1024 model
input, a 64 x 64 embedding grid, embedding dimension 256, and eight attention
heads. The mesh path receives padded `[batch, 6000, 10]` patches. Channel 9 of
`SOTA_mesh` is an auxiliary TSGCNet score, contractually `P(plaque)`.
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

TSGCNet uses 16,000 faces, `k=12`, batch size 1, Adam at `1e-4`, weight decay
`1e-5`, at most 50 epochs, and validation patient-macro area-weighted plaque
IoU for checkpoint selection. Its training command cannot receive a test or
external directory.

## Computing environment

Server checks on 2026-09-27 use the user-confirmed `py39_rt2` environment:
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

`train_dentalpsam.py --deterministic` seeds Python, NumPy, PyTorch, CUDA, and
DataLoader workers and requests deterministic PyTorch algorithms with warnings
for unsupported kernels. `train_tsgcnet.py` enables the same deterministic
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

The clean training entry points are explicit, validation-only protocols;
they are not verbatim copies of the historical trainers. In particular, the
clean TSGCNet trainer uses NLL loss and validation-area selection, whereas the
inspected historical trainer used DiceFocal and external label-1 IoU selection.
These changes must not be described as exact reproduction of historical
training. Existing frozen checkpoints should first be tested with matched
inference and the original evaluator; retraining is not required for that check.
