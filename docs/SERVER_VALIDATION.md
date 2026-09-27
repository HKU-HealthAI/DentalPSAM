# Server verification, 2026-09-27

## Current follow-up: input alignment and complete-cohort checks

Code revision: `40c8604`. Eight synthetic regression suites and fourteen CLI
import/help checks passed in Python 3.9.18, PyTorch 2.0.1 / CUDA 11.7 on the
verification server. Both DentalPSAM variants and the 3D feature model still
match the native implementations on the compatibility sample: input tensors
are identical and maximum absolute output differences are zero.

- [Gated compatibility](verification/compatibility_20260927_gated.json)
- [Concatenation compatibility](verification/compatibility_20260927_concat.json)
- [Aggregate checks and complete-cohort results](verification/verification_20260927.json)

The aggregate record includes source hashes, artifact hashes, and all metric
intervals without patient identifiers or private storage paths. A fresh
single-mesh run passed UV rendering, 3D feature export, preflight, all 22
DentalPSAM prediction patches, and original evaluation. Original PLY hashes
were unchanged. This is an integration check, not a paper-result claim.

Unlike the incomplete cache described in the earlier snapshot below, another
fixed prediction cache covers all 120 requested meshes (60 participants).
Its original equal-triangle, mesh-macro evaluation gives:

| Fixed input | Plaque IoU [95% CI] | Plaque Dice [95% CI] | Accuracy |
| --- | --- | --- | --- |
| DentalPSAM saved 2D predictions | .4055 [.3816, .4286] | .5687 [.5427, .5929] | .8078 |
| DentalPSAM saved 3D predictions | .4838 [.4601, .5063] | .6456 [.6224, .6667] | .8433 |
| DentalPSAM fixed 0.5/0.5 fusion | .4645 [.4409, .4870] | .6273 [.6036, .6492] | .8386 |
| Frozen 3D feature checkpoint | .4217 [.3853, .4562] | .5694 [.5277, .6084] | .8236 |
| Stored 3D input scores | .4823 [.4595, .5041] | .6444 [.6214, .6652] | .8398 |

Intervals use 10,000 participant-clustered bootstrap samples, seed 42. The
list hash is `ee5b59de469f9fe25f94ba8ba9752351fd6252cc7a690f4d3ba24f826581bc14`.
The frozen feature checkpoint has SHA-256
`bd54c8b2301ba5b41ae9603d55433edf0592400689434a5cf9137ca37075c0eb`.

Stored input scores and the checked feature checkpoint are not interchangeable:
all 120 meshes pass the geometry/order audit, but mean per-mesh absolute score
difference is .1104. The historical checkpoint-selection procedure used the
external cohort, so these results do not establish independent external
generalization. No cases were filtered, scores inverted, or checkpoints
selected by test performance in these checks.

The MICCAI no-stitching targets (.547/.700/.830) and feature-model targets
(.476/.626/.807) remain unreproduced. A successful code refactor does not close
that gap. The earlier evidence below is retained as a dated snapshot, not the
current coverage statement.

## Earlier snapshot

Code revision: `0df54afeb1dfccb53f3026f4ef0c57f2f567c801`.
Execution host: research23; environment: `py39_rt2`, Python 3.9.18,
PyTorch 2.0.1 / CUDA 11.7, RTX 3090. Original data, native source, and
checkpoints were read-only. Outputs were written to an isolated dated directory.

## Passed checks

`bash tests/run_checks.sh /path/to/py39_rt2/bin/python` passed five synthetic
regression suites and ten CLI import/help checks. The regression suites cover
padding-safe targets, prediction ordering, split checks, area evaluation, and
the original equal-triangle protocol (including strict thresholding, UV
truncation, empty classes, clustered bootstrap, and duplicate rejection).

`tools/verify_server_compatibility.py` compared one real mesh with trusted native
source. DentalPSAM comparisons use the first 256-pixel image/6000-row mesh
patch; TSGCNet comparisons use the full 16,000-face input. These are bounded
compatibility checks, not full-cohort performance or training acceptance.

| Comparison | Observed result |
| --- | --- |
| Historical gated DentalPSAM versus pinned historical source | Input tensors identical; 2D and 3D maximum absolute output differences both 0 |
| Newer concatenation DentalPSAM versus current server source | Input tensors identical; 2D and 3D maximum absolute output differences both 0 |
| Packaged TSGCNet versus current server TSGCNet | Input tensors identical; maximum absolute output difference 0 |
| Original equal-triangle fixed fusion versus native evaluator, one complete mesh | Plaque IoU and Dice exactly equal |

The checkpoints were chosen for architecture compatibility, not test-set
performance. The saved predictions used for evaluator comparison are a
separate fixed input; this test does **not** establish which checkpoint created
them. No training or checkpoint selection was performed.

Aggregate evidence, without patient data or identifiers:

- [Historical gated evidence](verification/compatibility_gated.json), SHA-256
  `2d43063fdafe38891e2c461e3c0f6f6ef0dd9ff10ab4a75ed5b4066e303c8386`.
- [Concatenation evidence](verification/compatibility_concat.json), SHA-256
  `a40e0c7e7e544692a62a602500bba720b6a55ed78b49ce1caed6ccba7e9a599c`.

Both reports contain checkpoint hashes and the hashes of 33 source modules;
these were checked against the committed source. Exact commands and complete
logs remain in the server's isolated verification directory.

## Findings fixed or explicitly bounded

- GPU inference initially failed because SAM's preprocessing buffers remained
  on CPU. Moving SAM before constructing DentalPSAM fixed the issue.
- Loading a historical 397-key checkpoint into the newer 403-key native model
  failed strictly. The package now distinguishes gated and concatenation
  variants rather than discarding missing keys.
- The compatibility driver keeps face indices on CPU for native NumPy
  adjacency construction; features are sent to the selected GPU.

## MICCAI full-cohort reproduction is still open

The inspected 120-mesh list contains 60 participants and has SHA-256
`ee5b59de469f9fe25f94ba8ba9752351fd6252cc7a690f4d3ba24f826581bc14`.
In the inspected `3D2D_Pred_Output/ckpt_ep60` cache, only 108 of those 120
requested meshes have all three PNG and 3D NPZ pairs; 12 are missing. The
complete-list evaluation correctly failed on missing input instead of dropping
cases. Two inspected ablation caches each covered only 76 of the 120 meshes.

Therefore no 120-mesh performance or CI is claimed here. Recovering the exact
paper checkpoint, input features, and all requested predictions remains
necessary. Do not label these compatibility tests as reproduction of the
MICCAI no-stitching .547/.700/.830 row, or the full stitching result.

The UV-generation/export commands have import/help coverage but have not yet
passed a complete fresh-data end-to-end generation run in this verification.
The training entry points have not been retrained or accepted against the
paper training result. These boundaries remain explicit rather than being
inferred from successful inference.
