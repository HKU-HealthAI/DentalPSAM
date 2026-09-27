# Server verification, 2026-09-27

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
