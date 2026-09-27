# Acceptance criteria

This document separates repository usability from scientific reproduction.
The current cleanup is a **Repository Release Gate**: clear Dataset, Testing,
and two-stage Training instructions; working public commands; preserved
scientific behavior; and no private artifacts in Git. Matching published
metrics is a separate **Scientific Reproduction Gate**, not a cleanup condition.

The structural candidate is not yet fully accepted: its final server regression
rerun and clean-checkout checks remain pending. A Git push does not mark either
gate passed. The scientific targets below are retained for later verification.

## 1. Clear code structure

- Keep model, data loading, checkpoint loading, generation, training,
  validation, prediction, and evaluation responsibilities separate.
- Expose documented command-line entry points without private hard-coded paths.
- Use descriptive snake_case functions/files, four-space indentation, and
  explicit array contracts. Preserve checkpoint parameter names when renaming
  would invalidate a historical checkpoint.
- Keep patient data, checkpoints, predictions, and credentials outside Git.
- Test the actual data flow on the server, including newly generated UV and
  mesh NPZ files, rather than relying only on `--help`.

## Separate scientific gate: results matching the MICCAI paper

The current requested scope excludes stitching. The target is the paper's
`w/o Stitch` row: plaque IoU **0.547**, Dice **0.700**, OA **0.830**.
TSGCNet is an independent acceptance target too: plaque IoU **0.476**,
Dice **0.626**, OA **0.807**, plus the non-plaque and mean-class entries in
the MICCAI comparison table. Producing an NPZ or passing inference is not
TSGCNet result reproduction. Bind the upstream checkpoint, score convention,
normalization, generated SOTA meshes, and downstream DentalPSAM checkpoint
as one reproducible chain; do not silently replace the upstream model.
Match all three at the paper's published precision on the same documented
cohort and original equal-triangle protocol. A change of evaluation unit,
split, threshold, or selected cases cannot count as reproducing that row.

Required evidence: fixed split and participant identities, exact input
features, checkpoint SHA-256, model variant, command, Git revision, complete
prediction set, evaluator configuration, per-mesh metrics, and aggregate
results. The reconstruction must run on research23 in `py39_rt2`. Keep raw
data read-only and write derived outputs into a new directory.

Do not tune to the test set or fill gaps by mixing incompatible caches. Do
not attach CIs from another run to the paper values. If the target cannot be
recovered, document the mismatch and missing provenance; mark this criterion
**not passed**, rather than changing the acceptance target.

## 3. Detailed, usable README

A new reader must be able to identify the intended model variant and run the
workflow from the README: dataset access, environment, weights, testing,
Stage 1 (3D branch), and Stage 2 (DentalPSAM). Keep optional intermediate
commands and file schemas in the detailed documentation. Explain prerequisites,
where files are written, known limitations, and expected outputs. Distinguish
paper targets, measured results, diagnostics, and optional area analysis.

## 4. Useful comments and documentation

Document shapes, ordering, label direction, units, padding, normalization,
fusion variants, and compatibility decisions. Explain why a non-obvious
operation is retained. Remove stale debug code and misleading shape comments.
Add regression tests for non-obvious indexing, padding, thresholding, and
projection behavior. Documentation cannot substitute for a missing test.

## Current acceptance boundary

The [server verification record](SERVER_VALIDATION.md) establishes bounded
inference compatibility, synthetic test coverage, and a fresh single-mesh
generation/prediction/evaluation run. Complete-cohort measurements are also
available, but do not match the paper targets. Overall paper-reproduction
acceptance is therefore **not passed**. That numerical mismatch does not block
the independent repository-readability gate.
