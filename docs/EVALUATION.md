# Evaluation protocols

## Original MICCAI protocol: default

`test.py` runs prediction and the original protocol together. For predictions
already saved, use `scripts/dentalpsam/evaluate.py --help`.

Each triangle is equally weighted. The 2D branch uses saved 8-bit PNG scores
with the native OpenCV rounding convention. UV coordinates at triangle centers
are truncated to integer pixels, clipped to the image, and sampled without
resizing. Fusion is fixed at 0.5 times the 2D score plus 0.5 times the 3D score.
Plaque decisions use strict `score > 0.5`, not `>= 0.5`.

Metrics are computed for each mesh and averaged over meshes. A 95% percentile
bootstrap resamples participants together with all their meshes, preserving
that mesh-macro estimator. The default is 10,000 repetitions and seed 42.
The report includes plaque/non-plaque IoU and Dice, mean-class metrics, accuracy,
sensitivity, specificity, predictive values, and F1. Empty-union IoU/Dice are
1; confusion-matrix F1 is 0 when no positives exist, matching the original.

Use the same fixed, complete list for every model. Duplicate faces, malformed
scores, missing inputs, and incomplete partitions are errors, not exclusions.
No checkpoint, threshold, or fusion weight is selected from the test outcome.

## Physical-area analysis

`tools/evaluate_mesh.py` is a separate analysis with per-triangle weights
`0.5 * ||(v2 - v1) x (v3 - v1)||`. It combines a participant's arches before
calculating metrics, then averages participants. It is not the default paper
estimator. Its separate `face` row also uses this participant-level aggregation
and is not interchangeable with the original mesh-macro metric.

Use `examples/evaluation_spec.json` to locate each method's predictions and
`--help` for the full CLI. All methods in a paired comparison must use the
same cohort and protocol. Paired bootstrap intervals, sign-flip tests, and Holm
adjustment are available; do not attach an interval from one run to a point
estimate from another.

## Vertex sensitivity analysis

The supplementary evaluator also supports vertex-level aggregation. This is a
different measurement unit; it does not establish physical area coverage when
mesh density varies. Segmentation IoU/Dice and plaque surface coverage are
distinct quantities. Name the weighting and aggregation in every reported row.
