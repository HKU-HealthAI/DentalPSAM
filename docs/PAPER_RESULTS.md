# MICCAI paper targets and reproduction boundary

Source: the supplied `MICCAI2026_DentalPSAM (1).zip`,
`LatexSource-4344/sec/experiment.tex`. Its comparison and ablation tables agree
with the earlier `MICCAI2026_DentalPSAM.zip`. The CVPR archive and the separate
clinical `Tables.pdf` are **not** the numerical authority for this release.

The MICCAI dataset comprises 400 intraoral scans from 200 participants; the
reported train/validation/test split is 220/60/120 meshes. A historical 88-mesh
cohort or clinical 60-case table cannot replace this 120-mesh test cohort.

## Comparison table (paper reference only)

| Method | Plaque IoU | Non-plaque IoU | Mean IoU | Plaque Dice | Non-plaque Dice | Mean Dice | OA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PointNet | .436 | .709 | .572 | .601 | .824 | .712 | .769 |
| TSGCNet | .476 | .758 | .617 | .626 | .858 | .742 | .807 |
| DentalMAE | .447 | .751 | .599 | .611 | .855 | .733 | .799 |
| Fine-tuned SAM | .463 | .745 | .604 | .623 | .851 | .737 | .800 |
| H-SAM | .472 | .700 | .586 | .635 | .819 | .727 | .772 |
| CrossTooth | .480 | .746 | .613 | .645 | .851 | .748 | .801 |
| SAM3 | .513 | .693 | .603 | .673 | .814 | .744 | .777 |
| DentalPSAM (with stitching) | .556 | .777 | .667 | .712 | .872 | .792 | .831 |

## Current scope: main model without stitching

The `w/o Stitch` ablation row reports plaque IoU **.547**, plaque Dice **.700**,
and OA **.830**. This is the reference target for the present package. Do not
claim that a no-stitching run reproduces the full model's .556/.712 row.

The default evaluator is `scripts/dentalpsam/evaluate.py`: equal triangle weights,
strict `> 0.5`, UV truncation, fixed 0.5/0.5 fusion, and mesh-macro averaging.
Compatibility with the inspected server evaluator and reproduction of a
paper target are two separate acceptance conditions.

## Evidence required for a reproduced result

Keep the Git commit/source hashes, fixed split hash, checkpoint hashes,
prediction provenance, complete evaluation manifest, and measured results
together. A checkpoint filename such as `ckpt_ep60.pth` is not an identity.
Passing unit tests or matching the original evaluator does not prove that a
particular checkpoint and saved predictions generated the paper table.

Do not attach newly computed CIs to paper reference values. Do not optimize
checkpoint, split membership, threshold, or fusion weights on test outcomes.
Any unmatched number must remain explicitly unresolved.
