# Known limitations

- The traced complete-cohort results do not reproduce the MICCAI table targets.
  See the fixed-artifact measurements in `SERVER_VALIDATION.md`.
- The dataset and task weights are not yet publicly released. The synthetic
  example verifies software interfaces only.
- Training entry points preserve the current validation-only protocols, which
  differ from the inspected historical training/selection procedure. Renaming
  the public workflow does not make these exact paper-training reproductions.
- The upstream checkpoint generating historical stored 3D scores is not fully
  bound to those scores. Do not replace or invert them based on test results.
- Gingival removal and downsampling precede this pipeline and are not included.
  The code does not perform uniform remeshing.
- UV separation assumes an appropriately oriented, preprocessed dental mesh.
  The synthetic demo therefore supplies artificial prepared views rather than
  claiming to exercise anatomical separation. Native normalization ignores the
  fixed-range arguments retained in its signature.
- The UV renderer's label-map-back diagnostic contains a historical shortcut
  for <=10 positive predictions. This is not used in reportable model evaluation.
- The 3D implementation retains its native reshape, padding-sensitive
  normalization, neighborhood ordering, and discarded dropout return values.
  Changing these is a scientific change, not cleanup.
- Project license and final citation metadata require author confirmation.
  Third-party licenses and notices are retained independently.
