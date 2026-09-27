# Model and checkpoint dependencies

## Required weights

- `sam_vit_b_01ec64.pth`: SAM ViT-B initialization, separate from task weights.
- `dentalpsam.pth`: trained DentalPSAM state dictionary.
- `3d_branch.pth`: frozen mesh feature weights, needed only when generating
  inputs from PLY or starting the two-stage training workflow.

Weights are not redistributed here. Task checkpoint release URLs and final
licensing are pending; no download endpoint is implied by these example names.
Load only trusted checkpoint/NPZ files: historical formats use pickle.

## Model variants

The gated decoder multiplies the auxiliary score into learned geometry
features. The concatenation decoder adds a learned fusion MLP. The loader
detects keys beginning with `mesh_decoder.mlp_fusion.` to select concatenation;
otherwise it constructs the gated variant. Both load with `strict=True`.
Missing or unexpected keys are errors; no head is silently reinitialized.

SAM is moved to the selected device before DentalPSAM is constructed because
its bound preprocessing method retains SAM's normalization buffers.
The modified vendored builder is retained in place. Do not replace it with an
unmodified SAM package during installation or restructuring.

## Public names and implementation boundaries

The public workflow uses 2D branch, 3D branch, fusion, and DentalPSAM. The
upstream geometry/color feature generator retains the internal `TSGCNet`
class and package names and its original attribution. It supplies the tenth
column of the prepared mesh input. This is not DentalPSAM's own mesh encoder
or decoder, and renaming the workflow does not imply a new architecture.

The input convention is `P(plaque)` in channel 9. Never invert a score because
a test metric increases. Bind the generating checkpoint, input scores, and
DentalPSAM checkpoint to the same run. A better independently trained feature
model is not automatically interchangeable with an existing input distribution.

Native input/forward compatibility evidence is retained under
`docs/verification/`. It does not establish full-cohort paper reproduction.
