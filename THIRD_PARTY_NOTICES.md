# Third-party notices

This source snapshot includes a modified copy of Meta's Segment Anything
implementation under `dentalpsam/segment_anything/`. Those files retain their
upstream copyright headers. The Apache License 2.0 text is included alongside
the vendored code at [its original license](dentalpsam/segment_anything/LICENSE).

`dentalpsam/model.py` contains model components derived from Segment Anything
and an MLP structure described in the historical source as adapted from
MaskFormer. `dentalpsam/branch3d/` is derived from the TSGCNet implementation associated
with:

- L. Zhang et al., “TSGCNet: Discriminative Geometric Feature Learning With
  Two-Stream Graph Convolutional Network for 3D Dental Model Segmentation,”
  CVPR 2021.
- Y. Zhao et al., “Two-Stream Graph Convolutional Network for Intra-Oral
  Scanner Image Segmentation,” IEEE TMI, 2022.

The copied TSGCNet server tree did not contain a license file. Redistribution
terms for that component and the final DentalPSAM project license must be
confirmed by the repository owner before creating a formal release tag. No
license is implied for patient data or model checkpoints; neither is included
in this repository.
