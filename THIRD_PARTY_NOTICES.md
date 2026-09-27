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

The copied server tree and the inspected
[upstream TSGCNet repository](https://github.com/zhanglingming1/tsgcnet) did not
provide a license establishing redistribution permission. Public visibility
alone is not an open-source license; see
[GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

Redistribution authorization remains **unresolved**, including for the current
public source distribution, not only for a future release tag. The repository
owner needs applicable license evidence or explicit permission from the rights
holder before treating this component as cleared for distribution. Alternatively,
replacement would require a separately scoped independent implementation and
compatibility review. Adding a DentalPSAM license would not relicense the
third-party code. No permission or legal clearance is asserted here.

The project license is also undecided. SAM's retained Apache-2.0 notice applies
to its covered components only. Patient data and model checkpoints are not
distributed, and no license for them is implied.
