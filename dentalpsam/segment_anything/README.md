# Vendored SAM implementation

This directory derives from Meta's Segment Anything project
(https://github.com/facebookresearch/segment-anything), distributed under
Apache-2.0. The license text is retained in `LICENSES/Apache-2.0.txt` at the
repository root; see `THIRD_PARTY_NOTICES.md` for attribution.

This copy is modified: the ViT-B builder accepts the image-size and class-count
arguments used by DentalPSAM. It is retained in place because the trained
weights and checked model construction depend on this interface. It must not
be replaced by an unmodified SAM installation as a packaging cleanup.

The inspected source hashes and checkpoint compatibility evidence are retained
in `docs/SOURCE_PROVENANCE.md` and `docs/verification/`. No SAM weights are
included in this directory.
