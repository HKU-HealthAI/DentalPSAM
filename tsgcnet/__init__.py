"""TSGCNet implementation used to create DentalPSAM mesh features."""

# Historical implementation name retained for checkpoint/source compatibility.
# Public commands refer to its feature-generation stage as the 3D branch.

from .model import TSGCNet

__all__ = ["TSGCNet"]
