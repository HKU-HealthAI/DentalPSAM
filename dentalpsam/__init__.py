"""DentalPSAM model, data, checkpoint, and preprocessing components.

The package root intentionally avoids importing PyTorch so dependency-light
I/O and evaluation helpers remain usable in a CPU-only audit environment.
Import :class:`dentalpsam.model.DentalPSAM` explicitly for model execution.
"""
