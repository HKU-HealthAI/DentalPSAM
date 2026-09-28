"""Load 3D weights with the normalization used during their training."""

from __future__ import annotations

from pathlib import Path

import torch

from dentalpsam.checkpoints import extract_model_state
from dentalpsam.branch3d.model import TSGCNet


def checkpoint_normalization(checkpoint):
    """Read the input contract; unmarked historical weights use saved moments.

    Normalization cannot be inferred from tensor shapes: both variants retain
    the same parameter and buffer keys. Unsupported coordinate preprocessing is
    rejected instead of silently applying a different loader to the weights.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a 3D branch checkpoint dictionary")
    configuration = checkpoint.get("configuration", {})
    arguments = configuration.get("arguments", {})
    coordinates = arguments.get("coordinate_normalization", "legacy")
    if coordinates != "legacy":
        raise ValueError("Unsupported 3D coordinate normalization in checkpoint")
    recorded = configuration.get("normalization")
    # Compatibility with the selected, independently validated training run.
    trial_version = arguments.get("experiment_policy_version")
    trial_mode = arguments.get("bn_statistics")
    if trial_version is not None or trial_mode is not None:
        if trial_version != "coordinate-bn-factorial-v2" or trial_mode not in ("running", "per_mesh"):
            raise ValueError("Unknown 3D checkpoint normalization metadata")
        if recorded is not None and recorded != trial_mode:
            raise ValueError("Conflicting 3D checkpoint normalization metadata")
        recorded = trial_mode
    mode = "running" if recorded is None else recorded
    if mode not in ("running", "per_mesh"):
        raise ValueError("Unknown 3D checkpoint normalization metadata")
    return mode


def load_branch3d(checkpoint: str | Path, device, k=12):
    """Strictly restore the frozen classifier for validation and feature export."""
    value = torch.load(checkpoint, map_location="cpu")
    model = TSGCNet(in_channels=9, output_channels=2, k=k,
                   normalization=checkpoint_normalization(value)).to(device)
    model.load_state_dict(extract_model_state(value), strict=True)
    return model.eval()
