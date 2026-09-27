"""Checkpoint construction and loading for the baseline DentalPSAM model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import DentalPSAM
from .segment_anything import sam_model_registry


def build_sam_vit_b(checkpoint: str | Path) -> torch.nn.Module:
    """Build the modified SAM ViT-B backbone used by DentalPSAM.

    The vendored SAM builder accepts ``num_classes=1``; this differs from the
    unmodified upstream registry and is required by the historical model.
    """
    return sam_model_registry["vit_b"](
        image_size=1024,
        num_classes=1,
        checkpoint=str(checkpoint),
    )


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return a model state dictionary from supported checkpoint wrappers."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")
    return checkpoint


def load_dentalpsam(
    sam_checkpoint: str | Path,
    dentalpsam_checkpoint: str | Path,
    device: torch.device,
) -> DentalPSAM:
    """Strictly load a baseline DentalPSAM checkpoint for inference."""
    # DentalPSAM retains SAM's bound preprocess method. Its pixel_mean/std
    # buffers belong to SAM, so move SAM before constructing DentalPSAM.
    checkpoint = torch.load(dentalpsam_checkpoint, map_location="cpu")
    state = extract_model_state(checkpoint)
    fusion = (
        "concat"
        if any(key.startswith("mesh_decoder.mlp_fusion.") for key in state)
        else "gated"
    )
    sam = build_sam_vit_b(sam_checkpoint).to(device)
    model = DentalPSAM(sam, mesh_fusion=fusion).to(device)
    # Never ignore missing/unexpected keys or initialize a missing fusion head.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
