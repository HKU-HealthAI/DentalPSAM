"""Shared supervision and metric helpers for padded mesh patches.

``label_mesh`` stores nine geometry/auxiliary channels followed by the
continuous plaque target in channel 9. Mesh patches are left-padded with rows
of zeroes to the public ``[6000, 10]`` shape. Those rows are implementation
padding, not negative examples, and must never contribute to 3D loss or
diagnostic metrics.
"""

from __future__ import annotations

import torch


METRIC_THRESHOLD = 0.5
GEOMETRY_CHANNEL_COUNT = 9
TARGET_CHANNEL_INDEX = 9


def _validate_label_mesh(label_mesh: torch.Tensor) -> None:
    """Validate the public ``[batch, points, 10]`` label-mesh contract."""
    if label_mesh.ndim != 3 or label_mesh.shape[-1] <= TARGET_CHANNEL_INDEX:
        raise ValueError(
            "label_mesh must have shape [batch, points, >=10] with target channel 9; "
            f"got {tuple(label_mesh.shape)}."
        )


def valid_mesh_point_mask(label_mesh: torch.Tensor) -> torch.Tensor:
    """Return ``[batch, points, 1]`` true only for non-padding mesh rows.

    A row is valid when at least one of channels 0--8 is nonzero. Using
    ``any`` rather than a signed sum avoids treating a legitimate geometry row
    whose feature values happen to sum to zero as padding.
    """
    _validate_label_mesh(label_mesh)
    return label_mesh[..., :GEOMETRY_CHANNEL_COUNT].ne(0).any(dim=-1, keepdim=True)


def soft_mesh_targets(label_mesh: torch.Tensor) -> torch.Tensor:
    """Return the continuous channel-9 target used by BCE training."""
    _validate_label_mesh(label_mesh)
    return label_mesh[..., TARGET_CHANNEL_INDEX : TARGET_CHANNEL_INDEX + 1]


def binary_mesh_targets(label_mesh: torch.Tensor, target_threshold: float) -> torch.Tensor:
    """Threshold continuous channel-9 targets for diagnostic IoU/Dice only."""
    if not 0.0 <= target_threshold <= 1.0:
        raise ValueError("target_threshold must be in [0, 1].")
    return soft_mesh_targets(label_mesh) >= target_threshold


def masked_mesh_loss(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Average an elementwise mesh loss over valid (non-padding) rows only.

    ``criterion`` must use ``reduction='none'`` and return a tensor with the
    same shape as ``logits``. This keeps the existing soft-label BCE objective
    while removing zero-padding rows from its denominator.
    """
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError(
            "logits, targets, and valid_mask must have identical shapes; got "
            f"{tuple(logits.shape)}, {tuple(targets.shape)}, {tuple(valid_mask.shape)}."
        )
    losses = criterion(logits, targets)
    if losses.shape != logits.shape:
        raise ValueError("Mesh loss criterion must use reduction='none'.")
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        raise ValueError("Mesh batch contains no valid (non-padding) points.")
    return (losses * valid_mask.to(dtype=losses.dtype)).sum() / valid_count


def binary_mesh_intersection_union(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count binary IoU intersection/union on valid mesh rows only."""
    if predictions.shape != targets.shape or predictions.shape != valid_mask.shape:
        raise ValueError(
            "predictions, targets, and valid_mask must have identical shapes; got "
            f"{tuple(predictions.shape)}, {tuple(targets.shape)}, {tuple(valid_mask.shape)}."
        )
    valid_predictions = predictions.bool() & valid_mask.bool()
    valid_targets = targets.bool() & valid_mask.bool()
    intersection = torch.logical_and(valid_predictions, valid_targets).sum()
    union = torch.logical_or(valid_predictions, valid_targets).sum()
    return intersection, union


def binary_mesh_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Return binary IoU on valid mesh rows, with an explicit empty-union guard."""
    intersection, union = binary_mesh_intersection_union(predictions, targets, valid_mask)
    return intersection.to(dtype=torch.float32) / (union.to(dtype=torch.float32) + epsilon)
