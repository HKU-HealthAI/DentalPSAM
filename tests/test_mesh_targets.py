#!/usr/bin/env python3
"""Focused regression test for padded label-mesh supervision semantics."""

from __future__ import annotations

import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dentalpsam.mesh_targets import (  # noqa: E402
    binary_mesh_intersection_union,
    binary_mesh_targets,
    masked_mesh_loss,
    soft_mesh_targets,
    valid_mesh_point_mask,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    # Two real rows followed by two left-padding rows. The second row has a
    # geometry vector summing to zero, but is valid because it is nonzero.
    labels = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00],
        ]]
    )
    valid = valid_mesh_point_mask(labels)
    require(valid.squeeze(-1).tolist() == [[True, True, False, False]], "padding mask is wrong")
    require(torch.equal(soft_mesh_targets(labels), labels[..., 9:]), "soft target channel is wrong")
    require(
        binary_mesh_targets(labels, 0.5).squeeze(-1).tolist() == [[True, False, False, False]],
        "target threshold is wrong",
    )

    predictions = torch.tensor([[[True], [False], [True], [True]]])
    targets = binary_mesh_targets(labels, 0.5)
    intersection, union = binary_mesh_intersection_union(predictions, targets, valid)
    require(intersection.item() == 1 and union.item() == 1, "padding leaked into IoU counts")

    logits = torch.tensor([[[2.0], [-2.0], [8.0], [8.0]]])
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    observed_loss = masked_mesh_loss(criterion, logits, soft_mesh_targets(labels), valid)
    expected_loss = criterion(logits[:, :2], labels[:, :2, 9:]).mean()
    require(torch.allclose(observed_loss, expected_loss), "padding leaked into mesh loss")
    print("mesh_targets_self_test_ok")


if __name__ == "__main__":
    main()
