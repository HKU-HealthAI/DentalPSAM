#!/usr/bin/env python3
"""Regression checks for the original equal-face protocol; no study data."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentalpsam.evaluation import (
    cluster_bootstrap, equal_face_metrics, load_mesh_predictions, read_mesh_ids,
)
from tests.test_evaluate_mesh import write_binary_ply


def main():
    result = equal_face_metrics(np.array([1, 1, 0, 0]), np.array([0.8, 0.5, 0.6, 0.1]))
    assert result["plaque_iou"] == 1 / 3
    assert result["plaque_dice"] == 0.5
    assert result["accuracy"] == 0.5
    empty = equal_face_metrics(np.array([0, 0]), np.array([0.5, 0.0]))
    assert empty["plaque_iou"] == empty["plaque_dice"] == 1
    assert empty["f1"] == 0
    rows = [{"mesh_id": "000101", "iou": 0.0}, {"mesh_id": "000102", "iou": 0.0},
            {"mesh_id": "000201", "iou": 1.0}]
    bootstrap = cluster_bootstrap(rows, 2000, 42)
    assert bootstrap == cluster_bootstrap(rows, 2000, 42)
    assert bootstrap["iou"]["mean"] == 1 / 3  # Mesh macro, not patient macro.
    assert bootstrap["iou"]["ci_low"] == 0 and bootstrap["iou"]["ci_high"] == 1

    with tempfile.TemporaryDirectory(prefix="dentalpsam_original_eval_") as temporary:
        root = Path(temporary)
        (root / "label").mkdir()
        (root / "manual_2D/info").mkdir(parents=True)
        (root / "pred/3Dpred").mkdir(parents=True)
        write_binary_ply(root / "label/000101.ply", label_colours=True)
        triangles = np.array([[3, 4, 5], [0, 1, 2]])
        uv = np.array([[0.8, 0.8]] * 3 + [[1.8, 0.8]] * 3)
        info = {}
        for index, view in enumerate(("up", "in", "out")):
            info[f"tri_{view}"] = triangles if index == 0 else np.empty((0, 3), dtype=int)
            info[f"uvpx_{view}"] = uv
            cv2.imwrite(str(root / f"pred/000101_{index}.png"), np.array([[0, 255], [64, 128]], dtype=np.uint8))
            np.savez(root / f"pred/3Dpred/000101_{index}.npz",
                     concatenated_array=np.array([0.2, 0.7]) if index == 0 else np.empty(0))
        np.savez(root / "manual_2D/info/000101.npz", **info)
        target, image, mesh = load_mesh_predictions(root, root / "pred", "000101")
        np.testing.assert_array_equal(target, [False, True])
        np.testing.assert_array_equal(image, [1.0, 0.0])  # Truncation, not rounding.
        np.testing.assert_array_equal(mesh, [0.2, 0.7])
        (root / "list.txt").write_text("000101.ply\n000102\n")
        assert read_mesh_ids(root / "list.txt") == ["000101", "000102"]
        (root / "list.txt").write_text("000101\n000101\n")
        try:
            read_mesh_ids(root / "list.txt")
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate mesh IDs were accepted")
        info["tri_in"] = triangles[:1]
        np.savez(root / "manual_2D/info/000101.npz", **info)
        np.savez(root / "pred/3Dpred/000101_1.npz", concatenated_array=np.array([0.1]))
        try:
            load_mesh_predictions(root, root / "pred", "000101")
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate faces were accepted")
    print("PASS: original equal-face metrics, UV truncation, strict threshold, clustered CI, and input guards")


if __name__ == "__main__":
    main()
