"""The displayed labels must match the evaluator's face order and threshold."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from dentalpsam.mesh_io import read_ply_mesh
from dentalpsam.visualization import load_case, write_coloured_mesh
from tests.test_evaluate_mesh import write_binary_ply


def test_visualization_restores_face_order_and_preserves_strict_threshold(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    for path in (data / "meshes", data / "labels", results / "processed/metadata", results / "predictions/3Dpred"):
        path.mkdir(parents=True)
    write_binary_ply(data / "meshes/000101.ply")
    write_binary_ply(data / "labels/000101.ply", label_colours=True)
    info = {}
    for index, view in enumerate(("up", "in", "out")):
        info[f"tri_{view}"] = np.array([[3, 4, 5], [0, 1, 2]]) if index == 0 else np.empty((0, 3), dtype=int)
        info[f"uvpx_{view}"] = np.array([[0.8, 0.8]] * 3 + [[1.8, 0.8]] * 3)
        cv2.imwrite(str(results / f"predictions/000101_{index}.png"), np.array([[255, 0]], dtype=np.uint8))
        # UV order is the reverse of PLY order. The non-plaque face's fused
        # score is exactly 0.5 and must stay negative.
        score = np.array([1.0, 0.8]) if index == 0 else np.empty(0)
        np.savez(results / f"predictions/3Dpred/000101_{index}.npz", concatenated_array=score)
    np.savez(results / "processed/metadata/000101.npz", **info)
    case = load_case(data, results, "000101")
    np.testing.assert_array_equal(case["target"], [True, False])
    np.testing.assert_array_equal(case["prediction"], [True, False])
    assert case["metrics"]["plaque_iou"] == 1.0
    assert case["metrics"]["plaque_dice"] == 1.0


def test_ply_export_preserves_geometry_and_does_not_blend_shared_vertices(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    colours = np.array([[0, 0, 1], [1, 1, 1]], dtype=float)
    output = tmp_path / "mesh.ply"
    write_coloured_mesh(output, vertices, faces, colours)
    written_vertices, written_colours, written_faces = read_ply_mesh(output)
    np.testing.assert_array_equal(written_vertices[written_faces], vertices[faces])
    np.testing.assert_array_equal(written_colours[written_faces], np.repeat(colours[:, None, :], 3, axis=1))
    with pytest.raises(FileExistsError):
        write_coloured_mesh(output, vertices, faces, colours)
