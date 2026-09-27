"""Protect the original mesh-label and auxiliary-feature export contract."""

from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from dentalpsam.branch3d.features import (
    VIEWS, patch_rows, rasterize, verify_patch_file,
)


@pytest.mark.parametrize("view, centres, expected", [
    ("up", [[-1/3, 97], [-1/3, 300], [5, -1/3], [800, 600]], [5, 2, 3, 5]),
    ("in", [[-1/3, 97], [-1/3, -1/3], [9999, 9999]], [7, 4, 7]),
    ("out", [[-1/3, 97], [-1/3, -1/3], [9999, 9999]], [7, 4, 7]),
])
def test_source_patch_assignment_including_negative_centres(view, centres, expected):
    faces = np.arange(3*len(centres)).reshape(-1, 3)
    vertices = np.arange(9*len(centres), dtype=float).reshape(-1, 3)
    uv = np.repeat(np.asarray(centres, dtype=float), 3, axis=0)
    scores = np.linspace(.1, .9, len(faces))
    parts, orders = patch_rows(view, faces, uv, vertices, scores)
    assignment = np.empty(len(faces), dtype=int)
    for patch_index, (part, order) in enumerate(zip(parts, orders)):
        assignment[order] = patch_index
        if len(order):
            np.testing.assert_array_equal(part[:, :9], vertices[faces[order]].reshape(-1, 9))
            np.testing.assert_array_equal(part[:, 9], scores[order])
        else:
            assert part.shape == (0,)  # Source exporter used np.array([]).
    np.testing.assert_array_equal(assignment, expected)


def test_invalid_source_patch_index_is_not_silently_clipped():
    with pytest.raises(ValueError, match="patch index"):
        patch_rows("up", np.array([[0, 1, 2]]), np.full((3, 2), -9999.),
                   np.ones((3, 3)), np.array([.2]))


def test_auxiliary_raster_matches_source_truncation_and_opencv_clipping():
    vertices_uv = np.array([[-.8, 1.8], [4.9, -1.2], [6.7, 6.8], [256.9, 255.9]])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    scores = np.array([.5, .123])
    expected = np.zeros((512, 768, 3), dtype=np.uint8)
    for face, score in zip(faces, scores):
        polygon = vertices_uv[face].reshape(-1, 1, 2).astype(np.int32)
        colour = tuple(int(value) for value in np.array([255, 255, 255]) * score)
        cv2.fillPoly(expected, [polygon], colour)
    np.testing.assert_array_equal(rasterize("up", faces, vertices_uv, scores), expected)


def cache_fixture():
    vertices = np.arange(18, dtype=float).reshape(6, 3)
    triangles = {"up": np.array([[0, 1, 2], [3, 4, 5]]),
                 "in": np.empty((0, 3), dtype=int), "out": np.empty((0, 3), dtype=int)}
    values = {"up": np.array([0., 1 - 32/255.]), "in": np.array([]), "out": np.array([])}
    uv = np.zeros((6, 2))
    cache = {}
    for view in VIEWS:
        cache[view], cache[f"face_order_{view}"] = patch_rows(
            view, triangles[view], uv, vertices, values[view])
    return vertices, triangles, values, cache


def test_saved_soft_targets_are_verified_against_their_source(tmp_path):
    vertices, triangles, values, cache = cache_fixture()
    path = tmp_path / "targets.npz"
    np.savez_compressed(path, **cache)
    verify_patch_file(path, vertices, triangles, values)
    # A valid probability vector is still wrong when written into the GT file.
    cache["up"][0][:, 9] = [.123, .789]
    np.savez_compressed(path, **cache)
    with pytest.raises(ValueError, match="Score/target mismatch"):
        verify_patch_file(path, vertices, triangles, values)


@pytest.mark.parametrize("corruption", ["geometry", "duplicate", "oversize"])
def test_saved_cache_rejects_broken_geometry_and_order(tmp_path, corruption):
    vertices, triangles, values, cache = cache_fixture()
    if corruption == "geometry":
        cache["up"][0][0, 0] += 1
    elif corruption == "duplicate":
        cache["face_order_up"][0][:] = 0
    else:
        cache["up"][0] = np.zeros((6001, 10))
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, **cache)
    with pytest.raises(ValueError):
        verify_patch_file(path, vertices, triangles, values)


def test_regeneration_help_without_patient_data():
    result = subprocess.run(
        [sys.executable, "-m", "dentalpsam.branch3d.features", "--help"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in ("--data", "--checkpoint", "--mesh-list", "--output"):
        assert option in result.stdout
