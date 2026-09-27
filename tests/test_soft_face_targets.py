"""Keep grayscale training supervision separate from binary test ground truth."""

import numpy as np
import pytest

from dentalpsam.branch3d.features import patch_rows, soft_face_targets


def test_grayscale_targets_match_original_exporter():
    colours = np.repeat(np.array([[0], [32], [128], [255]], dtype=np.uint8), 3, axis=1)
    faces = np.array([[0, 2, 3], [1, 2, 3], [2, 3, 3], [3, 3, 3]])
    # Literal NumPy calculation in the original exporter after Open3D RGB loading.
    vertex_colours = colours.astype(np.float64) / 255.0
    face_colours = np.min(vertex_colours[faces], axis=1)
    expected = (1 - face_colours)[:, 0]
    actual = soft_face_targets(colours, faces)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, [1, 1 - 32 / 255, 1 - 128 / 255, 0])
    assert actual.dtype == np.float64
    binary_classifier_target = np.all(colours[faces].min(axis=1) == 0, axis=1)
    np.testing.assert_array_equal(binary_classifier_target, [True, False, False, False])
    assert np.count_nonzero((actual > 0) & (actual < 1)) == 2


def test_export_keeps_first_channel_and_face_order():
    # Preserve the source's channel choice, even for non-grayscale RGB inputs.
    colours = np.array([[64, 32, 96], [128, 255, 128], [255, 255, 255]])
    faces = np.array([[2, 1, 1], [0, 1, 2]])
    target = soft_face_targets(colours, faces)
    np.testing.assert_array_equal(target, [1 - 128 / 255, 1 - 64 / 255])
    vertices = np.arange(9, dtype=float).reshape(3, 3)
    uv = np.zeros((3, 2))
    parts, order = patch_rows("up", faces, uv, vertices, target)
    np.testing.assert_array_equal(parts[0][:, 9], target)
    np.testing.assert_array_equal(parts[0][:, :9], vertices[faces].reshape(2, 9))
    np.testing.assert_array_equal(order[0], [0, 1])


@pytest.mark.parametrize("colours", [np.zeros((3, 2)), np.full((3, 3), np.nan),
                                    np.full((3, 3), -1), np.full((3, 3), 256)])
def test_invalid_annotation_colours_fail(colours):
    with pytest.raises(ValueError):
        soft_face_targets(colours, np.array([[0, 1, 2]]))
