"""Optional augmentation transforms geometry, never labels or scan colours."""

import numpy as np

from dentalpsam.branch3d.augmentation import DataAugmentation


def test_reflection_changes_only_y_and_does_not_mutate_input():
    features = np.arange(66, dtype=float).reshape(2, 33)
    before = features.copy()
    expected = before.copy()
    expected[:, [1, 4, 7, 10, 13, 16, 19]] *= -1
    augmenter = DataAugmentation(flip_prob=1)
    np.testing.assert_array_equal(augmenter.augment(features), expected)
    np.testing.assert_array_equal(features, before)
    np.testing.assert_array_equal(augmenter.vertical_flip(expected), before)
    assert augmenter.augment(features, is_training=False) is features


def test_rotation_preserves_colours_and_rotates_geometry_and_normals():
    features = np.arange(66, dtype=float).reshape(2, 33)
    result = DataAugmentation().random_rotation_z(features, (90, 90))
    for start in (0, 3, 6, 9, 12, 15, 18):
        expected = features[:, start:start+3].copy()
        expected[:, 0] = -features[:, start+1]
        expected[:, 1] = features[:, start]
        np.testing.assert_allclose(result[:, start:start+3], expected, atol=1e-12)
    np.testing.assert_array_equal(result[:, 21:], features[:, 21:])


def test_seeded_augmentation_is_repeatable():
    features = np.arange(66, dtype=float).reshape(2, 33)
    first = DataAugmentation(enable_rotation=True, seed=42).augment(features)
    second = DataAugmentation(enable_rotation=True, seed=42).augment(features)
    np.testing.assert_array_equal(first, second)
