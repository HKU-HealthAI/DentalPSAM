#!/usr/bin/env python3
"""Data-loader regressions with synthetic images and trusted NPZ fixtures."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentalpsam.data import (
    SAMDataset,
    VIEW_IMAGE_SHAPES,
    VIEW_KEYS,
    load_and_patchify_png,
    load_and_patchify_png_permesh,
    pad,
    patchify,
)


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Malformed data was accepted")


def main():
    image = np.arange(32).reshape(4, 8)
    patches, names = patchify(image, "000101_0.png", patch_size=2)
    np.testing.assert_array_equal(patches[0], image[:2, :2])
    np.testing.assert_array_equal(patches[4], image[2:, :2])
    assert names[4] == "000101_00100"
    expect_error(patchify, np.zeros((257, 256)), "bad.png")
    values = np.arange(20, dtype=np.float32).reshape(2, 10)
    padded = pad([values], (4, 10))
    assert padded.dtype == np.float32
    np.testing.assert_array_equal(padded[0, :2], 0)
    np.testing.assert_array_equal(padded[0, 2:], values)
    assert pad([np.empty(0)], (4, 10)).dtype == np.float64
    expect_error(pad, [np.zeros((2, 9))], (4, 10))
    expect_error(pad, [np.zeros((0, 11))], (4, 10))
    expect_error(pad, [np.zeros((5, 10))], (4, 10))
    expect_error(pad, [np.full((1, 10), np.nan)], (4, 10))
    expect_error(SAMDataset, [0], [], [], [])

    with tempfile.TemporaryDirectory(prefix="dentalpsam_data_test_") as temporary:
        root = Path(temporary)
        for folder in ("origin", "label", "SOTA_mesh", "label_mesh"):
            (root / folder).mkdir()
        scores, targets = {}, {}
        for position, key in VIEW_KEYS.items():
            height, width = VIEW_IMAGE_SHAPES[position]
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = 17  # Stored BGR blue -> RGB channel 2.
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[0, :3] = [1, 127, 128]
            cv2.imwrite(str(root / f"origin/000101_{position}.png"), rgb)
            cv2.imwrite(str(root / f"label/000101_{position}.png"), mask)
            count = (height // 256) * (width // 256)
            scores[key], targets[key] = np.empty(count, dtype=object), np.empty(
                count, dtype=object
            )
            for index in range(count):
                score = np.ones((1, 10), dtype=np.float64)
                score[0, 9] = 0.25
                target = score.copy()
                target[0, 9] = 1
                scores[key][index], targets[key][index] = score, target
        np.savez(root / "SOTA_mesh/000101.npz", **scores)
        np.savez(root / "label_mesh/000101.npz", **targets)
        images, masks, indices, mesh, labels = load_and_patchify_png_permesh(
            root, "000101"
        )
        assert indices == list(range(22))
        assert images.shape == (22, 256, 256, 3)
        assert mesh.shape == labels.shape == (22, 6000, 10)
        np.testing.assert_array_equal(images[0, 0, 0], [0, 0, 17])
        np.testing.assert_array_equal(masks[0, 0, :3], [0, 0, 1])
        assert mesh[0, -1, 9] == 0.25 and labels[0, -1, 9] == 1
        subset = load_and_patchify_png_permesh(root, "000101", min_positive_pixels=1)
        assert subset[2] == [0, 6, 14]
        trained = load_and_patchify_png(root, min_positive_pixels=1)
        for actual, expected in zip(
            trained, (subset[0], subset[1], subset[3], subset[4])
        ):
            np.testing.assert_array_equal(actual, expected)
        sample = SAMDataset(images, masks, mesh, labels)[0]
        assert tuple(sample["pixel_values"].shape) == (3, 256, 256)
        assert tuple(sample["ground_truth_mask"].shape) == (1, 256, 256)
        scores["up"] = scores["up"][:-1]
        np.savez(root / "SOTA_mesh/000101.npz", **scores)
        expect_error(load_and_patchify_png_permesh, root, "000101")
    print(
        "PASS: aligned loader, left padding, row-major ordering, threshold and error guards"
    )


if __name__ == "__main__":
    main()
