#!/usr/bin/env python3
"""Compare probability PNG conversion with the historical OpenCV write path."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_dentalpsam import write_probability_png


def main():
    probabilities = np.random.default_rng(42).random((128, 128)).astype(np.float32)
    probabilities[0, :6] = [0, 1, 0.5, 0.5 / 255, 127.5 / 255, 128.5 / 255]
    with tempfile.TemporaryDirectory(prefix="dentalpsam_png_test_") as temporary:
        root = Path(temporary)
        assert cv2.imwrite(str(root / "native.png"), probabilities * 255.0)
        write_probability_png(root / "clean.png", probabilities)
        native = cv2.imread(str(root / "native.png"), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(root / "clean.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(clean, native)
        assert not np.array_equal(clean, (probabilities * 255.0).astype(np.uint8))
    print("PASS: probability PNG rounding exactly matches historical OpenCV conversion")


if __name__ == "__main__":
    main()
