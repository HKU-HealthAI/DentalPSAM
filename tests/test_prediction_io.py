#!/usr/bin/env python3
"""Dependency-light tests for prediction/evaluation I/O helpers."""

from __future__ import annotations

import sys
import argparse
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dentalpsam.prediction_io import (  # noqa: E402
    flatten_patch_lists,
    load_face_orders,
    mesh_names_from_label_mesh,
    mesh_names_from_origin,
    read_mesh_list,
    restore_face_order,
)


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected!r}, got {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-test DentalPSAM prediction I/O helpers.")
    parser.add_argument(
        "--require_numpy",
        action="store_true",
        help="Fail instead of skipping the NPZ/face-order checks when NumPy is unavailable.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manual_2d = root / "manual_2D"
        (manual_2d / "label_mesh").mkdir(parents=True)
        (manual_2d / "origin").mkdir(parents=True)

        split_file = root / "split.txt"
        split_file.write_text("\n# comment\n001201\n  # skipped\n014201\n")
        assert_equal(read_mesh_list(split_file), ["001201", "014201"], "mesh list parsing")

        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            if args.require_numpy:
                raise RuntimeError("NumPy is required for full prediction_io self-test.") from exc
            print("prediction_io_self_test_skipped_numpy")
            return

        np.savez(
            manual_2d / "label_mesh" / "001201.npz",
            face_order_up=np.array([np.array([2, 0]), np.array([1])], dtype=object),
            face_order_in=np.array([np.array([1]), np.array([0, 2])], dtype=object),
            face_order_out=np.array([np.array([0, 2]), np.array([1])], dtype=object),
        )
        np.savez(
            manual_2d / "label_mesh" / "014201.npz",
            face_order_up=np.array([np.array([0])], dtype=object),
            face_order_in=np.array([np.array([0])], dtype=object),
            face_order_out=np.array([np.array([0])], dtype=object),
        )

        for stem in ["001201_0", "001201_1", "001201_2", "014201_0"]:
            (manual_2d / "origin" / f"{stem}.png").write_bytes(b"")

        assert_equal(mesh_names_from_label_mesh(manual_2d), ["001201", "014201"], "label_mesh mesh ids")
        assert_equal(mesh_names_from_origin(manual_2d), ["001201", "014201"], "origin mesh ids")
        assert_equal(flatten_patch_lists([[2, 0], [1]]), [2, 0, 1], "patch flattening")

        face_order_up, face_order_in, face_order_out = load_face_orders(manual_2d, "001201")
        assert_equal(face_order_up, [2, 0, 1], "face_order_up")
        assert_equal(face_order_in, [1, 0, 2], "face_order_in")
        assert_equal(face_order_out, [0, 2, 1], "face_order_out")

        restored = restore_face_order([0.2, 0.4, 0.6], face_order_up)
        np.testing.assert_allclose(restored, np.array([0.4, 0.6, 0.2], dtype=np.float32))

        try:
            restore_face_order([0.1, 0.2], face_order_up)
        except ValueError as exc:
            if "does not match face-order count" not in str(exc):
                raise
        else:
            raise AssertionError("restore_face_order should fail on length mismatch")

        try:
            restore_face_order([0.1, 0.2, 0.3], [0, 0, 2])
        except ValueError as exc:
            if "complete permutation" not in str(exc):
                raise
        else:
            raise AssertionError("restore_face_order should fail on duplicate face indices")

    print("prediction_io_self_test_ok")


if __name__ == "__main__":
    main()
