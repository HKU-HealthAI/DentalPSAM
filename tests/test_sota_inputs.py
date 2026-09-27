#!/usr/bin/env python3
"""Verify cached SOTA score ordering and reject geometry mismatches."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_evaluate_mesh import object_parts, write_binary_ply
from tools.audit_sota_inputs import read_cached_scores
from tools.evaluate_mesh import read_ply_mesh


def main():
    with tempfile.TemporaryDirectory(prefix="dentalpsam_sota_test_") as temporary:
        root = Path(temporary)
        for kind in ("origin", "manual_2D/info", "manual_2D/SOTA_mesh"):
            (root / kind).mkdir(parents=True)
        write_binary_ply(root / "origin/000101.ply")
        xyz, _rgb, faces = read_ply_mesh(root / "origin/000101.ply")
        tri = faces[::-1]
        order = np.array([1, 0])
        rows = np.column_stack((xyz[tri[order]].reshape(-1, 9), [0.2, 0.8]))
        cache = {
            "up": object_parts(rows),
            "in": np.empty(0, dtype=object),
            "out": np.empty(0, dtype=object),
            "face_order_up": object_parts(order),
            "face_order_in": np.empty(0, dtype=object),
            "face_order_out": np.empty(0, dtype=object),
        }
        np.savez(
            root / "manual_2D/info/000101.npz",
            tri_up=tri,
            tri_in=np.empty((0, 3), dtype=int),
            tri_out=np.empty((0, 3), dtype=int),
        )
        np.savez(root / "manual_2D/SOTA_mesh/000101.npz", **cache)
        scores, returned_faces = read_cached_scores(root, "000101")
        np.testing.assert_array_equal(returned_faces, faces)
        np.testing.assert_array_equal(scores, [0.2, 0.8])
        rows[0, 0] += 1
        cache["up"] = object_parts(rows)
        np.savez(root / "manual_2D/SOTA_mesh/000101.npz", **cache)
        try:
            read_cached_scores(root, "000101")
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched cached geometry was accepted")
    print("PASS: SOTA patch/view/PLY ordering and geometry validation")


if __name__ == "__main__":
    main()
