#!/usr/bin/env python3
"""Verify cached SOTA score ordering and reject geometry mismatches."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_evaluate_mesh import object_parts, write_binary_ply
from dentalpsam.mesh_io import read_ply_mesh, view_to_mesh_face_indices


def read_cached_scores(data_dir, mesh_id):
    """Reference restoration used to test cached patch-to-face correspondence."""
    xyz, _colours, faces = read_ply_mesh(data_dir / "origin" / f"{mesh_id}.ply")
    joined_faces, joined_scores = [], []
    with np.load(
        data_dir / "manual_2D/SOTA_mesh" / f"{mesh_id}.npz", allow_pickle=True
    ) as cache, np.load(data_dir / "manual_2D/info" / f"{mesh_id}.npz") as info:
        for view in ("up", "in", "out"):
            rows = np.asarray(
                [row for part in cache[view] for row in part], dtype=np.float64
            ).reshape(-1, 10)
            order = np.asarray(
                [index for part in cache[f"face_order_{view}"] for index in part],
                dtype=int,
            )
            tri = np.asarray(info[f"tri_{view}"], dtype=int)
            if len(rows) != len(tri) or sorted(order.tolist()) != list(range(len(tri))):
                raise ValueError(f"Invalid cache face permutation: {mesh_id}/{view}")
            if not np.allclose(
                rows[:, :9], xyz[tri[order]].reshape(-1, 9), rtol=0, atol=1e-6
            ):
                raise ValueError(
                    f"Cached geometry does not match original PLY: {mesh_id}/{view}"
                )
            scores = np.empty(len(order), dtype=np.float64)
            scores[order] = rows[:, 9]
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Invalid cached score range: {mesh_id}/{view}")
            joined_faces.append(tri)
            joined_scores.append(scores)
    mapping = view_to_mesh_face_indices(np.concatenate(joined_faces), faces)
    if len(set(mapping.tolist())) != len(faces):
        raise ValueError(f"Duplicate original face mapping: {mesh_id}")
    scores = np.empty(len(faces), dtype=np.float64)
    scores[mapping] = np.concatenate(joined_scores)
    return scores, faces


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
