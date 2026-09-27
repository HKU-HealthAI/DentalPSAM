#!/usr/bin/env python3
"""Render the three DentalPSAM UV views and reconstruction metadata.

Inputs are original and labelled PLY directories plus a fixed mesh list.
Outputs are written to a fresh staging directory; source data are never
modified.  The numerical renderer is the historical DentalPSAM projection
implementation, while this entry point adds validation and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from dentalpsam.uv_projection import render_single_mesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mesh_ids(path: Path) -> list[str]:
    values = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Mesh list must be non-empty and contain no duplicates.")
    return values


def assert_matching_topology(origin_mesh: o3d.geometry.TriangleMesh,
                             label_mesh: o3d.geometry.TriangleMesh,
                             mesh_id: str) -> None:
    origin_faces = np.asarray(origin_mesh.triangles)
    label_faces = np.asarray(label_mesh.triangles)
    if not np.array_equal(origin_faces, label_faces):
        raise ValueError(f"Origin/label topology mismatch: {mesh_id}")
    if len(origin_faces) == 0:
        raise ValueError(f"Mesh has no triangle faces: {mesh_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    origin_dir = args.origin_dir.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()
    mesh_list = args.mesh_list.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (origin_dir, label_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not mesh_list.is_file():
        raise FileNotFoundError(mesh_list)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    for source in (origin_dir, label_dir):
        if output_dir == source or source in output_dir.parents:
            raise ValueError("Output must not be inside a source PLY directory.")

    mesh_ids = read_mesh_ids(mesh_list)
    output_dir.mkdir(parents=True)
    cases: list[dict[str, object]] = []
    inputs: list[tuple[str, Path]] = [("mesh_list", mesh_list)]
    for index, mesh_id in enumerate(mesh_ids, start=1):
        origin_path = origin_dir / f"{mesh_id}.ply"
        label_path = label_dir / f"{mesh_id}.ply"
        if not origin_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing origin/label PLY pair: {mesh_id}")
        origin_mesh = o3d.io.read_triangle_mesh(str(origin_path))
        label_mesh = o3d.io.read_triangle_mesh(str(label_path))
        assert_matching_topology(origin_mesh, label_mesh, mesh_id)
        # The historical renderer removes duplicated triangles in memory.
        # Mirror that step before recording the expected projected face count.
        origin_mesh.remove_duplicated_triangles()
        label_mesh.remove_duplicated_triangles()
        assert_matching_topology(origin_mesh, label_mesh, mesh_id)
        face_count = len(origin_mesh.triangles)
        result = render_single_mesh(origin_mesh, label_mesh, str(output_dir), mesh_id)
        if any(value is None for value in result):
            raise RuntimeError(f"UV projection failed its map-back check: {mesh_id}")
        info_path = output_dir / "info" / f"{mesh_id}.npz"
        with np.load(info_path, allow_pickle=False) as info:
            required = {f"uvpx_{view}" for view in ("up", "in", "out")} | {
                f"tri_{view}" for view in ("up", "in", "out")
            }
            if set(info.files) != required:
                raise ValueError(f"Unexpected info NPZ keys for {mesh_id}: {info.files}")
            projected_face_count = sum(len(info[f"tri_{view}"]) for view in ("up", "in", "out"))
        if projected_face_count != face_count:
            raise ValueError(
                f"Projected faces ({projected_face_count}) do not cover PLY faces ({face_count}): {mesh_id}"
            )
        inputs.extend(((f"origin/{mesh_id}.ply", origin_path), (f"label/{mesh_id}.ply", label_path)))
        cases.append({"mesh_id": mesh_id, "face_count": face_count})
        print(f"[{index}/{len(mesh_ids)}] {mesh_id}: {face_count} faces", flush=True)

    input_digest = hashlib.sha256()
    for logical_name, path in inputs:
        input_digest.update(f"{logical_name}:{sha256_file(path)}\n".encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "mesh_count": len(mesh_ids),
        "input_file_set_sha256": input_digest.hexdigest(),
        "renderer_sha256": sha256_file(Path(__file__).parent / "dentalpsam" / "uv_projection.py"),
        "raw_data_modified": False,
        "outputs": ["origin/*.png", "label/*.png", "info/*.npz"],
        "cases": cases,
    }
    (output_dir / "projection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
