"""Mesh geometry and face-order helpers shared by inference and evaluation."""

from __future__ import annotations
from pathlib import Path
import numpy as np

PLY_SCALARS = {
    "char": "i1",
    "uchar": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "float": "<f4",
    "double": "<f8",
}


def read_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read vertex coordinates/colors and triangular faces from a binary PLY."""
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError(f"{path}: expected binary_little_endian")
        vertex_count = None
        properties: list[tuple[str, str]] = []
        current_element = None
        face_count = None
        face_list_types: tuple[str, str] | None = None
        while True:
            line = handle.readline().decode("ascii").strip()
            if line == "end_header":
                break
            fields = line.split()
            if fields[:1] == ["element"]:
                current_element = fields[1]
                if current_element == "vertex":
                    vertex_count = int(fields[2])
                elif current_element == "face":
                    face_count = int(fields[2])
            elif current_element == "vertex" and fields[:1] == ["property"]:
                if fields[1] == "list":
                    raise ValueError(f"{path}: unsupported list-valued vertex property")
                properties.append((fields[1], fields[2]))
            elif current_element == "face" and fields[:2] == ["property", "list"]:
                face_list_types = (fields[2], fields[3])
        if vertex_count is None or face_count is None or face_list_types is None:
            raise ValueError(f"{path}: missing vertex or face element")
        dtype = np.dtype([(name, PLY_SCALARS[kind]) for kind, name in properties])
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)
        count_kind, index_kind = face_list_types
        if count_kind not in PLY_SCALARS or index_kind not in PLY_SCALARS:
            raise ValueError(f"{path}: unsupported face list types {face_list_types}")
        face_dtype = np.dtype(
            [
                ("count", PLY_SCALARS[count_kind]),
                ("indices", PLY_SCALARS[index_kind], (3,)),
            ]
        )
        face_data = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if len(face_data) != face_count or np.any(face_data["count"] != 3):
            raise ValueError(f"{path}: expected {face_count} triangular faces")
    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    vertices = np.column_stack([data[key] for key in ("x", "y", "z")]).astype(
        np.float64
    )
    colors = (
        np.column_stack([data[key] for key in ("red", "green", "blue")]).astype(
            np.float64
        )
        / 255.0
    )
    return vertices, colors, face_data["indices"].astype(np.int64)


def triangle_areas(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return each triangular face's physical 3D area."""
    points = vertices[triangles]
    return 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )


def canonical_triangles(triangles: np.ndarray) -> np.ndarray:
    keys = np.sort(triangles, axis=1)
    return keys[np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))]


def view_to_mesh_face_indices(
    view_triangles: np.ndarray, mesh_faces: np.ndarray
) -> np.ndarray:
    """Map concatenated UV-view triangles back to their original PLY face order."""
    view_keys = np.sort(view_triangles, axis=1)
    mesh_keys = np.sort(mesh_faces, axis=1)
    view_order = np.lexsort((view_keys[:, 2], view_keys[:, 1], view_keys[:, 0]))
    mesh_order = np.lexsort((mesh_keys[:, 2], mesh_keys[:, 1], mesh_keys[:, 0]))
    if not np.array_equal(view_keys[view_order], mesh_keys[mesh_order]):
        raise ValueError("UV views are not an exact partition of mesh faces")
    indices = np.empty(len(view_triangles), dtype=np.int64)
    indices[view_order] = mesh_order
    return indices
