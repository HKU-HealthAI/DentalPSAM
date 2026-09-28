"""Render and export the same face predictions used by the mesh evaluator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ._layout import SplitLayout
from .mesh_io import read_ply_mesh, view_to_mesh_face_indices


PLAQUE = np.array([0.05, 0.35, 0.85])
TOOTH = np.array([0.89, 0.88, 0.84])


def load_case(data: Path, results: Path, mesh_id: str):
    """Restore evaluated view-order values to the original mesh face order."""
    from .evaluation import VIEWS, equal_face_metrics, load_mesh_prediction_files

    layout = SplitLayout.read(data)
    vertices, colours, faces = read_ply_mesh(layout.meshes / f"{mesh_id}.ply")
    label_path = layout.labels / f"{mesh_id}.ply"
    label_vertices, _, label_faces = read_ply_mesh(label_path)
    if not np.array_equal(faces, label_faces) or not np.array_equal(vertices, label_vertices):
        raise ValueError(f"Scan and annotation geometry differ: {mesh_id}")
    metadata = results / "processed" / "metadata" / f"{mesh_id}.npz"
    target, image_score, mesh_score = load_mesh_prediction_files(
        label_path, metadata, results / "predictions", mesh_id
    )
    score = 0.5 * image_score + 0.5 * mesh_score
    with np.load(metadata, allow_pickle=False) as info:
        view_faces = np.concatenate([info[f"tri_{view}"] for view in VIEWS])
    indices = view_to_mesh_face_indices(view_faces, faces)
    target_in_mesh_order = np.empty(len(faces), dtype=bool)
    score_in_mesh_order = np.empty(len(faces), dtype=np.float64)
    target_in_mesh_order[indices] = target
    score_in_mesh_order[indices] = score
    return {
        "vertices": vertices,
        "faces": faces,
        "colours": colours,
        "target": target_in_mesh_order,
        "prediction": score_in_mesh_order > 0.5,
        "metrics": equal_face_metrics(target, score),
    }


def write_coloured_mesh(path: Path, vertices, faces, face_colours):
    """Export flat face colours without blending plaque across shared vertices."""
    # Duplicate the three vertices of each face so viewers interpolate only
    # within that face. Coordinates and triangle coverage remain unchanged.
    points = np.asarray(vertices)[np.asarray(faces)].reshape(-1, 3)
    colours = np.repeat(np.asarray(face_colours), 3, axis=0)
    vertex_data = np.empty(len(points), dtype=[
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    for index, key in enumerate(("x", "y", "z")):
        vertex_data[key] = points[:, index]
    colours = np.rint(np.clip(colours, 0, 1) * 255).astype(np.uint8)
    for index, key in enumerate(("red", "green", "blue")):
        vertex_data[key] = colours[:, index]
    face_data = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_data["count"] = 3
    face_data["indices"] = np.arange(len(points)).reshape(-1, 3)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("xb") as handle:
        handle.write(header.encode("ascii"))
        vertex_data.tofile(handle)
        face_data.tofile(handle)


def render_case(case, mesh_id: str, output: Path, dpi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    vertices, faces = case["vertices"], case["faces"]
    # Align the arch plane for display; exported PLY coordinates stay intact.
    centred = vertices - vertices.mean(axis=0)
    _, basis = np.linalg.eigh(centred.T @ centred)
    basis = basis[:, ::-1]
    for index in range(3):
        if basis[np.argmax(np.abs(basis[:, index])), index] < 0:
            basis[:, index] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    display_vertices = centred @ basis
    triangles = display_vertices[faces]
    original = case["colours"][faces].mean(axis=1)
    truth = np.where(case["target"][:, None], PLAQUE, TOOTH)
    predicted = np.where(case["prediction"][:, None], PLAQUE, TOOTH)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([0.3, -0.4, 0.85])
    light /= np.linalg.norm(light)
    shade = 0.65 + 0.35 * np.abs(normals @ light)
    fig = plt.figure(figsize=(11, 8.5), facecolor="white")
    for row, (elevation, azimuth, label) in enumerate([
        (90, -90, "Top"), (25, -90, "Oblique 1"), (25, 90, "Oblique 2"),
    ]):
        elevation, azimuth = np.deg2rad([elevation, azimuth])
        toward_camera = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth), np.sin(elevation),
        ])
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
        up = np.cross(toward_camera, right)
        projected = np.column_stack([display_vertices @ right, display_vertices @ up])
        # Draw far triangles first, using the same orthographic projection
        # and depth order for the original scan, labels, and predictions.
        order = np.argsort((triangles @ toward_camera).mean(axis=1))
        low, high = projected.min(axis=0), projected.max(axis=0)
        padding = max(float((high - low).max()) * 0.05, 1e-6)
        for column, (title, colours) in enumerate([
            ("Original scan", original), ("Ground truth", truth), ("DentalPSAM", predicted),
        ]):
            ax = fig.add_subplot(3, 3, row * 3 + column + 1)
            surface = PolyCollection(
                projected[faces][order], facecolors=(colours * shade[:, None])[order],
                edgecolors="none", linewidths=0, antialiaseds=False, rasterized=True,
            )
            ax.add_collection(surface)
            ax.set(xlim=(low[0] - padding, high[0] + padding),
                   ylim=(low[1] - padding, high[1] + padding))
            ax.set_aspect("equal")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title, fontsize=14, pad=12)
            if column == 0:
                ax.text(-0.04, 0.5, label, transform=ax.transAxes, fontsize=10,
                        rotation=90, va="center", ha="right")
    metrics = case["metrics"]
    fig.suptitle(
        f"DentalPSAM | Case {mesh_id}\n"
        f"Plaque IoU {metrics['plaque_iou']:.3f}   |   Dice {metrics['plaque_dice']:.3f}",
        fontsize=16, y=0.99,
    )
    fig.text(0.5, 0.015, "Blue: plaque  |  Scores cover the complete mesh", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.05, top=0.85, wspace=0.10, hspace=0.06)
    fig.savefig(output / f"{mesh_id}.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    write_coloured_mesh(output / f"{mesh_id}_label.ply", vertices, faces, truth)
    write_coloured_mesh(output / f"{mesh_id}_prediction.ply", vertices, faces, predicted)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize original scans, ground truth, and DentalPSAM predictions.")
    parser.add_argument("--data", type=Path, required=True, help="Case directory containing meshes/, labels/, and mesh_ids.txt")
    parser.add_argument("--results", type=Path, required=True, help="Result directory containing processed/metadata/ and predictions/")
    parser.add_argument("--output", type=Path, required=True, help="New directory for PNG panels and coloured PLY meshes")
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default: 160)")
    args = parser.parse_args(argv)
    try:
        from .evaluation import read_mesh_ids

        if args.dpi < 50:
            raise ValueError("--dpi must be at least 50")
        data, results, output = (p.expanduser().resolve() for p in (args.data, args.results, args.output))
        if any(output == p or p in output.parents for p in (data, results)):
            raise ValueError("Output must be outside the source data and results")
        ids = read_mesh_ids(data / "mesh_ids.txt")
        cases = [(mid, load_case(data, results, mid)) for mid in ids]
        output.mkdir(parents=True, exist_ok=False)
        for index, (mid, case) in enumerate(cases, start=1):
            render_case(case, mid, output, args.dpi)
            print(f"[{index}/{len(cases)}] {mid}: Dice {case['metrics']['plaque_dice']:.4f}", flush=True)
        with (output / "metrics.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mesh_id", *cases[0][1]["metrics"]])
            writer.writeheader()
            writer.writerows({"mesh_id": mid, **case["metrics"]} for mid, case in cases)
        print(f"Visualizations saved to {output}")
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))
