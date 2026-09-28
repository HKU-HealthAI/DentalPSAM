"""Original equal-triangle measurement used by the MICCAI pipeline.

Three details are intentionally preserved: UV centres are truncated to integer
pixels, decisions use strict ``score > threshold``, and metrics are computed
per mesh before averaging. These are different from the optional area audit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dentalpsam.mesh_io import read_ply_mesh, view_to_mesh_face_indices

VIEWS = ("up", "in", "out")


def read_mesh_ids(path: Path) -> list[str]:
    """Read a fixed list without silently filtering or deduplicating cases."""
    mesh_ids = [
        Path(line.strip()).stem
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not mesh_ids or len(mesh_ids) != len(set(mesh_ids)):
        raise ValueError("Mesh list must be nonempty and contain unique mesh IDs")
    if any(len(name) != 6 or not name.isdigit() for name in mesh_ids):
        raise ValueError(
            "Expected six-digit mesh IDs: four participant digits plus arch"
        )
    return mesh_ids


def load_mesh_predictions(data_dir: Path, prediction_dir: Path, mesh_id: str):
    """Return targets and branch probabilities in concatenated UV-face order.

    Face labels reproduce the historical per-channel minimum RGB rule exactly.
    On grayscale black/white labels this means any black vertex makes the face
    plaque. Each UV face must occur exactly once; mismatches raise, never truncate.
    """
    return load_mesh_prediction_files(
        data_dir / "label" / f"{mesh_id}.ply",
        data_dir / "manual_2D" / "info" / f"{mesh_id}.npz",
        prediction_dir,
        mesh_id,
    )


def load_mesh_prediction_files(
    label_path: Path, metadata_path: Path, prediction_dir: Path, mesh_id: str
):
    """Read aligned targets and probabilities from explicit public file paths."""
    vertices, colours, faces = read_ply_mesh(label_path)
    targets, images, meshes, triangles = [], [], [], []
    with np.load(metadata_path) as info:
        for view_index, view in enumerate(VIEWS):
            tri = np.asarray(info[f"tri_{view}"], dtype=np.int64)
            # Preserve float32 mean rounding at integer pixel boundaries.
            uv = np.asarray(info[f"uvpx_{view}"])
            if (
                tri.ndim != 2
                or tri.shape[1] != 3
                or np.any(tri < 0)
                or np.any(tri >= len(vertices))
            ):
                raise ValueError(f"Invalid triangle indices: {mesh_id}/{view}")
            if uv.shape != (len(vertices), 2) or not np.isfinite(uv).all():
                raise ValueError(f"Invalid UV coordinates: {mesh_id}/{view}")
            image_path = prediction_dir / f"{mesh_id}_{view_index}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            # Historical code uses astype(int32), not rounding or interpolation.
            centres = uv[tri].mean(axis=1).astype(np.int32)
            x = np.clip(centres[:, 0], 0, image.shape[1] - 1)
            y = np.clip(centres[:, 1], 0, image.shape[0] - 1)
            images.append(image.astype(np.float64)[y, x] / 255.0)
            with np.load(
                prediction_dir / "3Dpred" / f"{mesh_id}_{view_index}.npz"
            ) as prediction:
                score = np.asarray(prediction["concatenated_array"]).reshape(-1)
            if (
                len(score) != len(tri)
                or not np.isfinite(score).all()
                or np.any((score < 0) | (score > 1))
            ):
                raise ValueError(f"Invalid 3D probability/count: {mesh_id}/{view}")
            meshes.append(score)
            targets.append(np.all(np.min(colours[tri], axis=1) == 0.0, axis=1))
            triangles.append(tri)
    joined = np.concatenate(triangles)
    if len(np.unique(np.sort(joined, axis=1), axis=0)) != len(joined):
        raise ValueError(f"Duplicate UV triangles: {mesh_id}")
    view_to_mesh_face_indices(joined, faces)
    return np.concatenate(targets), np.concatenate(images), np.concatenate(meshes)


def equal_face_metrics(target: np.ndarray, scores: np.ndarray, threshold: float = 0.5):
    """Compute historical mesh-level metrics, including empty-class conventions."""
    prediction = np.asarray(scores) > threshold
    target = np.asarray(target, dtype=bool)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction must have the same nonempty shape")
    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    fn = int(np.sum(target & ~prediction))
    tn = int(np.sum(~target & ~prediction))

    def divide(numerator, denominator, empty=0.0):
        return float(numerator / denominator) if denominator else empty

    # compute_metrics_tri returns 1 for an empty union; the old confusion-
    # matrix F1 returns 0 when no positives exist. Keep both explicit.
    plaque_iou = divide(tp, tp + fp + fn, 1.0)
    nonplaque_iou = divide(tn, tn + fp + fn, 1.0)
    plaque_dice = divide(2 * tp, 2 * tp + fp + fn, 1.0)
    nonplaque_dice = divide(2 * tn, 2 * tn + fp + fn, 1.0)
    return {
        "plaque_iou": plaque_iou,
        "plaque_dice": plaque_dice,
        "nonplaque_iou": nonplaque_iou,
        "nonplaque_dice": nonplaque_dice,
        "mean_iou": (plaque_iou + nonplaque_iou) / 2,
        "mean_dice": (plaque_dice + nonplaque_dice) / 2,
        "accuracy": divide(tp + tn, len(target)),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
    }


def cluster_bootstrap(rows: list[dict], repetitions: int, seed: int) -> dict:
    """Bootstrap participants while preserving the original mesh-macro estimator.

    Resampling a participant brings along all of that participant's meshes.
    The point estimate remains the mean over meshes, not over pooled faces.
    """
    patients = sorted({row["mesh_id"][:4] for row in rows})
    metrics = [key for key in rows[0] if key != "mesh_id"]
    counts = np.array(
        [sum(row["mesh_id"].startswith(pid) for row in rows) for pid in patients]
    )
    indices = np.random.default_rng(seed).integers(
        0, len(patients), (repetitions, len(patients))
    )
    result = {}
    for metric in metrics:
        sums = np.array(
            [
                sum(row[metric] for row in rows if row["mesh_id"].startswith(pid))
                for pid in patients
            ]
        )
        estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        result[metric] = {
            "mean": float(np.mean([row[metric] for row in rows])),
            "ci_low": float(np.quantile(estimates, 0.025)),
            "ci_high": float(np.quantile(estimates, 0.975)),
        }
    return result
