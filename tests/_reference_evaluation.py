#!/usr/bin/env python3
"""Reference evaluator retained only for face/area/vertex regression tests.

Ground truth is the existing binary ``label_mesh`` face label.  For each case,
the three UV views must form a disjoint triangle partition.  The script rejects
missing or length-mismatched predictions and reports patient-clustered
percentile bootstrap confidence intervals plus paired sign-flip p-values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentalpsam.mesh_io import (PLY_SCALARS, sha256_file, sha256_lines, hash_named_files, read_ply_mesh, triangle_areas, canonical_triangles, view_to_mesh_face_indices)


VIEW_NAMES = ("up", "in", "out")
VIEW_DIMS = {"up": (512, 768), "in": (256, 2048), "out": (256, 2048)}
METRICS = (
    "plaque_iou",
    "plaque_dsc",
    "oa",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "auc",
    "true_coverage",
    "pred_coverage",
)
SUPPORTED_PREDICTION_TYPES = {"3dpred", "face_npy", "png", "fusion"}
POSITIVE_RULES = {"greater_equal", "strict_greater"}
TARGET_POSITIVE_RULES = {"greater_equal", "strict_less"}
TARGET_SOURCES = {"label_ply_any_exact_black_vertex", "manual_label_mesh"}
SCORE_TRANSFORMS = {"identity", "one_minus"}
COMPARISON_FIELDS = (
    "primary_method",
    "comparator",
    "unit",
    "metric",
    "mean_difference",
    "difference_ci_low",
    "difference_ci_high",
    "p_value",
    "n_patients",
    "permutation_reps",
    "seed",
    "p_value_holm",
)


def normalize_mesh_id(value: str) -> str:
    stem = Path(value.strip()).stem
    return stem.rsplit("_", 1)[0] if stem.endswith(("_0", "_1", "_2")) else stem


def read_mesh_ids(path: Path) -> list[str]:
    result = []
    seen = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            mesh_id = normalize_mesh_id(line)
            if mesh_id not in seen:
                seen.add(mesh_id)
                result.append(mesh_id)
    return result








def target_input_files(
    data_dir: Path,
    mesh_ids: list[str],
    target_source: str = "label_ply_any_exact_black_vertex",
) -> list[tuple[str, Path]]:
    """List the exact original-mesh and target files consumed by evaluation."""
    manual = data_dir / "manual_2D"
    files: list[tuple[str, Path]] = []
    for mesh_id in mesh_ids:
        case_files = [
            (f"origin/{mesh_id}.ply", data_dir / "origin" / f"{mesh_id}.ply"),
            (f"manual_2D/info/{mesh_id}.npz", manual / "info" / f"{mesh_id}.npz"),
        ]
        if target_source == "label_ply_any_exact_black_vertex":
            case_files.append((f"label/{mesh_id}.ply", data_dir / "label" / f"{mesh_id}.ply"))
        elif target_source == "manual_label_mesh":
            case_files.append((
                f"manual_2D/label_mesh/{mesh_id}.npz",
                manual / "label_mesh" / f"{mesh_id}.npz",
            ))
        else:
            raise ValueError(f"Unsupported target source: {target_source!r}")
        files.extend(case_files)
    return files


def prediction_input_files(spec: dict[str, object], mesh_ids: list[str]) -> list[tuple[str, Path]]:
    """List the exact saved probability files for one method specification."""
    kind = str(spec["type"])
    root = Path(str(spec["path"]))
    files: list[tuple[str, Path]] = []
    for mesh_id in mesh_ids:
        if kind == "3dpred":
            files.extend([
                (f"{mesh_id}_{index}.npz", root / f"{mesh_id}_{index}.npz")
                for index in range(3)
            ])
        elif kind == "face_npy":
            files.append((f"{mesh_id}.npy", root / f"{mesh_id}.npy"))
        elif kind == "png":
            files.extend([
                (f"{mesh_id}_{index}.png", root / f"{mesh_id}_{index}.png")
                for index in range(3)
            ])
        elif kind == "fusion":
            png_root = Path(str(spec["png_path"]))
            files.extend([
                (f"3dpred/{mesh_id}_{index}.npz", root / f"{mesh_id}_{index}.npz")
                for index in range(3)
            ])
            files.extend([
                (f"png/{mesh_id}_{index}.png", png_root / f"{mesh_id}_{index}.png")
                for index in range(3)
            ])
        else:
            raise ValueError(f"Unsupported prediction type {kind!r}")
    return files


def validate_specs(specs: object, default_threshold: float) -> list[dict[str, object]]:
    """Validate method specifications before reading any prediction files.

    A result table is only meaningful when every column has an unambiguous
    name, input representation, and threshold.  This small validation layer
    prevents duplicate names from silently overwriting patient-level results.
    """
    if not isinstance(specs, list) or not specs:
        raise ValueError("spec-json must be a non-empty JSON list")
    validated: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw_spec in enumerate(specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Method specification {index} is not an object")
        name = raw_spec.get("name")
        kind = raw_spec.get("type")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Method specification {index} needs a non-empty name")
        if name in names:
            raise ValueError(f"Duplicate method name: {name!r}")
        if kind not in SUPPORTED_PREDICTION_TYPES:
            raise ValueError(
                f"{name}: type must be one of {sorted(SUPPORTED_PREDICTION_TYPES)}, got {kind!r}"
            )
        if not isinstance(raw_spec.get("path"), str) or not raw_spec["path"]:
            raise ValueError(f"{name}: missing non-empty prediction path")
        if kind == "fusion":
            if not isinstance(raw_spec.get("png_path"), str) or not raw_spec["png_path"]:
                raise ValueError(f"{name}: fusion input needs a non-empty png_path")
            # Keep this in the input contract rather than hard-coding an
            # experimental choice in the evaluator.  The default preserves
            # the original equal-weight implementation for older specs.
            fusion_weight_2d = float(raw_spec.get("fusion_weight_2d", 0.5))
            if not math.isfinite(fusion_weight_2d) or not 0.0 <= fusion_weight_2d <= 1.0:
                raise ValueError(f"{name}: fusion_weight_2d must be a finite value in [0, 1]")
            fusion_binarize_2d = raw_spec.get("fusion_binarize_2d", False)
            if not isinstance(fusion_binarize_2d, bool):
                raise ValueError(f"{name}: fusion_binarize_2d must be true or false")
            fusion_2d_threshold = float(raw_spec.get("fusion_2d_threshold", 0.5))
            if not math.isfinite(fusion_2d_threshold) or not 0.0 <= fusion_2d_threshold <= 1.0:
                raise ValueError(f"{name}: fusion_2d_threshold must be a finite value in [0, 1]")
        threshold = float(raw_spec.get("threshold", default_threshold))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name}: threshold must be a finite probability in [0, 1]")
        positive_rule = str(raw_spec.get("positive_rule", "greater_equal"))
        if positive_rule not in POSITIVE_RULES:
            raise ValueError(f"{name}: positive_rule must be one of {sorted(POSITIVE_RULES)}")
        score_transform = str(raw_spec.get("score_transform", "identity"))
        if score_transform not in SCORE_TRANSFORMS:
            raise ValueError(f"{name}: score_transform must be one of {sorted(SCORE_TRANSFORMS)}")
        names.add(name)
        spec = dict(raw_spec)
        spec["positive_rule"] = positive_rule
        spec["score_transform"] = score_transform
        if kind == "fusion":
            spec["fusion_weight_2d"] = fusion_weight_2d
            spec["fusion_binarize_2d"] = fusion_binarize_2d
            spec["fusion_2d_threshold"] = fusion_2d_threshold
        validated.append(spec)
    return validated


def validate_probability_scores(scores: np.ndarray, mesh_id: str, method: str) -> np.ndarray:
    """Reject logits, NaNs, and values outside the documented probability range."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{method}/{mesh_id}: prediction contains NaN or infinity")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError(f"{method}/{mesh_id}: prediction must contain probabilities in [0, 1]")
    return scores






def sorted_view_labels(
    label_npz: np.lib.npyio.NpzFile,
    view: str,
    target_threshold: float,
    target_positive_rule: str,
) -> np.ndarray:
    """Restore binary face labels using the declared continuous-label threshold.

    ``label_mesh`` channel 9 contains values in ``[0, 1]``.  It must not be
    treated as binary merely because a value is nonzero: interpolation can
    produce small positive boundary values.  The threshold is therefore an
    explicit part of the evaluation protocol and manifest.
    """
    values, order = [], []
    for part, face_order in zip(label_npz[view], label_npz[f"face_order_{view}"]):
        part = np.asarray(part)
        face_order = np.asarray(face_order)
        if part.ndim != 2 or part.shape[0] == 0:
            continue
        if part.shape[0] != face_order.shape[0]:
            raise ValueError(f"{view}: label/face-order length mismatch")
        values.append(part[:, 9])
        order.append(face_order.astype(np.int64))
    if not order:
        return np.empty(0, dtype=np.int8)
    values = np.concatenate(values)
    order = np.concatenate(order)
    if np.any(order < 0) or len(np.unique(order)) != len(order) or int(order.max()) + 1 != len(order):
        raise ValueError(f"{view}: face_order is not a complete permutation")
    result = np.empty(len(order), dtype=np.int8)
    if target_positive_rule == "strict_less":
        binary = values < target_threshold
    elif target_positive_rule == "greater_equal":
        binary = values >= target_threshold
    else:
        raise ValueError(f"Unsupported target positive rule: {target_positive_rule!r}")
    result[order] = binary.astype(np.int8)
    return result






def load_case_target(
    data_dir: Path,
    mesh_id: str,
    target_threshold: float = 0.5,
    target_positive_rule: str = "strict_less",
    target_source: str = "label_ply_any_exact_black_vertex",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manual = data_dir / "manual_2D"
    info = np.load(manual / "info" / f"{mesh_id}.npz", allow_pickle=True)
    vertices, _, mesh_faces = read_ply_mesh(data_dir / "origin" / f"{mesh_id}.ply")
    triangle_parts, area_parts = [], []
    for view in VIEW_NAMES:
        triangles = np.asarray(info[f"tri_{view}"], dtype=np.int64)
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError(f"{mesh_id}/{view}: triangle indices outside PLY vertex range")
        triangle_parts.append(triangles)
        area_parts.append(triangle_areas(vertices, triangles))
    triangles = np.concatenate(triangle_parts)
    if len(triangles) != len(mesh_faces):
        raise ValueError(f"{mesh_id}: UV views contain {len(triangles)} faces, mesh has {len(mesh_faces)}")
    keys = canonical_triangles(triangles)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(f"{mesh_id}: UV views contain duplicate triangles")
    if not np.array_equal(keys, canonical_triangles(mesh_faces)):
        raise ValueError(f"{mesh_id}: UV views are not an exact partition of mesh faces")
    view_to_mesh = view_to_mesh_face_indices(triangles, mesh_faces)

    if target_source == "label_ply_any_exact_black_vertex":
        label_vertices, label_colours, label_faces = read_ply_mesh(
            data_dir / "label" / f"{mesh_id}.ply"
        )
        if len(label_vertices) != len(vertices) or not np.array_equal(label_faces, mesh_faces):
            raise ValueError(f"{mesh_id}: origin and label PLY topology differ")
        black_vertex = np.all(label_colours == 0.0, axis=1)
        target_mesh = np.any(black_vertex[mesh_faces], axis=1).astype(np.int8)
        target = target_mesh[view_to_mesh]
    elif target_source == "manual_label_mesh":
        labels = np.load(manual / "label_mesh" / f"{mesh_id}.npz", allow_pickle=True)
        target_parts = [
            sorted_view_labels(labels, view, target_threshold, target_positive_rule)
            for view in VIEW_NAMES
        ]
        if any(len(part) != len(view_triangles) for part, view_triangles in zip(target_parts, triangle_parts)):
            raise ValueError(f"{mesh_id}: UV triangle and manual-label counts differ")
        target = np.concatenate(target_parts)
    else:
        raise ValueError(f"Unsupported target source: {target_source!r}")
    return target, triangles, np.concatenate(area_parts), vertices, view_to_mesh


def load_probability_image(path: Path, view: str, resize: str) -> np.ndarray | None:
    """Read one probability image, importing OpenCV only when PNG input is used."""
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PNG evaluation needs opencv-python; install requirements.txt or use face_npy/3dpred input."
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    target_h, target_w = VIEW_DIMS[view]
    if image.shape != (target_h, target_w):
        interpolation = cv2.INTER_NEAREST if resize == "nearest" else cv2.INTER_LINEAR
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    return np.clip(image.astype(np.float64) / 255.0, 0.0, 1.0)


def sample_triangle_centres(image: np.ndarray, triangles: np.ndarray, uv_pixels: np.ndarray) -> np.ndarray:
    centres = np.rint(np.mean(uv_pixels[triangles], axis=1)).astype(np.int64)
    x = np.clip(centres[:, 0], 0, image.shape[1] - 1)
    y = np.clip(centres[:, 1], 0, image.shape[0] - 1)
    return image[y, x]


def load_scores(data_dir: Path, spec: dict, mesh_id: str, view_face_indices: np.ndarray | None = None) -> np.ndarray | None:
    """Load one method's probabilities in concatenated UV-view face order.

    ``face_npy`` inputs are stored in original PLY face order and are mapped
    back to the UV-view order. The other supported input types already use the
    three-view representation. Keeping this conversion explicit prevents an
    accidental comparison of different face orderings.
    """
    kind = spec["type"]
    manual = data_dir / "manual_2D"
    if kind == "3dpred":
        parts = []
        for index in range(3):
            path = Path(spec["path"]) / f"{mesh_id}_{index}.npz"
            if not path.exists():
                return None
            parts.append(np.asarray(np.load(path, allow_pickle=True)["concatenated_array"], dtype=np.float64).reshape(-1))
        return np.concatenate(parts)
    if kind == "face_npy":
        if view_face_indices is None:
            raise ValueError(f"{mesh_id}: face_npy needs UV-to-mesh face indices")
        path = Path(spec["path"]) / f"{mesh_id}.npy"
        if not path.exists():
            return None
        scores = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64).reshape(-1)
        if len(scores) != len(view_face_indices):
            raise ValueError(f"{mesh_id}: face_npy has {len(scores)} scores, mesh has {len(view_face_indices)} faces")
        return scores[view_face_indices]
    if kind == "png":
        info = np.load(manual / "info" / f"{mesh_id}.npz", allow_pickle=True)
        parts = []
        for index, view in enumerate(VIEW_NAMES):
            image = load_probability_image(Path(spec["path"]) / f"{mesh_id}_{index}.png", view, spec.get("resize", "nearest"))
            if image is None:
                return None
            parts.append(sample_triangle_centres(image, np.asarray(info[f"tri_{view}"], dtype=np.int64), info[f"uvpx_{view}"]))
        return np.concatenate(parts)
    if kind == "fusion":
        three_d = load_scores(data_dir, {"type": "3dpred", "path": spec["path"]}, mesh_id, view_face_indices)
        two_d = load_scores(data_dir, {"type": "png", "path": spec["png_path"], "resize": spec.get("resize", "nearest")}, mesh_id, view_face_indices)
        if three_d is None or two_d is None or len(three_d) != len(two_d):
            return None
        # Keep both branches as probabilities until the final decision by
        # default.  A hard 2D branch is available only for an explicitly
        # declared sensitivity analysis; it is not the historical fixed-fusion
        # behavior in ``evaluation3d_confidence_fusion.py``.
        if bool(spec.get("fusion_binarize_2d", False)):
            two_d = (two_d >= float(spec.get("fusion_2d_threshold", 0.5))).astype(np.float64)
        weight_2d = float(spec.get("fusion_weight_2d", 0.5))
        return weight_2d * two_d + (1.0 - weight_2d) * three_d
    raise ValueError(f"{mesh_id}: unsupported prediction type {kind!r}")


def weighted_auc(target: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    finite = np.isfinite(score) & np.isfinite(weight)
    target, score, weight = target[finite], score[finite], weight[finite]
    positive_weight, negative_weight = weight[target == 1].sum(), weight[target == 0].sum()
    if positive_weight == 0 or negative_weight == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    score, target, weight = score[order], target[order], weight[order]
    negative_so_far, wins = 0.0, 0.0
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[end] == score[start]:
            end += 1
        group_target, group_weight = target[start:end], weight[start:end]
        group_positive = group_weight[group_target == 1].sum()
        group_negative = group_weight[group_target == 0].sum()
        wins += group_positive * (negative_so_far + 0.5 * group_negative)
        negative_so_far += group_negative
        start = end
    return float(wins / (positive_weight * negative_weight))


def positive_prediction(score: np.ndarray, threshold: float, rule: str) -> np.ndarray:
    """Apply an explicitly recorded probability decision relation."""
    if rule == "greater_equal":
        return score >= threshold
    if rule == "strict_greater":
        return score > threshold
    raise ValueError(f"Unsupported positive rule: {rule!r}")


def metric_row(
    target: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    threshold: float,
    positive_rule: str,
) -> dict[str, float]:
    """Compute binary segmentation and coverage metrics for one measurement unit."""
    pred = positive_prediction(score, threshold, positive_rule)
    target = target.astype(bool)
    tp = float(weight[pred & target].sum())
    fp = float(weight[pred & ~target].sum())
    fn = float(weight[~pred & target].sum())
    tn = float(weight[~pred & ~target].sum())
    def divide_or_nan(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else math.nan

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "plaque_iou": divide_or_nan(tp, tp + fp + fn),
        "plaque_dsc": divide_or_nan(2 * tp, 2 * tp + fp + fn),
        "oa": divide_or_nan(tp + tn, tp + fp + fn + tn),
        "sensitivity": divide_or_nan(tp, tp + fn),
        "specificity": divide_or_nan(tn, tn + fp),
        "ppv": divide_or_nan(tp, tp + fp),
        "npv": divide_or_nan(tn, tn + fn),
        "auc": weighted_auc(target.astype(np.int8), score, weight),
        "true_coverage": divide_or_nan(weight[target].sum(), weight.sum()),
        "pred_coverage": divide_or_nan(weight[pred].sum(), weight.sum()),
    }


def face_to_vertex(target: np.ndarray, score: np.ndarray, triangles: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    score_sum = np.zeros(vertex_count, dtype=np.float64)
    count = np.zeros(vertex_count, dtype=np.int64)
    label = np.zeros(vertex_count, dtype=np.int8)
    for column in range(3):
        index = triangles[:, column]
        np.add.at(score_sum, index, score)
        np.add.at(count, index, 1)
        np.maximum.at(label, index, target)
    used = count > 0
    return label[used], score_sum[used] / count[used]


def patient_id(mesh_id: str) -> str:
    """Derive the participant key when each participant has arches ``01`` and ``02``."""
    if len(mesh_id) < 3 or mesh_id[-2:] not in {"01", "02"}:
        raise ValueError(f"{mesh_id}: cannot infer paired patient id")
    return mesh_id[:-2]


def forbid_output_inside_sources(output: Path, sources: list[Path]) -> None:
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if output == source or source in output.parents:
            raise ValueError(f"Refusing to write outputs inside source data: {output}")


def require_fresh_output_dir(output: Path) -> None:
    """Avoid silently mixing a new evaluation with files from an earlier run."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse a non-empty output directory: {output}. "
            "Choose a new directory or archive the earlier evaluation first."
        )


def bootstrap_rows(rows: list[dict[str, object]], method: str, unit: str, reps: int, seed: int) -> list[dict[str, object]]:
    """Return percentile CIs for patient-macro metrics by resampling patients."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(reps, len(rows)))
    result: list[dict[str, object]] = []
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            continue
        samples = values[indices]
        boot = np.nanmean(samples, axis=1)
        boot = boot[np.isfinite(boot)]
        result.append({
            "method": method, "unit": unit, "metric": metric,
            "estimate": float(np.nanmean(values)),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "n_patients": len(rows), "n_valid": int(valid.sum()),
            "bootstrap_reps": reps, "seed": seed,
        })
    return result


def sign_flip_pvalue(difference: np.ndarray, reps: int, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(difference)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(reps, len(difference)))
    null = np.abs(np.mean(signs * difference, axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (reps + 1))


def paired_bootstrap_difference(difference: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(difference), size=(reps, len(difference)))
    estimates = difference[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def holm_adjust(rows: list[dict[str, object]]) -> None:
    order = np.argsort([float(row["p_value"]) for row in rows])
    previous = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index]["p_value"]))
        adjusted = max(previous, adjusted)
        rows[index]["p_value_holm"] = adjusted
        previous = adjusted


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    empty_fieldnames: tuple[str, ...] | None = None,
) -> None:
    """Write a table, including a header-only file for an empty valid table.

    A one-method evaluation has no paired comparisons.  It is still a complete
    evaluation, so it must emit an empty, schema-defined comparison table that
    can be hashed in the provenance manifest rather than failing after all mesh
    metrics were already calculated.
    """
    if not rows and empty_fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else list(empty_fieldnames or ())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Split directory containing origin/ and manual_2D/.")
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--spec-json", type=Path, required=True, help="JSON list of method specifications used by existing scripts.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.5,
        help="Binary threshold for continuous label_mesh channel-9 targets.",
    )
    parser.add_argument(
        "--target-positive-rule",
        choices=sorted(TARGET_POSITIVE_RULES),
        default="strict_less",
        help=(
            "Direction defining plaque in label_mesh channel 9. The study's "
            "black-plaque convention is strict_less (value < target threshold)."
        ),
    )
    parser.add_argument(
        "--target-source",
        choices=sorted(TARGET_SOURCES),
        default="label_ply_any_exact_black_vertex",
        help=(
            "Ground-truth source. The study protocol uses the raw label PLY and "
            "marks a face positive when any vertex is exactly black."
        ),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--permutation-reps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--primary-method", help="Method name for paired tests against every other method.")
    args = parser.parse_args()

    if not 0.0 <= args.target_threshold <= 1.0:
        raise ValueError("--target-threshold must be in [0, 1]")

    forbid_output_inside_sources(args.out_dir, [args.data_dir])
    require_fresh_output_dir(args.out_dir)

    mesh_ids = read_mesh_ids(args.mesh_list)
    if not mesh_ids:
        raise ValueError("No mesh ids")
    patients: dict[str, list[str]] = defaultdict(list)
    for mesh_id in mesh_ids:
        patients[patient_id(mesh_id)].append(mesh_id)
    incomplete = {pid: ids for pid, ids in patients.items() if sorted(mesh[-2:] for mesh in ids) != ["01", "02"]}
    if incomplete:
        raise ValueError(f"Incomplete patient pairs: {incomplete}")
    specs = validate_specs(json.loads(args.spec_json.read_text()), args.threshold)

    targets = {
        mesh_id: load_case_target(
            args.data_dir,
            mesh_id,
            args.target_threshold,
            args.target_positive_rule,
            args.target_source,
        )
        for mesh_id in mesh_ids
    }
    all_patient_rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    ci_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []

    for spec_index, spec in enumerate(specs):
        method = str(spec["name"])
        threshold = float(spec.get("threshold", args.threshold))
        positive_rule = str(spec["positive_rule"])
        score_transform = str(spec["score_transform"])
        case_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for mesh_id in mesh_ids:
            target, triangles, area, vertices, view_face_indices = targets[mesh_id]
            score = load_scores(args.data_dir, spec, mesh_id, view_face_indices)
            if score is None:
                raise FileNotFoundError(f"{method}: missing prediction for {mesh_id}")
            score = validate_probability_scores(score, mesh_id, method)
            if score_transform == "one_minus":
                score = 1.0 - score
            if len(score) != len(target):
                raise ValueError(f"{method}/{mesh_id}: score length {len(score)} != label length {len(target)}")
            vertex_target, vertex_score = face_to_vertex(target, score, triangles, len(vertices))
            case_data[mesh_id] = {
                "face": (target, score, np.ones(len(target), dtype=np.float64)),
                "area": (target, score, area),
                "vertex": (vertex_target, vertex_score, np.ones(len(vertex_target), dtype=np.float64)),
            }
        for unit in ("face", "area", "vertex"):
            patient_rows: list[dict[str, object]] = []
            for pid, ids in sorted(patients.items()):
                target = np.concatenate([case_data[mesh_id][unit][0] for mesh_id in ids])
                score = np.concatenate([case_data[mesh_id][unit][1] for mesh_id in ids])
                weight = np.concatenate([case_data[mesh_id][unit][2] for mesh_id in ids])
                row: dict[str, object] = metric_row(target, score, weight, threshold, positive_rule)
                row.update({
                    "method": method, "threshold": threshold, "positive_rule": positive_rule,
                    "score_transform": score_transform,
                    "unit": unit, "patient_id": pid, "n_cases": len(ids),
                })
                patient_rows.append(row)
            all_patient_rows[(method, unit)] = patient_rows
            ci_rows.extend(bootstrap_rows(patient_rows, method, unit, args.bootstrap_reps, args.seed + spec_index))
            for mesh_id, data in case_data.items():
                row: dict[str, object] = metric_row(*data[unit], threshold, positive_rule)
                row.update({
                    "method": method, "threshold": threshold, "positive_rule": positive_rule,
                    "score_transform": score_transform,
                    "unit": unit, "mesh_id": mesh_id, "patient_id": patient_id(mesh_id),
                })
                case_rows.append(row)

    comparison_rows: list[dict[str, object]] = []
    if args.primary_method:
        rng = np.random.default_rng(args.seed)
        for unit in ("face", "area", "vertex"):
            primary = all_patient_rows.get((args.primary_method, unit))
            if primary is None:
                raise ValueError(f"Unknown primary method {args.primary_method!r}")
            primary_by_id = {str(row["patient_id"]): row for row in primary}
            for spec in specs:
                other = str(spec["name"])
                if other == args.primary_method:
                    continue
                other_by_id = {str(row["patient_id"]): row for row in all_patient_rows[(other, unit)]}
                for metric in ("plaque_iou", "plaque_dsc"):
                    diff = np.asarray(
                        [float(primary_by_id[pid][metric]) - float(other_by_id[pid][metric]) for pid in sorted(primary_by_id)],
                        dtype=np.float64,
                    )
                    if not np.all(np.isfinite(diff)):
                        raise ValueError(
                            f"{unit}/{metric}: paired comparison has an undefined participant value; "
                            "define the metric convention before reporting this comparison"
                        )
                    ci_low, ci_high = paired_bootstrap_difference(diff, args.bootstrap_reps, rng)
                    comparison_rows.append({
                        "primary_method": args.primary_method, "comparator": other, "unit": unit, "metric": metric,
                        "mean_difference": float(diff.mean()), "difference_ci_low": ci_low, "difference_ci_high": ci_high,
                        "p_value": sign_flip_pvalue(diff, args.permutation_reps, rng),
                        "n_patients": len(diff), "permutation_reps": args.permutation_reps, "seed": args.seed,
                    })
        for unit in ("face", "area", "vertex"):
            for metric in ("plaque_iou", "plaque_dsc"):
                subset = [row for row in comparison_rows if row["unit"] == unit and row["metric"] == metric]
                holm_adjust(subset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_case_metrics.csv", case_rows)
    patient_rows = [row for rows in all_patient_rows.values() for row in rows]
    write_csv(args.out_dir / "per_patient_metrics.csv", patient_rows)
    write_csv(args.out_dir / "patient_bootstrap_95ci.csv", ci_rows)
    write_csv(
        args.out_dir / "paired_sign_flip_tests.csv",
        comparison_rows,
        empty_fieldnames=COMPARISON_FIELDS,
    )
    output_files = {
        "per_case_metrics": args.out_dir / "per_case_metrics.csv",
        "per_patient_metrics": args.out_dir / "per_patient_metrics.csv",
        "patient_bootstrap_95ci": args.out_dir / "patient_bootstrap_95ci.csv",
        "paired_sign_flip_tests": args.out_dir / "paired_sign_flip_tests.csv",
    }
    method_protocol = []
    for spec in specs:
        protocol = {
            "name": str(spec["name"]),
            "type": str(spec["type"]),
            "threshold": float(spec.get("threshold", args.threshold)),
            "positive_rule": str(spec["positive_rule"]),
            "score_transform": str(spec["score_transform"]),
        }
        if spec["type"] == "fusion":
            protocol["fusion_weight_2d"] = float(spec["fusion_weight_2d"])
            protocol["fusion_weight_3d"] = 1.0 - float(spec["fusion_weight_2d"])
            protocol["fusion_binarize_2d"] = bool(spec["fusion_binarize_2d"])
            protocol["fusion_2d_threshold"] = float(spec["fusion_2d_threshold"])
            protocol["png_resize"] = str(spec.get("resize", "nearest"))
        method_protocol.append(protocol)
    manifest = {
        "schema_version": 4,
        "mesh_count": len(mesh_ids),
        "patient_count": len(patients), "threshold": args.threshold,
        "target_threshold": args.target_threshold,
        "target_positive_rule": args.target_positive_rule,
        "target_source": args.target_source,
        "bootstrap_reps": args.bootstrap_reps, "permutation_reps": args.permutation_reps, "seed": args.seed,
        "statistical_unit": "paired patient (01 and 02 arches combined)",
        "face_definition": "each triangle counted equally; every original mesh face appears exactly once",
        "target_definition": (
            "raw label PLY face is plaque when any incident vertex RGB is exactly [0, 0, 0]"
            if args.target_source == "label_ply_any_exact_black_vertex"
            else (
                "manual label_mesh channel 9 < target_threshold"
                if args.target_positive_rule == "strict_less"
                else "manual label_mesh channel 9 >= target_threshold"
            )
        ),
        "area_definition": "original PLY triangle area with the same binary face target",
        "vertex_definition": "incident-face mean score and incident-face-any target",
        "primary_method": args.primary_method,
        "methods": method_protocol,
        "mesh_list_sha256": sha256_file(args.mesh_list),
        "mesh_ids_sha256": sha256_lines(mesh_ids),
        "patient_ids_sha256": sha256_lines(sorted(patients)),
        "spec_json_sha256": sha256_file(args.spec_json),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "target_input": hash_named_files(
            target_input_files(args.data_dir, mesh_ids, args.target_source)
        ),
        "prediction_inputs": {
            str(spec["name"]): hash_named_files(prediction_input_files(spec, mesh_ids))
            for spec in specs
        },
        "output_files": {label: sha256_file(path) for label, path in output_files.items()},
    }
    (args.out_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
