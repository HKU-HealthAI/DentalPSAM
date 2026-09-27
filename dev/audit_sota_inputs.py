#!/usr/bin/env python3
"""Audit the actual cached SOTA scores supplied to DentalPSAM, without altering them.

Validate geometry and face ordering, measure scores exactly as stored, and
optionally compare them with a frozen TSGCNet score export. Never invert scores
or select a model based on these test-set diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentalpsam.evaluation import cluster_bootstrap, equal_face_metrics, read_mesh_ids
from dev.evaluate_mesh import (
    hash_named_files,
    read_ply_mesh,
    view_to_mesh_face_indices,
)


def read_cached_scores(data_dir, mesh_id):
    """Restore cached scores to original PLY order after exact contract checks."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "mesh-list", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-scores",
        type=Path,
        help="Optional original-face scores/*.npy from evaluate_tsgcnet.py",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for source in (args.data_dir, args.checkpoint_scores):
        if source is not None and (
            output == source.resolve() or source.resolve() in output.parents
        ):
            raise ValueError("Output must be outside source data/score directories")
    if output.exists():
        raise FileExistsError(output)
    rows, differences, class_scores = [], [], {False: [], True: []}
    files = [("mesh_list", args.mesh_list)]
    for mesh_id in read_mesh_ids(args.mesh_list):
        scores, faces = read_cached_scores(args.data_dir, mesh_id)
        _xyz, colours, label_faces = read_ply_mesh(
            args.data_dir / "label" / f"{mesh_id}.ply"
        )
        np.testing.assert_array_equal(faces, label_faces)
        target = np.all(np.min(colours[faces], axis=1) == 0, axis=1)
        rows.append({"mesh_id": mesh_id, **equal_face_metrics(target, scores)})
        for label in (False, True):
            class_scores[label].extend(scores[target == label].tolist())
        for kind in ("origin", "label", "manual_2D/SOTA_mesh", "manual_2D/info"):
            suffix = "ply" if kind in ("origin", "label") else "npz"
            path = args.data_dir / kind / f"{mesh_id}.{suffix}"
            files.append((f"{kind}/{mesh_id}.{suffix}", path))
        if args.checkpoint_scores is not None:
            path = args.checkpoint_scores / f"{mesh_id}.npy"
            candidate = np.load(path)
            if candidate.shape != scores.shape or not np.isfinite(candidate).all():
                raise ValueError(f"Checkpoint/cache shape mismatch: {mesh_id}")
            differences.append(
                {
                    "mean_abs": float(np.abs(candidate - scores).mean()),
                    "max_abs": float(np.abs(candidate - scores).max()),
                }
            )
            files.append((f"checkpoint_scores/{mesh_id}.npy", path))
    result = {
        "scope": "cached scores as stored; generating checkpoint/convention not assumed",
        "mesh_count": len(rows),
        "patient_count": len({r["mesh_id"][:4] for r in rows}),
        "geometry_and_face_order": "passed",
        "score_transform": "identity",
        "equal_triangle_metrics": cluster_bootstrap(rows, 10000, 42),
        "face_pooled_mean_score_on_plaque": float(np.mean(class_scores[True])),
        "face_pooled_mean_score_on_nonplaque": float(np.mean(class_scores[False])),
        "checkpoint_comparison": (
            None
            if not differences
            else {
                "mean_mesh_mean_absolute_difference": float(
                    np.mean([x["mean_abs"] for x in differences])
                ),
                "maximum_absolute_difference": max(x["max_abs"] for x in differences),
                "exactly_equal": all(x["max_abs"] == 0 for x in differences),
            }
        ),
        "input_files": hash_named_files(files),
    }
    output.mkdir(parents=True)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "input_files"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
