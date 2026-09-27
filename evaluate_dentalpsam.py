#!/usr/bin/env python3
"""Evaluate MICCAI predictions with the original equal-triangle protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dentalpsam.evaluation import (
    cluster_bootstrap, equal_face_metrics, load_mesh_predictions, read_mesh_ids,
)
from tools.evaluate_mesh import hash_named_files, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion-weight-2d", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if not 0 <= args.fusion_weight_2d <= 1 or not 0 <= args.threshold <= 1:
        raise ValueError("Fusion weight and threshold must lie in [0, 1]")
    if args.bootstrap_reps < 1:
        raise ValueError("--bootstrap-reps must be positive")
    output = args.output_dir.resolve()
    for source in (args.data_dir.resolve(), args.prediction_dir.resolve()):
        if output == source or source in output.parents:
            raise ValueError("Output must be outside input data and prediction directories")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    ids = read_mesh_ids(args.mesh_list)
    if args.expected_count is not None and len(ids) != args.expected_count:
        raise ValueError("Mesh-list count differs from --expected-count")
    rows = {branch: [] for branch in ("2d", "3d", "fusion")}
    files = [("mesh_list", args.mesh_list)]
    for mesh_id in ids:
        target, two_d, three_d = load_mesh_predictions(args.data_dir, args.prediction_dir, mesh_id)
        fused = args.fusion_weight_2d * two_d + (1 - args.fusion_weight_2d) * three_d
        for branch, scores in (("2d", two_d), ("3d", three_d), ("fusion", fused)):
            rows[branch].append({"mesh_id": mesh_id, **equal_face_metrics(target, scores, args.threshold)})
        files.extend([
            (f"label/{mesh_id}.ply", args.data_dir / "label" / f"{mesh_id}.ply"),
            (f"info/{mesh_id}.npz", args.data_dir / "manual_2D/info" / f"{mesh_id}.npz"),
        ])
        for index in range(3):
            files.append((f"2d/{mesh_id}_{index}.png", args.prediction_dir / f"{mesh_id}_{index}.png"))
            files.append((f"3d/{mesh_id}_{index}.npz", args.prediction_dir / "3Dpred" / f"{mesh_id}_{index}.npz"))
    summary = {branch: cluster_bootstrap(values, args.bootstrap_reps, args.seed)
               for branch, values in rows.items()}
    manifest = {
        "protocol": "MICCAI original equal-triangle, mesh-macro",
        "mesh_count": len(ids), "participant_count": len({name[:4] for name in ids}),
        "target": "per-channel minimum label-PLY vertex RGB equals zero",
        "decision": "strict_greater", "threshold": args.threshold,
        "uv_sampling": "mean vertex UV, truncate to int32, clip, no image resizing",
        "fusion_weight_2d": args.fusion_weight_2d, "fusion_weight_3d": 1 - args.fusion_weight_2d,
        "aggregation": "arithmetic mean of per-mesh metrics",
        "ci": "95% percentile bootstrap of participants carrying all their meshes",
        "bootstrap_reps": args.bootstrap_reps, "seed": args.seed,
        "input_files": hash_named_files(files),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "metric_module_sha256": sha256_file(Path(__file__).parent / "dentalpsam/evaluation.py"),
    }
    output.mkdir(parents=True)
    with (output / "per_mesh_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["branch", *rows["fusion"][0]])
        writer.writeheader()
        for branch, values in rows.items():
            writer.writerows({"branch": branch, **row} for row in values)
    for name, payload in (("summary.json", summary), ("evaluation_manifest.json", manifest)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({branch: {key: result[key] for key in ("plaque_iou", "plaque_dice", "accuracy")}
                      for branch, result in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
