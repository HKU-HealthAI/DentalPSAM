#!/usr/bin/env python3
"""End-to-end regression test for the face/area/vertex reference evaluator.

The fixture contains two paired participants, two unequal-area faces per arch,
and UV triangles deliberately stored in a different order from the PLY faces.
It proves that ``face_npy`` predictions are reordered correctly, arches are
combined at the participant level, and an existing output directory is never
silently reused.  No study data, checkpoints, or image dependencies are used.
"""

from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tests" / "_reference_evaluation.py"
SUBPROCESS_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_binary_ply(
    path: Path,
    label_colours: bool = False,
    positive_face: bool = True,
) -> None:
    """Write two disjoint triangles with areas 0.5 and 2.0 in PLY face order."""
    vertices = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (10.0, 0.0, 0.0),
        (12.0, 0.0, 0.0),
        (10.0, 2.0, 0.0),
    )
    faces = ((0, 1, 2), (3, 4, 5))
    header = "\n".join((
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        "",
    )).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        for index, (x, y, z) in enumerate(vertices):
            if label_colours:
                colour = 0 if positive_face and index < 3 else 255
            else:
                colour = 128
            handle.write(struct.pack("<fffBBB", x, y, z, colour, colour, colour))
        for triangle in faces:
            handle.write(struct.pack("<Biii", 3, *triangle))


def object_parts(value: np.ndarray) -> np.ndarray:
    result = np.empty(1, dtype=object)
    result[0] = value
    return result


def write_case(data_dir: Path, mesh_id: str, positive_face: bool = True) -> None:
    origin = data_dir / "origin"
    label = data_dir / "label"
    manual = data_dir / "manual_2D"
    (manual / "info").mkdir(parents=True, exist_ok=True)
    (manual / "label_mesh").mkdir(parents=True, exist_ok=True)
    origin.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)
    write_binary_ply(origin / f"{mesh_id}.ply")
    write_binary_ply(
        label / f"{mesh_id}.ply",
        label_colours=True,
        positive_face=positive_face,
    )

    # UV order is face 1 then face 0; PLY order is face 0 then face 1.
    up_triangles = np.asarray(((3, 4, 5), (0, 1, 2)), dtype=np.int64)
    empty_triangles = np.empty((0, 3), dtype=np.int64)
    np.savez(
        manual / "info" / f"{mesh_id}.npz",
        tri_up=up_triangles,
        tri_in=empty_triangles,
        tri_out=empty_triangles,
    )

    # The second UV triangle (PLY face 0) is black/plaque (< 0.5) when
    # requested; the first is white/non-plaque (>= 0.5).
    labels = np.zeros((2, 10), dtype=np.float32)
    labels[:, 9] = (0.75, 0.25 if positive_face else 0.75)
    empty_parts = np.empty(0, dtype=object)
    empty_order = np.empty(0, dtype=object)
    np.savez(
        manual / "label_mesh" / f"{mesh_id}.npz",
        **{
            "up": object_parts(labels),
            "in": empty_parts,
            "out": empty_parts,
            "face_order_up": object_parts(np.asarray((0, 1), dtype=np.int64)),
            "face_order_in": empty_order,
            "face_order_out": empty_order,
        },
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_evaluator(data_dir: Path, mesh_list: Path, spec_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(EVALUATOR),
            "--data-dir", str(data_dir),
            "--mesh-list", str(mesh_list),
            "--spec-json", str(spec_path),
            "--out-dir", str(output_dir),
            "--primary-method", "perfect",
            "--bootstrap-reps", "200",
            "--permutation-reps", "1000",
            "--seed", "7",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=SUBPROCESS_ENV,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dentalpsam_mesh_eval_") as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        mesh_ids = ("00101", "00102", "00201", "00202")
        for mesh_id in mesh_ids:
            write_case(data_dir, mesh_id)

        mesh_list = root / "mesh_ids.txt"
        mesh_list.write_text("\n".join(mesh_ids) + "\n")
        perfect_dir, inverse_dir = root / "perfect", root / "inverse"
        perfect_dir.mkdir()
        inverse_dir.mkdir()
        for mesh_id in mesh_ids:
            # Arrays are saved in PLY face order, not the UV-view order.
            np.save(perfect_dir / f"{mesh_id}.npy", np.asarray((0.9, 0.1)))
            np.save(inverse_dir / f"{mesh_id}.npy", np.asarray((0.1, 0.9)))
        specs = [
            {"name": "perfect", "type": "face_npy", "path": str(perfect_dir), "threshold": 0.5},
            {"name": "inverse", "type": "face_npy", "path": str(inverse_dir), "threshold": 0.5},
        ]
        spec_path = root / "methods.json"
        spec_path.write_text(json.dumps(specs))
        output_dir = root / "evaluation"

        completed = run_evaluator(data_dir, mesh_list, spec_path, output_dir)
        require(completed.returncode == 0, completed.stderr)

        patient_rows = read_csv(output_dir / "per_patient_metrics.csv")
        require(len(patient_rows) == 12, "expected 2 methods x 3 units x 2 participants")
        for row in patient_rows:
            expected_iou = 1.0 if row["method"] == "perfect" else 0.0
            require(abs(float(row["plaque_iou"]) - expected_iou) < 1e-12, "face order was not restored")

        comparison_rows = read_csv(output_dir / "paired_sign_flip_tests.csv")
        require(len(comparison_rows) == 6, "expected face/area/vertex IoU and Dice comparisons")
        for row in comparison_rows:
            require(abs(float(row["mean_difference"]) - 1.0) < 1e-12, "unexpected paired difference")
            require(abs(float(row["difference_ci_low"]) - 1.0) < 1e-12, "unexpected paired CI lower bound")
            require(abs(float(row["difference_ci_high"]) - 1.0) < 1e-12, "unexpected paired CI upper bound")
            require("p_value_holm" in row, "missing Holm-adjusted p-value")

        manifest = json.loads((output_dir / "evaluation_manifest.json").read_text())
        require(manifest["mesh_count"] == 4 and manifest["patient_count"] == 2, "wrong participant aggregation")
        require(manifest["target_threshold"] == 0.5, "continuous target threshold was not recorded")
        require(
            manifest["target_source"] == "label_ply_any_exact_black_vertex",
            "raw label-PLY target source was not recorded",
        )
        require(manifest["schema_version"] == 4, "missing provenance schema version")
        require(manifest["target_input"]["file_count"] == 12, "target provenance missed an input file")
        require(manifest["prediction_inputs"]["perfect"]["file_count"] == 4, "prediction provenance missed an input file")
        require(len(manifest["evaluator_sha256"]) == 64, "missing evaluator source hash")
        require(len(manifest["output_files"]["per_patient_metrics"]) == 64, "missing output hash")

        # A valid one-method run has no paired comparisons.  It must still
        # produce a header-only comparison CSV so the output manifest can bind
        # every declared result artifact without failing after computation.
        one_method_spec_path = root / "one_method.json"
        one_method_spec_path.write_text(json.dumps(specs[:1]))
        one_method_output = root / "one_method_evaluation"
        one_method = subprocess.run(
            [
                sys.executable, str(EVALUATOR),
                "--data-dir", str(data_dir),
                "--mesh-list", str(mesh_list),
                "--spec-json", str(one_method_spec_path),
                "--out-dir", str(one_method_output),
                "--bootstrap-reps", "200", "--permutation-reps", "1000", "--seed", "7",
            ],
            text=True, capture_output=True, check=False, env=SUBPROCESS_ENV,
        )
        require(one_method.returncode == 0, one_method.stderr)
        require(len(read_csv(one_method_output / "paired_sign_flip_tests.csv")) == 0, "single method has comparisons")
        one_method_manifest = json.loads((one_method_output / "evaluation_manifest.json").read_text())
        require(len(one_method_manifest["output_files"]["paired_sign_flip_tests"]) == 64, "empty table was not hashed")

        # The historical evaluator used a strict final ``> 0.5`` relation.
        # Boundary values must not silently become positive under the canonical
        # default ``>=`` rule when that historical protocol is requested.
        boundary_specs = [
            {"name": "inclusive", "type": "face_npy", "path": str(perfect_dir), "threshold": 0.9},
            {
                "name": "strict", "type": "face_npy", "path": str(perfect_dir),
                "threshold": 0.9, "positive_rule": "strict_greater",
            },
        ]
        boundary_spec_path = root / "boundary_methods.json"
        boundary_spec_path.write_text(json.dumps(boundary_specs))
        boundary_output = root / "boundary_evaluation"
        boundary = subprocess.run(
            [
                sys.executable, str(EVALUATOR),
                "--data-dir", str(data_dir),
                "--mesh-list", str(mesh_list),
                "--spec-json", str(boundary_spec_path),
                "--out-dir", str(boundary_output),
                "--bootstrap-reps", "200", "--permutation-reps", "1000", "--seed", "7",
            ],
            text=True, capture_output=True, check=False, env=SUBPROCESS_ENV,
        )
        require(boundary.returncode == 0, boundary.stderr)
        boundary_rows = read_csv(boundary_output / "per_patient_metrics.csv")
        inclusive_face = [row for row in boundary_rows if row["method"] == "inclusive" and row["unit"] == "face"]
        strict_face = [row for row in boundary_rows if row["method"] == "strict" and row["unit"] == "face"]
        require(all(float(row["plaque_iou"]) == 1.0 for row in inclusive_face), "inclusive threshold is wrong")
        require(all(float(row["plaque_iou"]) == 0.0 for row in strict_face), "strict threshold is wrong")

        reused = run_evaluator(data_dir, mesh_list, spec_path, output_dir)
        require(reused.returncode != 0, "expected non-empty output directory rejection")
        require("Refusing to reuse a non-empty output directory" in reused.stderr, "missing output reuse guard")

        # A patient with no positive face and no predicted face has undefined
        # plaque IoU/Dice. A paired comparison must fail rather than silently
        # drop that patient and report an inflated effective sample size.
        undefined_data = root / "undefined_data"
        for mesh_id in mesh_ids:
            write_case(undefined_data, mesh_id, positive_face=not mesh_id.startswith("001"))
        undefined_perfect, undefined_inverse = root / "undefined_perfect", root / "undefined_inverse"
        undefined_perfect.mkdir()
        undefined_inverse.mkdir()
        for mesh_id in mesh_ids:
            if mesh_id.startswith("001"):
                np.save(undefined_perfect / f"{mesh_id}.npy", np.asarray((0.1, 0.1)))
                np.save(undefined_inverse / f"{mesh_id}.npy", np.asarray((0.1, 0.9)))
            else:
                np.save(undefined_perfect / f"{mesh_id}.npy", np.asarray((0.9, 0.1)))
                np.save(undefined_inverse / f"{mesh_id}.npy", np.asarray((0.1, 0.9)))
        undefined_specs = [
            {"name": "perfect", "type": "face_npy", "path": str(undefined_perfect), "threshold": 0.5},
            {"name": "inverse", "type": "face_npy", "path": str(undefined_inverse), "threshold": 0.5},
        ]
        undefined_spec_path = root / "undefined_methods.json"
        undefined_spec_path.write_text(json.dumps(undefined_specs))
        undefined_run = run_evaluator(
            undefined_data, mesh_list, undefined_spec_path, root / "undefined_evaluation"
        )
        require(undefined_run.returncode != 0, "undefined paired IoU/Dice was silently accepted")
        require("paired comparison has an undefined participant value" in undefined_run.stderr, "missing undefined-patient guard")

    print("mesh_evaluation_self_test_ok")


if __name__ == "__main__":
    main()
