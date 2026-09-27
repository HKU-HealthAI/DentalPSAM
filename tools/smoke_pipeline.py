#!/usr/bin/env python3
"""Exercise PLY -> UV -> TSGCNet NPZ -> DentalPSAM -> original evaluation.

This single-mesh integration test validates interfaces, not paper performance.
All generated files and logs go to a fresh directory outside the source data.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_mesh import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "sam-checkpoint",
        "dentalpsam-checkpoint",
        "tsgcnet-checkpoint",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mesh-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.mesh_id) != 6 or not args.mesh_id.isdigit():
        raise ValueError("--mesh-id must contain six digits")
    source = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the source data tree")
    if output.exists():
        raise FileExistsError(output)
    raw_inputs = [source / kind / f"{args.mesh_id}.ply" for kind in ("origin", "label")]
    checkpoints = [
        args.sam_checkpoint,
        args.dentalpsam_checkpoint,
        args.tsgcnet_checkpoint,
    ]
    for path in raw_inputs + checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    before = {str(path): sha256_file(path) for path in raw_inputs}
    output.mkdir(parents=True)
    mesh_list = output / "mesh_ids.txt"
    mesh_list.write_text(args.mesh_id + "\n")
    root = Path(__file__).resolve().parents[1]
    record = {
        "scope": "single-mesh integration only; not cohort performance",
        "status": "started",
        "python": platform.python_version(),
        "raw_input_sha256_before": before,
        "stages": [],
        "source_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*.py"))
            if ".git" not in path.parts
        },
        "checkpoint_sha256": {str(path): sha256_file(path) for path in checkpoints},
    }
    manifest = output / "smoke_manifest.json"

    def save_record():
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    def run(stage, entry, arguments):
        command = [sys.executable, str(root / entry), *map(str, arguments)]
        record["stages"].append(
            {"stage": stage, "command": command, "status": "running"}
        )
        save_record()
        print(f"Running {stage}", flush=True)
        with (output / f"{stage}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        record["stages"][-1].update(
            status="passed" if result.returncode == 0 else "failed",
            returncode=result.returncode,
        )
        save_record()
        if result.returncode:
            raise RuntimeError(f"{stage} failed; inspect {output / (stage + '.log')}")

    try:
        run(
            "uv",
            "scripts/data/prepare_views.py",
            [
                "--origin-dir",
                source / "origin",
                "--label-dir",
                source / "label",
                "--mesh-list",
                mesh_list,
                "--output-dir",
                output / "uv",
            ],
        )
        run(
            "tsgcnet",
            "scripts/branch3d/export_features.py",
            [
                "--checkpoint",
                args.tsgcnet_checkpoint,
                "--data-dir",
                source,
                "--info-dir",
                output / "uv/info",
                "--mesh-list",
                mesh_list,
                "--out-dir",
                output / "features",
                "--device",
                args.device,
            ],
        )
        # Links are created only in the new output tree. Raw targets remain read-only.
        derived = output / "evaluation_data"
        manual = derived / "manual_2D"
        manual.mkdir(parents=True)
        for kind in ("origin", "label"):
            (derived / kind).symlink_to(source / kind, target_is_directory=True)
        for kind in ("origin", "label", "info"):
            (manual / kind).symlink_to(output / "uv" / kind, target_is_directory=True)
        for kind in ("SOTA_mesh", "label_mesh"):
            (manual / kind).symlink_to(
                output / "features" / kind, target_is_directory=True
            )
        run(
            "preflight",
            "tools/preflight_split.py",
            ["--data-dir", manual, "--out", output / "preflight.json"],
        )
        run(
            "prediction",
            "scripts/dentalpsam/predict.py",
            [
                "--checkpoint",
                args.dentalpsam_checkpoint,
                "--sam-checkpoint",
                args.sam_checkpoint,
                "--data-dir",
                manual,
                "--mesh-list",
                mesh_list,
                "--expected-count",
                "1",
                "--output-dir",
                output / "predictions",
                "--device",
                args.device,
            ],
        )
        run(
            "evaluation",
            "scripts/dentalpsam/evaluate.py",
            [
                "--data-dir",
                derived,
                "--prediction-dir",
                output / "predictions",
                "--mesh-list",
                mesh_list,
                "--expected-count",
                "1",
                "--output-dir",
                output / "evaluation",
                "--bootstrap-reps",
                "100",
                "--fusion-weight-2d",
                "0.5",
                "--threshold",
                "0.5",
                "--seed",
                "42",
            ],
        )
        record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        after = {str(path): sha256_file(path) for path in raw_inputs}
        record["raw_input_sha256_after"] = after
        record["raw_inputs_unchanged"] = before == after
        if before != after:
            record["status"] = "failed"
        save_record()
    if not record["raw_inputs_unchanged"]:
        raise RuntimeError("Raw input hashes changed during the test")
    print("PASS: five pipeline stages; original PLY hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
