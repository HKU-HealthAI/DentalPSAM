#!/usr/bin/env python3
"""Check prepared images and mesh patches before training or prediction.

The tool is intentionally dependency-free and read-only with respect to the
input split.  It verifies the four directories required by the public data
contract, checks the three-view naming convention, and produces a compact
summary. Patient images and labels are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from dentalpsam._layout import prepared_directories


REQUIRED_DIRS = ("images", "image_labels", "mesh_features", "mesh_labels")
VIEW_PATTERN = re.compile(r"^(?P<mesh>.+)_(?P<view>[012])\.png$")


def _fail(message: str) -> None:
    raise ValueError(message)


def _mesh_identifier(filename: str) -> tuple[str, int]:
    """Extract a mesh ID and its view index from one label PNG filename."""
    match = VIEW_PATTERN.fullmatch(filename)
    if not match:
        _fail(
            "Label image must be named '<mesh_id>_<view>.png' with view 0, 1, "
            f"or 2; got {filename!r}."
        )
    return match.group("mesh"), int(match.group("view"))


def _read_labels(label_dir: Path) -> dict[str, dict[int, str]]:
    """Return mesh -> view -> label filename, rejecting duplicate views."""
    meshes: dict[str, dict[int, str]] = defaultdict(dict)
    label_files = sorted(path for path in label_dir.iterdir() if path.is_file())
    if not label_files:
        _fail(f"No label images found in {label_dir}.")

    for path in label_files:
        mesh, view = _mesh_identifier(path.name)
        if view in meshes[mesh]:
            _fail(f"Duplicate view {view} for mesh {mesh!r} in {label_dir}.")
        meshes[mesh][view] = path.name
    return dict(meshes)


def inspect_split(data_dir: Path, participant_id_prefix_length: int | None = None) -> dict[str, Any]:
    """Validate one split and return a machine-readable manifest dictionary.

    ``participant_id_prefix_length`` is optional because participant IDs are a
    study-specific convention. When supplied, it is used only for a count;
    raw participant identifiers are not included in the summary.
    """
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        _fail(f"Split directory does not exist: {data_dir}")
    if participant_id_prefix_length is not None and participant_id_prefix_length <= 0:
        _fail("participant_id_prefix_length must be positive when supplied.")

    split_dirs = {name: path for name, path in prepared_directories(data_dir).items()
                  if name in REQUIRED_DIRS}
    missing = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing:
        _fail(f"Missing required subdirectories in {data_dir}: {', '.join(missing)}")

    labels = _read_labels(split_dirs["image_labels"])
    mesh_ids = sorted(labels)
    expected_views = {0, 1, 2}
    missing_views = {
        mesh: sorted(expected_views - set(views))
        for mesh, views in labels.items()
        if set(views) != expected_views
    }
    if missing_views:
        _fail(f"Every mesh must have views 0, 1, and 2; missing views: {missing_views}")

    for mesh, views in labels.items():
        for label_name in views.values():
            if not (split_dirs["images"] / label_name).is_file():
                _fail(f"Missing matching image for label {label_name!r}.")
        for folder in ("mesh_features", "mesh_labels"):
            expected = split_dirs[folder] / f"{mesh}.npz"
            if not expected.is_file():
                _fail(f"Missing {folder} file for mesh {mesh!r}: {expected}")

    mesh_file_names = {
        folder: sorted(path.name for path in split_dirs[folder].glob("*.npz"))
        for folder in ("mesh_features", "mesh_labels")
    }
    expected_mesh_files = [f"{mesh}.npz" for mesh in mesh_ids]
    for folder, files in mesh_file_names.items():
        if files != expected_mesh_files:
            _fail(
                f"{folder} must contain exactly one .npz file per label mesh. "
                f"Expected {len(expected_mesh_files)}, found {len(files)}."
            )

    origin_files = sorted(path.name for path in split_dirs["images"].glob("*.png"))
    label_files = sorted(filename for views in labels.values() for filename in views.values())
    if origin_files != label_files:
        _fail("images must contain exactly the same PNG filenames as image_labels.")

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "data_dir": str(data_dir),
        "required_directories": list(REQUIRED_DIRS),
        "mesh_count": len(mesh_ids),
        "view_count": len(label_files),
        "file_counts": {
            "images_png": len(origin_files),
            "image_labels_png": len(label_files),
            "mesh_features_npz": len(mesh_file_names["mesh_features"]),
            "mesh_labels_npz": len(mesh_file_names["mesh_labels"]),
        },
    }
    if participant_id_prefix_length is not None:
        participants = sorted({mesh[:participant_id_prefix_length] for mesh in mesh_ids})
        if any(len(identifier) != participant_id_prefix_length for identifier in participants):
            _fail("At least one mesh ID is shorter than participant_id_prefix_length.")
        manifest["participant_id_prefix_length"] = participant_id_prefix_length
        manifest["participant_count"] = len(participants)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only checks of prepared images and mesh patches."
    )
    parser.add_argument("--data-dir", required=True, type=Path, help="Split root or processed/ directory.")
    parser.add_argument(
        "--participant-id-prefix-length",
        type=int,
        default=None,
        help="Optional fixed prefix length used to count participant IDs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output JSON path outside the source split.",
    )
    args = parser.parse_args()

    manifest = inspect_split(args.data_dir, args.participant_id_prefix_length)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(payload, end="")
        return

    output = args.out.resolve()
    split_root = args.data_dir.resolve()
    if output.is_relative_to(split_root):
        _fail("Refusing to write a manifest inside --data-dir source data.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"Wrote split manifest: {output}")


if __name__ == "__main__":
    main()
