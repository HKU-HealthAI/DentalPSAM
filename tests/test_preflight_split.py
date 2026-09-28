#!/usr/bin/env python3
"""Dependency-free functional test for the prepared-input preflight library."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dentalpsam._preflight import inspect_split  # noqa: E402


def populate_valid_split(root: Path) -> None:
    """Create a minimal two-mesh, three-view fixture without image decoding."""
    for name in ("origin", "label", "SOTA_mesh", "label_mesh"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for mesh in ("000101", "000102"):
        for view in range(3):
            filename = f"{mesh}_{view}.png"
            (root / "origin" / filename).write_bytes(b"origin")
            (root / "label" / filename).write_bytes(b"label")
        (root / "SOTA_mesh" / f"{mesh}.npz").write_bytes(b"sota")
        (root / "label_mesh" / f"{mesh}.npz").write_bytes(b"label_mesh")


def expect_value_error(callable_object, contains: str) -> None:
    try:
        callable_object()
    except ValueError as error:
        if contains not in str(error):
            raise AssertionError(f"Expected {contains!r}, received {error!s}") from error
    else:
        raise AssertionError("Expected ValueError")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        split = Path(temporary_directory) / "manual_2D"
        populate_valid_split(split)
        manifest = inspect_split(split, participant_id_prefix_length=4)
        assert manifest["mesh_count"] == 2
        assert manifest["view_count"] == 6
        assert manifest["participant_count"] == 1
        assert manifest["file_counts"]["mesh_features_npz"] == 2

        (split / "origin" / "000102_2.png").unlink()
        expect_value_error(lambda: inspect_split(split), "Missing matching image")
    print("ok preflight split self-test")


if __name__ == "__main__":
    main()
