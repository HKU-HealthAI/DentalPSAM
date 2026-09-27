"""Shared I/O helpers for DentalPSAM prediction and evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Union


def flatten_patch_lists(patch_lists: Iterable[Iterable]) -> list:
    """Flatten per-patch arrays while preserving their original order."""
    return [item for patch in patch_lists for item in patch]


PathLike = Union[str, os.PathLike[str]]


def load_face_orders(base_dir: PathLike, mesh_name: str) -> tuple[list, list, list]:
    """Load and flatten face-order arrays for the upper, inner, and outer views."""
    import numpy as np

    order_npz = Path(base_dir) / "label_mesh" / f"{mesh_name}.npz"
    with np.load(order_npz, allow_pickle=True) as mesh_file:
        return (
            flatten_patch_lists(mesh_file["face_order_up"]),
            flatten_patch_lists(mesh_file["face_order_in"]),
            flatten_patch_lists(mesh_file["face_order_out"]),
        )


def restore_face_order(values, face_order):
    """Restore flattened predictions to the original mesh face order.

    ``face_order`` must be a complete permutation of ``0..N-1`` for the
    current UV view. Rejecting malformed order arrays here prevents silently
    overwriting duplicated destinations or leaving zero-filled face scores in
    exported predictions.
    """
    import numpy as np

    face_order = [int(order) for order in face_order]
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) != len(face_order):
        raise ValueError(
            f"Prediction count ({len(values)}) does not match face-order count ({len(face_order)})."
        )
    if not face_order:
        return np.array([], dtype=np.float32)
    expected_order = list(range(len(face_order)))
    if sorted(face_order) != expected_order:
        raise ValueError(
            "face_order must be a complete permutation of 0..N-1; "
            "found duplicated, missing, or negative face indices."
        )

    restored = np.empty(len(face_order), dtype=np.float32)
    for idx, order in enumerate(face_order):
        restored[order] = values[idx]
    return restored


def make_dataloader(dataset, num_workers: int):
    """Create the single-sample DataLoader used by prediction and evaluation."""
    from torch.utils.data import DataLoader

    loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 3
    return DataLoader(dataset, **loader_kwargs)


def read_mesh_list(mesh_list: PathLike) -> list[str]:
    """Read mesh ids from a one-id-per-line split file."""
    with open(mesh_list, "r") as f:
        mesh_names = []
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                mesh_names.append(stripped)
        return mesh_names


def mesh_names_from_label_mesh(test_dir: PathLike) -> list[str]:
    """Resolve mesh ids from label_mesh NPZ files."""
    label_mesh_dir = Path(test_dir) / "label_mesh"
    return sorted(path.stem for path in label_mesh_dir.glob("*.npz"))


def mesh_names_from_origin(test_dir: PathLike) -> list[str]:
    """Resolve mesh ids from origin PNG view files."""
    origin_dir = Path(test_dir) / "origin"
    origin_files = sorted(path.stem for path in origin_dir.glob("*.png"))
    return sorted({name.rsplit("_", 1)[0] if "_" in name else name for name in origin_files})
