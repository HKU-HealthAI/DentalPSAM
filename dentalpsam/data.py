"""Aligned RGB, mask, and mesh-patch loading for DentalPSAM.

The numerical contract is historical: RGB values remain in [0, 255], labels
use ``gray > 127``, patches are row-major, and mesh rows are LEFT padded to
6000 x 10. Never substitute labels for the auxiliary SOTA score channel.
Object-array NPZ files require pickle support; load only trusted study files.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


PATCH_SIZE = 256
MESH_PATCH_SHAPE = (6000, 10)
VIEW_KEYS = {"0": "up", "1": "in", "2": "out"}
VIEW_IMAGE_SHAPES = {"0": (512, 768), "1": (256, 2048), "2": (256, 2048)}
REQUIRED_SPLIT_DIRS = ("origin", "label", "SOTA_mesh", "label_mesh")


def require_split_dirs(base_dir):
    """Reject incomplete manual_2D layouts before loading any patches."""
    for name in REQUIRED_SPLIT_DIRS:
        path = Path(base_dir) / name
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required {name} directory: {path}")


def read_image_or_raise(path, flags, description):
    """Read one image; a missing or unreadable file is an error, not a skip."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not load {description} image: {path}")
    return image


def load_npz_or_raise(path, description):
    """Open a trusted object-array NPZ; the caller must close the result."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required {description} NPZ: {path}")
    return np.load(path, allow_pickle=True)


class SAMDataset(Dataset):
    """Adapt aligned patches to the model's historical dictionary interface.

    Images are HWC arrays; returned image/mask tensors have shapes [3, 256,
    256] and [1, 256, 256]. Mesh arrays remain [6000, 10]. The unusual
    ``SOTA_mesh`` dictionary key is retained for existing training callers.
    """

    def __init__(self, train_images, train_masks, train_mesh, label_mesh):
        counts = list(map(len, (train_images, train_masks, train_mesh, label_mesh)))
        if len(set(counts)) != 1:
            raise ValueError(f"Image/mask/mesh dataset lengths differ: {counts}")
        self.images = train_images
        self.masks = train_masks
        self.mesh = train_mesh
        self.label_mesh = label_mesh

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "pixel_values": torch.tensor(self.images[index]).float().permute(2, 0, 1),
            "ground_truth_mask": torch.tensor(self.masks[index]).float().unsqueeze(0),
            "SOTA_mesh": np.array(self.mesh[index]),
            "label_mesh": np.array(self.label_mesh[index]),
        }


def patchify(image, name, patch_size=PATCH_SIZE):
    """Split a divisible image in row-major order, preserving legacy names.

    Do not resize or discard edge pixels: both would change the correspondence
    between rendered patches, mesh rows, and the saved face-order metadata.
    """
    if patch_size < 1 or image.ndim not in (2, 3):
        raise ValueError("Expected a 2D/3D image and positive patch size")
    height, width = image.shape[:2]
    if not height or not width or height % patch_size or width % patch_size:
        raise ValueError(f"Image shape {image.shape} is not divisible by {patch_size}")
    patches, names = [], []
    for row in range(height // patch_size):
        for column in range(width // patch_size):
            patches.append(
                image[
                    row * patch_size : (row + 1) * patch_size,
                    column * patch_size : (column + 1) * patch_size,
                ]
            )
            names.append(f"{name[:-4]}0{row}0{column}")
    return np.array(patches), np.array(names)


def pad(data, output_shape):
    """Left-pad mesh rows without changing channel order, values, or dtype.

    Empty historical patches can be shape (0,) or (0, 10); they produce a
    float64 zero patch, as before. Nonempty patches must have exactly ten
    columns. Overlong patches are errors, never truncated to fit a model.
    """
    patches = []
    for matrix in data:
        matrix = np.asarray(matrix)
        if matrix.shape in ((0,), (0, output_shape[1])):
            patches.append(np.zeros(output_shape))
            continue
        if (
            matrix.ndim != 2
            or matrix.shape[1] != output_shape[1]
            or matrix.shape[0] > output_shape[0]
        ):
            raise ValueError(
                f"Mesh patch shape {matrix.shape} cannot fit {output_shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Mesh patch contains non-finite values")
        padding_rows = output_shape[0] - matrix.shape[0]
        patches.append(np.pad(matrix, [(padding_rows, 0), (0, 0)], mode="constant"))
    return np.array(patches)


def load_npz(mesh_npz, position):
    """Load up/in/out mesh patches with the fixed 6000 x 10 padding contract."""
    if position not in VIEW_KEYS:
        raise ValueError(f"Unsupported view position {position!r}")
    return pad(mesh_npz[VIEW_KEYS[position]], MESH_PATCH_SHAPE)


def _load_view(base_dir, file_name):
    """Load one aligned view and validate all four patch counts before zipping."""
    match = re.fullmatch(r"(\d{6})_([012])\.png", file_name)
    if match is None:
        raise ValueError(f"Expected six-digit mesh ID and view 0/1/2: {file_name}")
    mesh_id, view = match.groups()
    base = Path(base_dir)
    image = read_image_or_raise(base / "origin" / file_name, cv2.IMREAD_COLOR, "origin")
    label = read_image_or_raise(
        base / "label" / file_name, cv2.IMREAD_GRAYSCALE, "label"
    )
    if (
        image.shape[:2] != VIEW_IMAGE_SHAPES[view]
        or label.shape != VIEW_IMAGE_SHAPES[view]
    ):
        raise ValueError(f"Unexpected image/label shape for view {file_name}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use the same target rule for training and prediction diagnostics.
    label = (label > 127).astype(np.uint8)
    images, _ = patchify(image, file_name)
    labels, _ = patchify(label, file_name)
    with load_npz_or_raise(
        base / "SOTA_mesh" / f"{mesh_id}.npz", "SOTA_mesh"
    ) as scores:
        mesh_patches = load_npz(scores, view)
    with load_npz_or_raise(
        base / "label_mesh" / f"{mesh_id}.npz", "label_mesh"
    ) as targets:
        target_patches = load_npz(targets, view)
    counts = list(map(len, (images, labels, mesh_patches, target_patches)))
    if len(set(counts)) != 1:
        raise ValueError(f"Image/mesh patch counts differ for {file_name}: {counts}")
    return images, labels, mesh_patches, target_patches


def _collect_patches(base_dir, file_names, min_positive_pixels):
    """Keep arrays aligned through optional training-only positive-patch filtering."""
    images, labels, meshes, targets, indices = [], [], [], [], []
    patch_index = 0
    for name in file_names:
        for image, label, mesh, target in zip(*_load_view(base_dir, name)):
            if np.count_nonzero(label) >= min_positive_pixels:
                images.append(image)
                labels.append(label)
                meshes.append(mesh)
                targets.append(target)
                indices.append(patch_index)
            patch_index += 1
    return (
        np.array(images),
        np.array(labels),
        indices,
        np.array(meshes),
        np.array(targets),
    )


def load_and_patchify_png(base_dir, min_positive_pixels=50):
    """Load a split in sorted PNG order, retaining the historical training filter.

    Returns images [P,256,256,3], masks [P,256,256], SOTA and target mesh
    patches [P,6000,10]. Non-PNG metadata is ignored; malformed PNGs fail.
    """
    require_split_dirs(base_dir)
    names = sorted(path.name for path in (Path(base_dir) / "label").glob("*.png"))
    if not names:
        raise ValueError("The split contains no label PNGs")
    images, labels, _indices, meshes, targets = _collect_patches(
        base_dir, names, min_positive_pixels
    )
    return images, labels, meshes, targets


def load_and_patchify_png_permesh(base_dir, mesh_name, min_positive_pixels=-1):
    """Load the fixed 22-patch order: six upper, eight inner, eight outer.

    Prediction uses the default -1 so every patch is retained, independent
    of its ground truth. Returned indices refer to the unfiltered 0..21 order.
    """
    require_split_dirs(base_dir)
    names = [f"{mesh_name}_{view}.png" for view in VIEW_KEYS]
    return _collect_patches(base_dir, names, min_positive_pixels)
