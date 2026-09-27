"""Export DentalPSAM 2D and 3D probabilities for a fixed mesh list."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from patchify import unpatchify

from dentalpsam.checkpoints import load_dentalpsam
from dentalpsam.data import SAMDataset, load_and_patchify_png_permesh
from dentalpsam.mesh_targets import binary_mesh_targets, valid_mesh_point_mask
from dentalpsam.prediction_io import (
    load_face_orders,
    make_dataloader,
    read_mesh_list,
    restore_face_order,
)


PREDICTION_THRESHOLD = 0.5

# The loader emits six upper, eight inner, then eight outer patches. These
# slices reconstruct the view images; face_order metadata separately restores
# mesh rows. Neither operation may sort patches or faces by their predictions.
VIEW_SLICES = {
    0: ("up", slice(0, 6), (2, 3, 256, 256), (512, 768)),
    1: ("in", slice(6, 14), (1, 8, 256, 256), (256, 2048)),
    2: ("out", slice(14, 22), (1, 8, 256, 256), (256, 2048)),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Write a small JSON record atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare_output_directory(output_dir: Path, data_dir: Path) -> Path:
    """Create a fresh output directory outside the source-data tree."""
    output = output_dir.expanduser().resolve()
    source = data_dir.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("Refusing to write predictions inside --data-dir source data.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "3Dpred").mkdir()
    return output


def binary_iou_dice(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute positive-class diagnostic IoU/Dice at threshold 0.5."""
    predicted_positive = np.asarray(prediction) >= PREDICTION_THRESHOLD
    target_positive = np.asarray(target, dtype=bool)
    intersection = int(np.logical_and(predicted_positive, target_positive).sum())
    union = int(np.logical_or(predicted_positive, target_positive).sum())
    denominator = int(predicted_positive.sum() + target_positive.sum())
    if union == 0:
        return float("nan"), float("nan")
    return intersection / union, 2.0 * intersection / denominator


def write_probability_png(path: Path, probabilities: np.ndarray) -> None:
    """Write a [0, 1] probability raster as an 8-bit compatibility PNG."""
    # Native cv2.imwrite received float32 probabilities * 255 and rounded
    # during uint8 conversion. astype(uint8) truncates and changes fusion near
    # the threshold; convertScaleAbs preserves the native rounding for [0,255].
    image = cv2.convertScaleAbs(np.clip(probabilities * 255.0, 0, 255))
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write probability PNG: {path}")


def predict_mesh(
    mesh_id: str,
    data_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    num_workers: int,
    target_threshold: float,
) -> dict[str, float]:
    """Predict all 22 patches and restore the three original view-face orders."""
    images, masks, patch_indices, sota_mesh, label_mesh = load_and_patchify_png_permesh(
        str(data_dir), mesh_id
    )
    if len(images) != 22 or patch_indices != list(range(22)):
        raise ValueError(
            f"{mesh_id}: expected the fixed 22-patch order 0..21; "
            f"received {len(images)} patches with indices {patch_indices}."
        )

    face_orders = dict(zip(("up", "in", "out"), load_face_orders(data_dir, mesh_id)))
    dataloader = make_dataloader(
        SAMDataset(images, masks, sota_mesh, label_mesh), num_workers
    )

    mask_patches = np.empty((22, 256, 256), dtype=np.float32)
    face_score_patches: list[np.ndarray] = []
    mesh_scores: list[np.ndarray] = []
    mesh_targets: list[np.ndarray] = []
    image_scores: list[np.ndarray] = []
    image_targets: list[np.ndarray] = []

    with torch.no_grad():
        for patch_number, batch in enumerate(dataloader):
            outputs = model(
                image=batch["pixel_values"].to(device),
                SOTA_mesh=batch["SOTA_mesh"].float().to(device),
            )
            face_probabilities = torch.sigmoid(outputs["pred_mesh"])
            face_labels = batch["label_mesh"].float().to(device)
            valid_faces = valid_mesh_point_mask(face_labels)
            target = binary_mesh_targets(face_labels, target_threshold)

            valid_scores = (
                face_probabilities[valid_faces].detach().cpu().numpy().reshape(-1)
            )
            valid_targets = target[valid_faces].detach().cpu().numpy().reshape(-1)
            face_score_patches.append(valid_scores)
            mesh_scores.append(valid_scores)
            mesh_targets.append(valid_targets)

            mask_probabilities = (
                torch.sigmoid(outputs["pred_masks"].squeeze(1))
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            mask_patches[patch_indices[patch_number]] = mask_probabilities
            image_scores.append(mask_probabilities.reshape(-1))
            image_targets.append(
                (
                    batch["ground_truth_mask"].cpu().numpy().squeeze()
                    >= target_threshold
                ).reshape(-1)
            )

    for view_index, (
        view_name,
        patch_slice,
        patch_shape,
        image_shape,
    ) in VIEW_SLICES.items():
        restored = restore_face_order(
            np.concatenate(face_score_patches[patch_slice]), face_orders[view_name]
        )
        np.savez(
            output_dir / "3Dpred" / f"{mesh_id}_{view_index}.npz",
            concatenated_array=restored,
        )
        raster = unpatchify(mask_patches[patch_slice].reshape(patch_shape), image_shape)
        np.savez(output_dir / f"{mesh_id}_{view_index}.npz", probability=raster)
        write_probability_png(output_dir / f"{mesh_id}_{view_index}.png", raster)

    mesh_iou, mesh_dice = binary_iou_dice(
        np.concatenate(mesh_scores), np.concatenate(mesh_targets)
    )
    image_iou, image_dice = binary_iou_dice(
        np.concatenate(image_scores), np.concatenate(image_targets)
    )
    return {
        "mesh_iou": mesh_iou,
        "mesh_dice": mesh_dice,
        "image_iou": image_iou,
        "image_dice": image_dice,
    }


def predict_split(args) -> None:
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if not 0.0 <= args.target_threshold <= 1.0:
        raise ValueError("--target-threshold must be in [0, 1].")

    checkpoint = args.checkpoint.expanduser().resolve()
    sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    mesh_list = args.mesh_list.expanduser().resolve()
    for path in (checkpoint, sam_checkpoint, mesh_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    mesh_ids = read_mesh_list(mesh_list)
    if not mesh_ids:
        raise ValueError("--mesh-list contains no mesh IDs.")
    if len(mesh_ids) != len(set(mesh_ids)):
        raise ValueError("--mesh-list contains duplicate mesh IDs.")
    if args.expected_count is not None and len(mesh_ids) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} meshes, but --mesh-list contains {len(mesh_ids)}."
        )

    output_dir = prepare_output_directory(args.output_dir, data_dir)
    manifest_path = output_dir / "prediction_run_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "started",
        "entry_point": Path(__file__).name,
        "entry_point_sha256": sha256_file(Path(__file__).resolve()),
        "command": sys.argv,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "sam_checkpoint": str(sam_checkpoint),
        "sam_checkpoint_sha256": sha256_file(sam_checkpoint),
        "data_dir": str(data_dir),
        "mesh_list": str(mesh_list),
        "mesh_list_sha256": sha256_file(mesh_list),
        "mesh_count": len(mesh_ids),
        "target_threshold": args.target_threshold,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "device": args.device,
        "num_workers": args.num_workers,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    write_json(manifest_path, manifest)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")

    try:
        model = load_dentalpsam(sam_checkpoint, checkpoint, device)
        results = []
        for index, mesh_id in enumerate(mesh_ids, start=1):
            metrics = predict_mesh(
                mesh_id,
                data_dir,
                output_dir,
                model,
                device,
                args.num_workers,
                args.target_threshold,
            )
            results.append({"mesh_id": mesh_id, **metrics})
            rendered = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            print(f"[{index}/{len(mesh_ids)}] {mesh_id}: {rendered}", flush=True)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        write_json(manifest_path, manifest)
        raise

    metric_names = ("mesh_iou", "mesh_dice", "image_iou", "image_dice")
    diagnostic_means = {
        metric: float(np.nanmean([row[metric] for row in results]))
        for metric in metric_names
    }
    manifest.update(
        {
            "status": "completed",
            "processed_mesh_count": len(results),
            "diagnostic_mesh_macro_means": diagnostic_means,
            "reported_result_boundary": (
                "Diagnostics above are not the participant-level reported result; "
                "run the original evaluator or use the unified test.py workflow."
            ),
        }
    )
    write_json(manifest_path, manifest)
    write_json(
        output_dir / "prediction_summary.json",
        {"per_mesh": results, **diagnostic_means},
    )
    print(json.dumps(diagnostic_means, sort_keys=True), flush=True)
