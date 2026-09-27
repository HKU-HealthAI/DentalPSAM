"""Train the checkpoint-compatible DentalPSAM model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import monai
import numpy as np
import torch
from torch.utils.data import DataLoader

from dentalpsam.checkpoints import build_sam_vit_b
from dentalpsam.data import SAMDataset, load_and_patchify_png
from dentalpsam.model import DentalPSAM
from dentalpsam.mesh_targets import (
    binary_mesh_intersection_union,
    binary_mesh_targets,
    masked_mesh_loss,
    soft_mesh_targets,
    valid_mesh_point_mask,
)


PREDICTION_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_output_directory(output_dir: Path, source_dirs: tuple[Path, Path]) -> Path:
    output = output_dir.expanduser().resolve()
    for source_dir in source_dirs:
        source = source_dir.expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("Refusing to write a training run inside source data.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty --save-dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def inspect_patient_split(
    train_dir: Path,
    val_dir: Path,
    participant_prefix_length: int,
) -> dict:
    """Reject train/validation participant overlap using the declared ID prefix."""
    if participant_prefix_length < 1:
        raise ValueError("--participant-id-prefix-length must be positive.")

    def read_ids(data_dir: Path) -> tuple[list[str], list[str]]:
        mesh_ids = sorted(path.stem for path in (data_dir / "label_mesh").glob("*.npz"))
        if not mesh_ids:
            raise ValueError(f"No label_mesh NPZ files found in {data_dir}")
        participants = sorted({mesh_id[:participant_prefix_length] for mesh_id in mesh_ids})
        return mesh_ids, participants

    train_meshes, train_participants = read_ids(train_dir)
    val_meshes, val_participants = read_ids(val_dir)
    overlap = sorted(set(train_participants) & set(val_participants))
    if overlap:
        raise ValueError(
            f"Train/validation participant overlap detected for {len(overlap)} ID prefix(es)."
        )
    return {
        "participant_id_prefix_length": participant_prefix_length,
        "train_mesh_count": len(train_meshes),
        "train_participant_count": len(train_participants),
        "train_mesh_ids_sha256": sha256_lines(train_meshes),
        "train_participant_ids_sha256": sha256_lines(train_participants),
        "val_mesh_count": len(val_meshes),
        "val_participant_count": len(val_participants),
        "val_mesh_ids_sha256": sha256_lines(val_meshes),
        "val_participant_ids_sha256": sha256_lines(val_participants),
    }


def set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    min_positive_pixels: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Load aligned patches, applying the same filter to images and mesh rows.

    Both train and validation use this loader. The caller selects shuffling;
    the fixed-list test pipeline uses a separate loader without patch filtering.
    """
    images, masks, mesh, labels = load_and_patchify_png(
        str(data_dir), min_positive_pixels=min_positive_pixels
    )
    dataset = SAMDataset(images, masks, mesh, labels)
    if len(dataset) == 0:
        raise ValueError(f"No eligible patches were loaded from {data_dir}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    image_loss_fn: torch.nn.Module,
    mesh_loss_fn: torch.nn.Module,
    image_loss_weight: float,
    mesh_loss_weight: float,
    target_threshold: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Run training when an optimizer is supplied, otherwise validation.

    Combine 2D Dice-CE and mesh BCE losses; exclude padded faces from mesh
    supervision. Returned IoUs are training diagnostics, not the fused mesh
    estimates produced by the public test command.
    """
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "image_loss": 0.0, "mesh_loss": 0.0}
    image_intersection = image_union = 0
    mesh_intersection = mesh_union = 0

    for batch in dataloader:
        images = batch["pixel_values"].to(device)
        image_targets = batch["ground_truth_mask"].float().to(device)
        mesh_inputs = batch["SOTA_mesh"].float().to(device)
        mesh_labels = batch["label_mesh"].float().to(device)

        with torch.set_grad_enabled(training):
            outputs = model(image=images, SOTA_mesh=mesh_inputs)
            image_loss = image_loss_fn(outputs["pred_masks"], image_targets)
            valid_mesh = valid_mesh_point_mask(mesh_labels)
            mesh_loss = masked_mesh_loss(
                mesh_loss_fn,
                outputs["pred_mesh"],
                soft_mesh_targets(mesh_labels),
                valid_mesh,
            )
            loss = image_loss_weight * image_loss + mesh_loss_weight * mesh_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        image_predictions = torch.sigmoid(outputs["pred_masks"]) >= PREDICTION_THRESHOLD
        binary_image_targets = image_targets >= target_threshold
        image_intersection += int((image_predictions & binary_image_targets).sum().item())
        image_union += int((image_predictions | binary_image_targets).sum().item())

        mesh_predictions = torch.sigmoid(outputs["pred_mesh"]) >= PREDICTION_THRESHOLD
        binary_targets = binary_mesh_targets(mesh_labels, target_threshold)
        intersection, union = binary_mesh_intersection_union(
            mesh_predictions, binary_targets, valid_mesh
        )
        mesh_intersection += int(intersection.item())
        mesh_union += int(union.item())

        totals["loss"] += float(loss.detach().item())
        totals["image_loss"] += float(image_loss.detach().item())
        totals["mesh_loss"] += float(mesh_loss.detach().item())

    batch_count = len(dataloader)
    if batch_count == 0:
        raise ValueError("Dataloader produced no batches.")
    return {
        **{key: value / batch_count for key, value in totals.items()},
        "image_iou": image_intersection / image_union if image_union else float("nan"),
        "mesh_iou": mesh_intersection / mesh_union if mesh_union else float("nan"),
    }


def run_training(args) -> None:
    positive_integers = (args.epochs, args.batch_size, args.save_every)
    if any(value < 1 for value in positive_integers) or args.num_workers < 0:
        raise ValueError("Epochs, batch size and save interval must be positive; workers cannot be negative.")
    if args.min_positive_pixels < 0:
        raise ValueError("--min-positive-pixels must be non-negative.")
    if not 0.0 <= args.target_threshold <= 1.0:
        raise ValueError("--target-threshold must be in [0, 1].")

    train_dir = args.train_dir.expanduser().resolve()
    val_dir = args.val_dir.expanduser().resolve()
    sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    if train_dir == val_dir:
        raise ValueError("--train-dir and --val-dir must be different.")
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise NotADirectoryError("Training and validation directories must exist.")
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(sam_checkpoint)

    output_dir = prepare_output_directory(args.save_dir, (train_dir, val_dir))
    split_contract = inspect_patient_split(
        train_dir, val_dir, args.participant_id_prefix_length
    )
    set_random_seed(args.seed, args.deterministic)

    manifest_path = output_dir / "training_run_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "started",
        "entry_point": Path(__file__).name,
        "entry_point_sha256": sha256_file(Path(__file__).resolve()),
        "command": sys.argv,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "sam_checkpoint": str(sam_checkpoint),
        "sam_checkpoint_sha256": sha256_file(sam_checkpoint),
        "split_contract": split_contract,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "image_loss_weight": args.image_loss_weight,
        "mesh_loss_weight": args.mesh_loss_weight,
        "mesh_fusion": args.mesh_fusion,
        "min_positive_pixels": args.min_positive_pixels,
        "target_threshold": args.target_threshold,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "seed": args.seed,
        "deterministic_requested": args.deterministic,
        "selection_metric": "validation mesh BCE loss over non-padding tokens",
        "early_stopping": None,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "monai": monai.__version__,
        },
    }
    write_json(manifest_path, manifest)

    try:
        train_loader = make_loader(
            train_dir,
            args.batch_size,
            args.num_workers,
            args.min_positive_pixels,
            True,
            args.seed,
        )
        val_loader = make_loader(
            val_dir,
            args.batch_size,
            args.num_workers,
            args.min_positive_pixels,
            False,
            args.seed + 1,
        )
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
        sam = build_sam_vit_b(sam_checkpoint).to(device)
        model = DentalPSAM(sam, mesh_fusion=args.mesh_fusion).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        image_loss_fn = monai.losses.DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        mesh_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        history = []
        best_val_mesh_loss = float("inf")
        best_checkpoint = None
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                image_loss_fn,
                mesh_loss_fn,
                args.image_loss_weight,
                args.mesh_loss_weight,
                args.target_threshold,
                optimizer,
            )
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    image_loss_fn,
                    mesh_loss_fn,
                    args.image_loss_weight,
                    args.mesh_loss_weight,
                    args.target_threshold,
                    None,
                )
            scheduler.step()
            history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
            write_json(output_dir / "history.json", {"epochs": history})
            print(
                f"epoch={epoch}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_mesh_iou={val_metrics['mesh_iou']:.6f}",
                flush=True,
            )

            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": val_metrics,
            }
            if epoch % args.save_every == 0:
                torch.save(checkpoint_payload, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")
            if val_metrics["mesh_loss"] < best_val_mesh_loss:
                best_val_mesh_loss = val_metrics["mesh_loss"]
                best_checkpoint = output_dir / "best_model.pth"
                torch.save(checkpoint_payload, best_checkpoint)
    except Exception as error:
        manifest.update({
            "status": "failed",
            "error_type": type(error).__name__,
            "error_message": str(error),
        })
        write_json(manifest_path, manifest)
        raise

    manifest.update({
        "status": "completed",
        "completed_epochs": args.epochs,
        "best_checkpoint": str(best_checkpoint),
        "best_validation_mesh_loss": best_val_mesh_loss,
    })
    write_json(manifest_path, manifest)
