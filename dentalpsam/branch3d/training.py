"""Train TSGCNet with patient-disjoint, area-weighted model selection.

The upstream TSGCNet dataloader pads a 15,999-face mesh by repeating its last
face.  This entry point excludes that repeated row from the loss and from
validation.  Checkpoint selection uses the fixed 0.5 threshold and the mean
physical-area plaque IoU across validation participants.  Test and External
directories are never accepted as command-line inputs.
"""


from __future__ import annotations


import argparse


import csv


import hashlib


import json


import os


import random


from collections import defaultdict


from pathlib import Path


from typing import Any


import numpy as np


import torch


from plyfile import PlyData


from torch.utils.data import DataLoader


PADDED_FACE_COUNT = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_id(mesh_name: str) -> str:
    stem = Path(mesh_name).stem
    if len(stem) < 4 or not stem[:4].isdigit():
        raise ValueError(f"Cannot derive participant identifier from {mesh_name}")
    return stem[:4]


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_mesh_target_and_area(label_path: Path, origin_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label = PlyData.read(str(label_path))
    origin = PlyData.read(str(origin_path))
    label_vertices = label["vertex"].data
    origin_vertices = origin["vertex"].data
    label_faces = np.asarray([row[0] for row in label["face"].data], dtype=np.int64)
    origin_faces = np.asarray([row[0] for row in origin["face"].data], dtype=np.int64)
    if not np.array_equal(label_faces, origin_faces):
        raise ValueError(f"Origin/label topology mismatch: {label_path.stem}")
    if len(label_faces) > PADDED_FACE_COUNT:
        raise ValueError(f"{label_path.stem} has more than {PADDED_FACE_COUNT} faces")

    colour_names = ("red", "green", "blue")
    if not all(name in label_vertices.dtype.names for name in colour_names):
        raise ValueError(f"Label PLY lacks RGB fields: {label_path}")
    colours = np.stack([label_vertices[name] for name in colour_names], axis=1)
    target = np.any(np.all(colours[label_faces] == 0, axis=2), axis=1).astype(np.int64)

    xyz = np.stack([origin_vertices[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    triangles = xyz[origin_faces]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.all(np.isfinite(area)) or np.any(area <= 0):
        raise ValueError(f"Non-positive or invalid triangle area: {origin_path}")
    return target, area


class MetadataCache:
    """Cache immutable target, area, and raw-face-count metadata."""

    def __init__(self, split_dir: Path, mesh_names: list[str]):
        self.values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in mesh_names:
            self.values[name] = read_mesh_target_and_area(
                split_dir / "label" / name,
                split_dir / "origin" / name,
            )

    def __getitem__(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.values[name]


def assert_split_contract(train_names: list[str], val_names: list[str]) -> dict[str, Any]:
    train_patients = {patient_id(name) for name in train_names}
    val_patients = {patient_id(name) for name in val_names}
    overlap = sorted(train_patients & val_patients)
    if overlap:
        raise ValueError(f"Train/validation participant overlap: {overlap}")
    return {
        "train_meshes": len(train_names),
        "train_patients": len(train_patients),
        "validation_meshes": len(val_names),
        "validation_patients": len(val_patients),
        "participant_overlap": overlap,
    }


def load_model_state(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor]:
    value = torch.load(checkpoint, map_location=device)
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in value and isinstance(value[key], dict):
                return value[key]
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {checkpoint}")
    return value


def validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    metadata: MetadataCache,
    device: torch.device,
) -> dict[str, float]:
    participant_parts: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for index_face, points, legacy_target, _onehot, names, _raw_points in loader:
            name = names[0]
            target, area = metadata[name]
            raw_count = len(target)
            if raw_count > points.shape[1]:
                raise ValueError(f"Dataloader truncated {name}")
            if not np.array_equal(legacy_target[0, :raw_count, 0].numpy(), target):
                raise ValueError(f"Upstream dataloader target differs from raw PLY target: {name}")
            log_probability = model(
                points.transpose(2, 1).to(torch.float32).to(device),
                index_face.numpy(),
            )
            score = log_probability[0, :raw_count, 1].exp().cpu().numpy()
            participant_parts[patient_id(name)].append((target, score, area))

    rows: list[dict[str, float]] = []
    for parts in participant_parts.values():
        target = np.concatenate([part[0] for part in parts]).astype(bool)
        score = np.concatenate([part[1] for part in parts])
        area = np.concatenate([part[2] for part in parts])
        prediction = score >= 0.5
        tp = float(area[target & prediction].sum())
        fp = float(area[~target & prediction].sum())
        fn = float(area[target & ~prediction].sum())
        tn = float(area[~target & ~prediction].sum())
        rows.append({
            "plaque_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "sensitivity": tp / (tp + fn) if tp + fn else 1.0,
            "specificity": tn / (tn + fp) if tn + fp else 1.0,
            "pred_coverage": (tp + fp) / (tp + fp + fn + tn),
        })
    return {
        f"patient_macro_{metric}": float(np.mean([row[metric] for row in rows]))
        for metric in rows[0]
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(args) -> None:
    args.train_dir = args.train_dir.expanduser().resolve()
    args.val_dir = args.val_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    if args.train_dir == args.val_dir:
        raise ValueError("Training and validation directories must differ")
    if args.epochs <= 0 or args.patience <= 0 or args.plaque_class_weight <= 0:
        raise ValueError("Epochs, patience, and class weight must be positive")
    for split in (args.train_dir, args.val_dir):
        if not (split / "label").is_dir() or not (split / "origin").is_dir():
            raise NotADirectoryError(f"Split must contain label/ and origin/: {split}")

    set_determinism(args.seed)
    args.out_dir.mkdir(parents=True)
    from dentalpsam.branch3d.data import PlyDataset
    from dentalpsam.branch3d.model import TSGCNet

    train_dataset = PlyDataset(str(args.train_dir / "label"), enable_augmentation=False)
    val_dataset = PlyDataset(str(args.val_dir / "label"), enable_augmentation=False)
    train_dataset.file_list = sorted(train_dataset.file_list)
    val_dataset.file_list = sorted(val_dataset.file_list)
    split_contract = assert_split_contract(train_dataset.file_list, val_dataset.file_list)

    train_metadata = MetadataCache(args.train_dir, train_dataset.file_list)
    val_metadata = MetadataCache(args.val_dir, val_dataset.file_list)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TSGCNet(in_channels=9, output_channels=2, k=args.k).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    class_weight = torch.tensor([1.0, args.plaque_class_weight], device=device)
    criterion = torch.nn.NLLLoss(weight=class_weight)

    log_rows: list[dict[str, Any]] = []
    best_iou = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for index_face, points, legacy_target, _onehot, names, _raw_points in train_loader:
            name = names[0]
            target, _area = train_metadata[name]
            raw_count = len(target)
            if not np.array_equal(legacy_target[0, :raw_count, 0].numpy(), target):
                raise ValueError(f"Upstream dataloader target differs from raw PLY target: {name}")
            optimizer.zero_grad(set_to_none=True)
            log_probability = model(
                points.transpose(2, 1).to(torch.float32).to(device),
                index_face.numpy(),
            )
            target_tensor = torch.from_numpy(target).to(torch.long).to(device)
            loss = criterion(log_probability[0, :raw_count], target_tensor)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = validation_metrics(model, val_loader, val_metadata, device)
        val_iou = metrics["patient_macro_plaque_iou"]
        scheduler.step(val_iou)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)
        write_csv(args.out_dir / "training_log.csv", log_rows)
        print(json.dumps(row, sort_keys=True), flush=True)

        if val_iou > best_iou + 1e-12:
            best_iou = val_iou
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "validation_metrics": metrics,
                "plaque_class_weight": args.plaque_class_weight,
                "seed": args.seed,
                "threshold": 0.5,
            }, args.out_dir / "best.pth")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = args.out_dir / "best.pth"
    manifest = {
        "schema_version": 1,
        "method": "TSGCNet",
        "selection_metric": "validation participant-macro original-triangle-area plaque IoU",
        "selection_threshold": 0.5,
        "best_epoch": best_epoch,
        "best_validation_patient_macro_area_plaque_iou": best_iou,
        "epochs_completed": len(log_rows),
        "maximum_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "plaque_class_weight": args.plaque_class_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "batch_size": 1,
        "padding_policy": "exclude repeated final face beyond raw PLY face count",
        "test_or_external_used_for_selection": False,
        "split_contract": split_contract,
        "train_dir": str(args.train_dir),
        "validation_dir": str(args.val_dir),
        "train_mesh_list_sha256": hashlib.sha256(
            "\n".join(train_dataset.file_list).encode("utf-8")
        ).hexdigest(),
        "validation_mesh_list_sha256": hashlib.sha256(
            "\n".join(val_dataset.file_list).encode("utf-8")
        ).hexdigest(),
        "source": {
            "trainer": str(Path(__file__).resolve()),
            "trainer_sha256": sha256(Path(__file__).resolve()),
            "model_sha256": sha256(Path(__file__).parent / "model.py"),
            "dataloader_sha256": sha256(Path(__file__).parent / "data.py"),
            "utils_sha256": sha256(Path(__file__).parent / "utils.py"),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
