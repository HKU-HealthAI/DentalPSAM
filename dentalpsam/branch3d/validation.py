"""Validate a frozen 3D branch checkpoint with equal-triangle metrics."""


from __future__ import annotations


import argparse


import json


from pathlib import Path


import torch


from torch.utils.data import DataLoader


from dentalpsam.branch3d.training import MetadataCache, validation_metrics


from dentalpsam.branch3d.data import PlyDataset


from dentalpsam.branch3d.checkpoints import load_branch3d


def validate_checkpoint(args) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    val_dir = args.val_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (val_dir / "origin").is_dir() or not (val_dir / "label").is_dir():
        raise NotADirectoryError("--val-dir must contain origin/ and label/.")
    if output.exists():
        raise FileExistsError(output)
    if output == val_dir or val_dir in output.parents:
        raise ValueError("Output must be outside the validation data directory.")

    dataset = PlyDataset(str(val_dir / "label"), enable_augmentation=False)
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise ValueError("Validation split contains no PLY meshes.")
    metadata = MetadataCache(val_dir, dataset.file_list)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    model = load_branch3d(checkpoint, device, k=args.k)
    metrics = validation_metrics(model, loader, metadata, device)
    payload = {
        "schema_version": 2,
        "normalization": model.normalization,
        "checkpoint": str(checkpoint),
        "threshold": 0.5,
        "decision_rule": "score > 0.5",
        "aggregation": "arithmetic mean of per-mesh metrics",
        "weighting": "equal original triangles",
        "padding_policy": "exclude repeated rows beyond raw PLY face count",
        "mesh_count": len(dataset.file_list),
        "metrics": metrics,
        "scope": "validation checkpoint diagnostic only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
