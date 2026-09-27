#!/usr/bin/env python3
"""Validate a frozen TSGCNet checkpoint with original-triangle area weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_tsgcnet import MetadataCache, load_model_state, sha256, validation_metrics
from tsgcnet.data import PlyDataset
from tsgcnet.model import TSGCNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    model = TSGCNet(in_channels=9, output_channels=2, k=args.k).to(device)
    model.load_state_dict(load_model_state(checkpoint, device), strict=True)
    metrics = validation_metrics(model, loader, metadata, device)
    payload = {
        "schema_version": 1,
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": 0.5,
        "aggregation": "participant macro after combining meshes by four-character ID prefix",
        "weighting": "original PLY triangle surface area",
        "padding_policy": "exclude repeated rows beyond raw PLY face count",
        "mesh_count": len(dataset.file_list),
        "metrics": metrics,
        "scope": "validation checkpoint diagnostic only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
