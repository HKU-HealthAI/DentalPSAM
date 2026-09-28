"""Validate a frozen DentalPSAM checkpoint on a declared validation split."""


from __future__ import annotations


import argparse


import json


from pathlib import Path


import monai


import torch


from dentalpsam.checkpoints import load_dentalpsam


from dentalpsam.training import make_loader, run_epoch


def validate_checkpoint(args) -> None:
    checkpoint = args.checkpoint.expanduser().resolve()
    sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    val_dir = args.val_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (checkpoint, sam_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not val_dir.is_dir():
        raise NotADirectoryError(val_dir)
    if output.exists():
        raise FileExistsError(output)
    if output == val_dir or val_dir in output.parents:
        raise ValueError("Output must be outside the validation data directory.")
    if not 0.0 <= args.target_threshold <= 1.0:
        raise ValueError("--target-threshold must be in [0, 1].")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    loader = make_loader(
        val_dir, args.batch_size, args.num_workers, args.min_positive_pixels, False, args.seed
    )
    model = load_dentalpsam(sam_checkpoint, checkpoint, device)
    image_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    mesh_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    with torch.inference_mode():
        metrics = run_epoch(
            model, loader, device, image_loss, mesh_loss,
            args.image_loss_weight, args.mesh_loss_weight, args.target_threshold, None,
        )
    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "sam_checkpoint": str(sam_checkpoint),
        "target_threshold": args.target_threshold,
        "prediction_threshold": 0.5,
        "patch_selection": f"at least {args.min_positive_pixels} positive label pixels",
        "metrics": metrics,
        "scope": "validation diagnostics; not participant-level reportable test metrics",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))
