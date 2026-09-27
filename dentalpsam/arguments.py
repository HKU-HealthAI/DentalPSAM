"""Argument definitions for public commands; no model computation."""

import argparse
from pathlib import Path


def dentalpsam_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2: train DentalPSAM using the frozen 3D branch inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Selected weights: OUTPUT/best_model.pth. See docs/TRAINING.md.",
    )
    parser.add_argument("--train-dir", type=Path, help="Advanced: explicit training split instead of --data")
    parser.add_argument("--val-dir", type=Path, help="Advanced: explicit validation split instead of --data")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam.pth"),
                        help="SAM initialization (default: checkpoints/sam.pth)")
    parser.add_argument(
        "--output", "--save-dir", dest="save_dir", type=Path, required=True,
        help="New directory for checkpoints and training history",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Aligned image/mesh patches per batch")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay")
    parser.add_argument("--image-loss-weight", type=float, default=2.0, help="2D Dice-CE loss multiplier")
    parser.add_argument("--mesh-loss-weight", type=float, default=1.0, help="Mesh BCE loss multiplier")
    parser.add_argument("--mesh-fusion", choices=("gated", "concat"), default="gated",
                        help="Mesh-decoder architecture; not the test-time fusion weights")
    parser.add_argument("--save-every", type=int, default=2, help="Periodic checkpoint interval in epochs")
    parser.add_argument("--min-positive-pixels", type=int, default=50,
                        help="Minimum plaque pixels per patch in the training and validation loaders")
    parser.add_argument("--participant-id-prefix-length", type=int, default=4,
                        help="Filename-prefix length used by the prepared-input split check")
    parser.add_argument("--target-threshold", type=float, default=0.5,
                        help="Binary target threshold for training diagnostics; does not change test evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch operations")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    return parser


def branch3d_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 1: train the 3D branch using participant-disjoint splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("Selected weights: OUTPUT/best.pth. Selection uses mesh-macro equal-triangle "
                "validation plaque IoU with strict probability > 0.5. "
                "See docs/TRAINING.md."),
    )
    parser.add_argument("--train-dir", type=Path, help="Advanced: explicit training split instead of --data")
    parser.add_argument("--val-dir", type=Path, help="Advanced: explicit validation split instead of --data")
    parser.add_argument(
        "--output", "--out-dir", dest="out_dir", type=Path, required=True,
        help="New directory for checkpoints and training history",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, default=12, help="Early-stop patience in validation epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--plaque-class-weight", type=float, default=1.0, help="Plaque-class NLL loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Training random seed")
    parser.add_argument("--k", type=int, default=12, help="3D neighbourhood size; must match feature export")
    return parser
