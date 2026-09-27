"""Argument definitions for public commands; no model computation."""

import argparse
from pathlib import Path


def dentalpsam_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the checkpoint-compatible DentalPSAM model."
    )
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam.pth"),
                        help="SAM initialization (default: checkpoints/sam.pth)")
    parser.add_argument(
        "--save-dir", "--output", dest="save_dir", type=Path, required=True
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-loss-weight", type=float, default=2.0)
    parser.add_argument("--mesh-loss-weight", type=float, default=1.0)
    parser.add_argument("--mesh-fusion", choices=("gated", "concat"), default="gated")
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--min-positive-pixels", type=int, default=50)
    parser.add_argument("--participant-id-prefix-length", type=int, default=4)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser


def branch3d_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 3D branch with patient-disjoint, area-weighted model selection."
    )
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--val-dir", type=Path)
    parser.add_argument(
        "--out-dir", "--output", dest="out_dir", type=Path, required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--plaque-class-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=12)
    return parser


def predict_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export DentalPSAM 2D and 3D probabilities for a fixed mesh list."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a frozen DentalPSAM checkpoint on a declared validation split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-positive-pixels", type=int, default=50)
    parser.add_argument("--image-loss-weight", type=float, default=2.0)
    parser.add_argument("--mesh-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def evaluate_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate MICCAI predictions with the original equal-triangle protocol."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fusion-weight-2d", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-count", type=int)
    return parser


def prepare_views_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the three DentalPSAM UV views and reconstruction metadata."
    )
    parser.add_argument("--origin-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def export_features_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export frozen 3D branch features as DentalPSAM mesh patches."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--info-dir",
        type=Path,
        help="UV reconstruction NPZ directory (default: DATA_DIR/manual_2D/info).",
    )
    parser.add_argument("--mesh-list", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=12)
    return parser


def branch3d_validate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a frozen 3D branch checkpoint with original-triangle area weights."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=12)
    return parser


def branch3d_evaluate_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen 3D branch checkpoint with original equal-triangle metrics."
    )
    for name in ("data-dir", "checkpoint", "mesh-list", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser
