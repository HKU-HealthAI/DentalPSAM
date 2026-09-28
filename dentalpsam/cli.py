"""The two public commands; imports of model libraries are deferred until execution."""

import argparse
from pathlib import Path
import sys

from dentalpsam.arguments import branch3d_train_parser, dentalpsam_train_parser


def train_main(argv=None):
    selector = argparse.ArgumentParser(
        description="Train the 3D branch, then DentalPSAM.",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stage 1: python train.py --stage 3d --data data --output outputs/3d\n"
            "Stage 2: python train.py --stage dentalpsam --data data\n"
            "           --branch-checkpoint outputs/3d/best.pth --output outputs/dentalpsam\n"
            "See docs/TRAINING.md for data preparation and checkpoint requirements."
        ),
    )
    selector.add_argument("--stage", choices=("3d", "dentalpsam"), required=True)
    tokens = sys.argv[1:] if argv is None else argv
    if not tokens or ("--help" in tokens and "--stage" not in tokens):
        selector.print_help()
        print("Use --stage 3d --help or --stage dentalpsam --help for stage options.")
        return
    stage, _ = selector.parse_known_args(tokens)
    parser = (
        branch3d_train_parser() if stage.stage == "3d" else dentalpsam_train_parser()
    )
    parser.add_argument("--stage", choices=("3d", "dentalpsam"), required=True)
    parser.add_argument(
        "--data", type=Path, help="Dataset root containing train/ and val/"
    )
    if stage.stage == "dentalpsam":
        parser.add_argument(
            "--branch-checkpoint",
            "--3d-checkpoint",
            dest="branch_checkpoint",
            type=Path,
            help="Frozen 3D branch checkpoint used to prepare training inputs",
        )
    args = parser.parse_args(tokens)
    if args.data is not None:
        if args.train_dir is not None or args.val_dir is not None:
            parser.error("Use --data or --train-dir/--val-dir, not both")
        args.train_dir, args.val_dir = args.data / "train", args.data / "val"
    if args.train_dir is None or args.val_dir is None:
        parser.error("Provide --data or both --train-dir and --val-dir")
    from dentalpsam.workflows import train_model

    try:
        train_model(args)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))


def test_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run DentalPSAM prediction and paper evaluation on labelled scans.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python test.py --data data/test --weights checkpoints --output results. "
            "Use a new output directory. Read summary.txt for scores and run.log for "
            "details. Data layout: docs/DATASET.md."
        ),
    )
    inputs = parser.add_argument_group("Data, weights, and results")
    inputs.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Test split containing meshes/, labels/, and mesh_ids.txt",
    )
    weights = inputs.add_mutually_exclusive_group(required=True)
    weights.add_argument(
        "--weights",
        type=Path,
        help="Directory containing dentalpsam.pth, branch3d.pth, and sam.pth",
    )
    weights.add_argument(
        "--checkpoint", type=Path, help="Advanced: explicit DentalPSAM checkpoint file"
    )
    inputs.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New result directory outside the input data",
    )
    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--device", default="cuda:0", help="PyTorch device")
    runtime.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    advanced = parser.add_argument_group("Advanced overrides")
    advanced.add_argument(
        "--sam-checkpoint", type=Path, help="Advanced: override SAM initialization"
    )
    advanced.add_argument(
        "--branch-checkpoint",
        "--3d-checkpoint",
        dest="branch_checkpoint",
        type=Path,
        help="Advanced: matching 3D branch weights",
    )
    advanced.add_argument(
        "--mesh-list", type=Path, help="Advanced: override DATA/mesh_ids.txt"
    )
    advanced.add_argument("--expected-count", type=int, help="Assert fixed test-list size")
    advanced.add_argument("--bootstrap-reps", type=int, default=10000,
                          help="Participant bootstrap repetitions for confidence intervals")
    advanced.add_argument("--seed", type=int, default=42, help="Bootstrap random seed")
    args = parser.parse_args(argv)
    if args.weights is not None:
        args.checkpoint = args.weights / "dentalpsam.pth"
        args.sam_checkpoint = args.sam_checkpoint or args.weights / "sam.pth"
        args.branch_checkpoint = args.branch_checkpoint or args.weights / "branch3d.pth"
    elif args.branch_checkpoint is None or args.sam_checkpoint is None:
        parser.error("Explicit --checkpoint requires --branch-checkpoint and --sam-checkpoint; alternatively use --weights DIR")
    from dentalpsam.workflows import test_model

    try:
        test_model(args)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))
