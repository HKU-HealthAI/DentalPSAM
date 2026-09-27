#!/usr/bin/env python3
"""Train the 3D branch or DentalPSAM; run --stage STAGE --help for all options."""

import argparse
from pathlib import Path

from dentalpsam.arguments import branch3d_train_parser, dentalpsam_train_parser


def main(argv=None):
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--stage", choices=("3d", "dentalpsam"), required=True)
    import sys

    tokens = sys.argv[1:] if argv is None else argv
    if not tokens or ("--help" in tokens and "--stage" not in tokens):
        selector.print_help()
        return
    stage, _ = selector.parse_known_args(tokens)
    parser = (
        branch3d_train_parser() if stage.stage == "3d" else dentalpsam_train_parser()
    )
    parser.add_argument("--stage", choices=("3d", "dentalpsam"), required=True)
    parser.add_argument(
        "--data", type=Path, help="Dataset root with train/ and val/ splits"
    )
    if stage.stage == "dentalpsam":
        parser.add_argument(
            "--3d-checkpoint",
            dest="branch_checkpoint",
            type=Path,
            help="Generate fresh 3D inputs with this frozen checkpoint",
        )
    args = parser.parse_args(tokens)
    if args.data is not None:
        if args.train_dir is not None or args.val_dir is not None:
            parser.error("Use --data or --train-dir/--val-dir, not both")
        args.train_dir, args.val_dir = args.data / "train", args.data / "val"
        if args.stage == "dentalpsam" and args.branch_checkpoint is None:
            args.train_dir /= "manual_2D"
            args.val_dir /= "manual_2D"
    if args.train_dir is None or args.val_dir is None:
        parser.error("Provide --data or both --train-dir and --val-dir")
    if args.stage == "3d":
        from tsgcnet.training import run_training
    else:
        if args.branch_checkpoint is not None:
            from dentalpsam.workflows import prepare_training_inputs

            args.train_dir, args.val_dir = prepare_training_inputs(args)
        from dentalpsam.training import run_training
    run_training(args)


if __name__ == "__main__":
    main()
