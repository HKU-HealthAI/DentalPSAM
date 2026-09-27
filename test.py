#!/usr/bin/env python3
"""Run fixed-checkpoint DentalPSAM prediction and original paper evaluation."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Test split with origin/, label/, and prepared manual_2D/",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="Default: sam_vit_b_01ec64.pth beside the DentalPSAM checkpoint",
    )
    parser.add_argument(
        "--3d-checkpoint",
        dest="branch_checkpoint",
        type=Path,
        help="Generate fresh inputs from PLY with these fixed 3D weights",
    )
    parser.add_argument(
        "--mesh-list",
        type=Path,
        help="Fixed list; default: DATA/mesh_ids.txt (no automatic case selection)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from dentalpsam.workflows import test_model

    test_model(args)


if __name__ == "__main__":
    main()
