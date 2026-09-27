#!/usr/bin/env python3
"""Evaluate one frozen the 3D branch checkpoint with original equal-triangle metrics."""
from dentalpsam.arguments import branch3d_evaluate_parser


def main():
    args = branch3d_evaluate_parser().parse_args()
    from dentalpsam.branch3d.reporting import evaluate_checkpoint

    evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
