#!/usr/bin/env python3
"""Validate a frozen the 3D branch checkpoint with original-triangle area weights."""
from dentalpsam.arguments import branch3d_validate_parser


def main():
    args = branch3d_validate_parser().parse_args()
    from tsgcnet.validation import validate_checkpoint

    validate_checkpoint(args)


if __name__ == "__main__":
    main()
