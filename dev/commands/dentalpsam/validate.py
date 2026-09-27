#!/usr/bin/env python3
"""Validate a frozen DentalPSAM checkpoint on a declared validation split."""
from dentalpsam.arguments import validate_parser


def main():
    args = validate_parser().parse_args()
    from dentalpsam.validation import validate_checkpoint

    validate_checkpoint(args)


if __name__ == "__main__":
    main()
