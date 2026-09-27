#!/usr/bin/env python3
"""Export DentalPSAM 2D and 3D probabilities for a fixed mesh list."""
from dentalpsam.arguments import predict_parser


def main():
    args = predict_parser().parse_args()
    from dentalpsam.inference import predict_split

    predict_split(args)


if __name__ == "__main__":
    main()
