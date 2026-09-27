#!/usr/bin/env python3
"""Export a frozen the 3D branch checkpoint to DentalPSAM SOTA-mesh patches."""
from dentalpsam.arguments import export_features_parser


def main():
    args = export_features_parser().parse_args()
    from tsgcnet.features import export_features

    export_features(args)


if __name__ == "__main__":
    main()
