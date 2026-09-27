#!/usr/bin/env python3
"""Render the three DentalPSAM UV views and reconstruction metadata."""
from dentalpsam.arguments import prepare_views_parser


def main():
    args = prepare_views_parser().parse_args()
    from dentalpsam.preparation import prepare_views
    prepare_views(args)


if __name__ == "__main__":
    main()
