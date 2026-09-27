#!/usr/bin/env python3
"""Evaluate MICCAI predictions with the original equal-triangle protocol."""
from dentalpsam.arguments import evaluate_parser


def main():
    args = evaluate_parser().parse_args()
    from dentalpsam.reporting import evaluate_predictions

    evaluate_predictions(args)


if __name__ == "__main__":
    main()
