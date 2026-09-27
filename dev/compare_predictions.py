#!/usr/bin/env python3
"""Check exact numerical equality across a repository refactor, without labels."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def compare(left, right):
    files = sorted(path.relative_to(left) for path in left.rglob("*")
                   if path.suffix in (".npz", ".png"))
    other = sorted(path.relative_to(right) for path in right.rglob("*")
                   if path.suffix in (".npz", ".png"))
    if files != other or not files:
        raise ValueError("Prediction file sets differ or are empty")
    arrays = 0
    for relative in files:
        if relative.suffix == ".png":
            np.testing.assert_array_equal(cv2.imread(str(left / relative), -1),
                                          cv2.imread(str(right / relative), -1))
            arrays += 1
        else:
            with np.load(left / relative, allow_pickle=False) as a, np.load(right / relative, allow_pickle=False) as b:
                if sorted(a.files) != sorted(b.files):
                    raise ValueError("NPZ keys differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                    arrays += 1
    return {"files_compared": len(files), "arrays_compared": arrays,
            "exactly_equal": True,
            "file_set_sha256": hashlib.sha256("\n".join(map(str, files)).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.before.resolve(), args.after.resolve()):
        if source == args.output.resolve() or source in args.output.resolve().parents:
            raise ValueError("Comparison report must be outside the predictions")
    report = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
