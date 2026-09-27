"""Check wheel contents without installing any development utilities."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    args = parser.parse_args()
    wheels = list(args.wheel_dir.glob("dentalpsam-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one DentalPSAM wheel")
    with ZipFile(wheels[0]) as archive:
        files = archive.namelist()
    unexpected = [
        name
        for name in files
        if not name.startswith("dentalpsam/") and ".dist-info/" not in name
    ]
    if unexpected:
        raise ValueError(f"Non-package files included in wheel: {unexpected}")
    assert "dentalpsam/branch3d/model.py" in files
    assert "dentalpsam/README.md" in files
    assert "dentalpsam/segment_anything/LICENSE" in files
    print("PASS: wheel installs only dentalpsam and its distribution metadata")


if __name__ == "__main__":
    main()
