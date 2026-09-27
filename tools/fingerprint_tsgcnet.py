#!/usr/bin/env python3
"""Identify checkpoint candidates by equality to cached scores, not test accuracy.

This compares one mesh's unmodified probabilities. It does not compute IoU,
select for performance, modify inputs, or certify whole-cohort reproduction.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dentalpsam.checkpoints import extract_model_state
from tools.audit_sota_inputs import read_cached_scores
from tools.evaluate_mesh import sha256_file
from tsgcnet.data import PlyDataset
from tsgcnet.model import TSGCNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "checkpoint-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mesh-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.data_dir, args.checkpoint_dir):
        if source.resolve() in args.output.resolve().parents:
            raise ValueError(
                "Evidence must be written outside data/checkpoint directories"
            )
    paths = sorted(args.checkpoint_dir.glob("*.pth"))
    if not paths:
        raise ValueError(
            "No checkpoint candidates in the explicitly supplied directory"
        )
    reference, faces = read_cached_scores(args.data_dir, args.mesh_id)
    dataset = PlyDataset(str(args.data_dir / "label"), enable_augmentation=False)
    dataset.file_list = [args.mesh_id + ".ply"]
    indices, features, _targets, _onehot, _name, _raw = dataset[0]
    np.testing.assert_array_equal(indices[: len(faces)], faces)
    device = torch.device(args.device)
    features = torch.as_tensor(features).float().T.unsqueeze(0).to(device)
    model = TSGCNet(in_channels=9, output_channels=2, k=12).to(device).eval()
    records = []
    for path in paths:
        record = {"checkpoint": str(path), "sha256": sha256_file(path)}
        try:
            state = extract_model_state(torch.load(path, map_location="cpu"))
            model.load_state_dict(state, strict=True)
            with torch.inference_mode():
                prediction = (
                    model(features, np.expand_dims(indices, 0))[0, : len(faces), 1]
                    .exp()
                    .cpu()
                    .numpy()
                )
            error = np.abs(prediction - reference)
            record.update(
                status="compared",
                exactly_equal=bool(np.array_equal(prediction, reference)),
                mean_abs=float(error.mean()),
                max_abs=float(error.max()),
            )
        except RuntimeError as error:
            record.update(status="incompatible", error=str(error))
        records.append(record)
        print(json.dumps(record), flush=True)
    payload = {
        "scope": "single-mesh probability fingerprint; no test-performance selection",
        "k": 12,
        "records": records,
        "reference_sha256": sha256_file(
            args.data_dir / "manual_2D/SOTA_mesh" / f"{args.mesh_id}.npz"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
