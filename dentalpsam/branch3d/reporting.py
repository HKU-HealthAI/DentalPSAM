"""Evaluate one frozen TSGCNet checkpoint with original equal-triangle metrics.

This performs no checkpoint selection or training. A fixed mesh list is
required; no meshes are dropped. Repeated padding faces are excluded.
"""


from __future__ import annotations


import argparse


import csv


import json


from pathlib import Path


import numpy as np


import torch


from dentalpsam.checkpoints import extract_model_state


from dentalpsam.evaluation import cluster_bootstrap, equal_face_metrics, read_mesh_ids


from dentalpsam.mesh_io import hash_named_files, read_ply_mesh, sha256_file


from dentalpsam.branch3d.data import PlyDataset


from dentalpsam.branch3d.model import TSGCNet


def evaluate_checkpoint(args) -> None:
    source, output = args.data_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the input data tree")
    if output.exists():
        raise FileExistsError(output)
    ids = read_mesh_ids(args.mesh_list)
    if args.expected_count is not None and len(ids) != args.expected_count:
        raise ValueError("Mesh count differs from --expected-count")
    if args.bootstrap_reps < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    dataset = PlyDataset(str(source / "label"), enable_augmentation=False)
    dataset.file_list = [f"{name}.ply" for name in ids]
    files = [("mesh_list", args.mesh_list), ("checkpoint", args.checkpoint)]
    for name in ids:
        for kind in ("origin", "label"):
            path = source / kind / f"{name}.ply"
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((f"{kind}/{name}.ply", path))
    device = torch.device(args.device)
    model = TSGCNet(in_channels=9, output_channels=2, k=args.k).to(device).eval()
    state = extract_model_state(torch.load(args.checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    (output / "scores").mkdir(parents=True)
    rows = []
    with torch.inference_mode():
        for index, name in enumerate(ids):
            _xyz, colours, faces = read_ply_mesh(source / "label" / f"{name}.ply")
            _origin_xyz, _rgb, origin_faces = read_ply_mesh(
                source / "origin" / f"{name}.ply"
            )
            if not np.array_equal(faces, origin_faces) or not 0 < len(faces) <= 16000:
                raise ValueError(f"Invalid face count or mismatched topology: {name}")
            target = np.all(np.min(colours[faces], axis=1) == 0, axis=1)
            indices, points, labels, _onehot, _name, _raw = dataset[index]
            if not np.array_equal(indices[: len(faces)], faces):
                raise ValueError(f"Dataset changed original face ordering: {name}")
            np.testing.assert_array_equal(labels[: len(faces), 0].astype(bool), target)
            features = torch.as_tensor(points).float().T.unsqueeze(0).to(device)
            log_scores = model(features, np.expand_dims(indices, axis=0))
            scores = log_scores[0, : len(faces), 1].exp().cpu().numpy()
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Invalid class-1 probabilities: {name}")
            np.save(output / "scores" / f"{name}.npy", scores)
            rows.append({"mesh_id": name, **equal_face_metrics(target, scores)})
            print(f"[{index + 1}/{len(ids)}] completed", flush=True)
    summary = cluster_bootstrap(rows, args.bootstrap_reps, args.seed)
    with (output / "per_mesh_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "status": "completed",
        "method": "TSGCNet frozen checkpoint",
        "protocol": "original equal-triangle, strict >0.5, mesh-macro",
        "score": "exp(log_probability[..., 1]) = P(plaque)",
        "padding": "exclude rows beyond original PLY face count",
        "mesh_count": len(ids),
        "participant_count": len({x[:4] for x in ids}),
        "bootstrap_unit": "participant carrying all meshes",
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "k": args.k,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_files": hash_named_files(files),
        "source_sha256": {
            name: sha256_file(root / name)
            for name in (
                "dentalpsam/branch3d/reporting.py",
                "dentalpsam/branch3d/model.py",
                "dentalpsam/branch3d/data.py",
                "dentalpsam/branch3d/utils.py",
                "dentalpsam/evaluation.py",
            )
        },
        "score_sha256": {
            name: sha256_file(output / "scores" / f"{name}.npy") for name in ids
        },
    }
    for file_name, value in (
        ("summary.json", summary),
        ("evaluation_manifest.json", manifest),
    ):
        (output / file_name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)
