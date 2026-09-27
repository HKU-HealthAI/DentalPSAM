#!/usr/bin/env python3
"""Synthetic functionality demo — NOT a scientific reproduction dataset.

Creates 22 artificial triangles, prepared view/patch files, and explicitly
fabricated probabilities. No patient data, weights, GPU, or network is used.
It demonstrates contracts and reporting, not neural inference or model quality.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from dentalpsam.data import SAMDataset, load_and_patchify_png_permesh
from dentalpsam.reporting import evaluate_predictions
from tools.preflight_split import inspect_split


def write_mesh(path, vertices, faces, colors):
    vertex_data = np.empty(
        len(vertices),
        dtype=[(key, "f4") for key in "xyz"]
        + [(key, "u1") for key in ("red", "green", "blue")],
    )
    for column, key in enumerate("xyz"):
        vertex_data[key] = vertices[:, column]
    for column, key in enumerate(("red", "green", "blue")):
        vertex_data[key] = colors[:, column]
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ],
        text=False,
    ).write(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    data, predictions = output / "data", output / "predictions"
    manual = data / "manual_2D"
    for directory in [
        data / "origin",
        data / "label",
        predictions / "3Dpred",
        *[
            manual / name
            for name in ("origin", "label", "info", "SOTA_mesh", "label_mesh")
        ],
    ]:
        directory.mkdir(parents=True)
    mesh_id = "000001"  # Artificial identifier, not a participant from the study.
    faces = np.arange(66, dtype=np.int32).reshape(22, 3)
    vertices = np.array(
        [
            point
            for i in range(22)
            for point in ((2 * i, 0, 0), (2 * i + 1, 0, 0), (2 * i, 1, 0))
        ],
        dtype=np.float32,
    )
    targets = np.arange(22) % 2 == 0
    colors = np.full((66, 3), 160, dtype=np.uint8)
    labels = np.repeat(np.where(targets, 0, 255), 3)
    write_mesh(data / f"origin/{mesh_id}.ply", vertices, faces, colors)
    write_mesh(
        data / f"label/{mesh_id}.ply",
        vertices,
        faces,
        np.repeat(labels[:, None], 3, axis=1),
    )
    info, features, annotations = {}, {}, {}
    offset = 0
    for index, (view, rows, columns) in enumerate(
        (("up", 2, 3), ("in", 1, 8), ("out", 1, 8))
    ):
        count = rows * columns
        rgb = np.full((rows * 256, columns * 256, 3), 160, dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mock = np.zeros(rgb.shape[:2], dtype=np.uint8)
        uv = np.zeros((66, 2), dtype=np.float32)
        parts, label_parts, probabilities = [], [], []
        for patch in range(count):
            face_index = offset + patch
            row, column = divmod(patch, columns)
            score = 0.8 if targets[face_index] else 0.2
            region = np.s_[
                row * 256 : (row + 1) * 256, column * 256 : (column + 1) * 256
            ]
            mask[region] = 255 if targets[face_index] else 0
            mock[region] = round(score * 255)
            uv[faces[face_index]] = [
                (column * 256 + 64, row * 256 + 64),
                (column * 256 + 96, row * 256 + 64),
                (column * 256 + 64, row * 256 + 96),
            ]
            geometry = vertices[faces[face_index]].reshape(-1)
            parts.append(np.r_[geometry, score].reshape(1, 10))
            label_parts.append(
                np.r_[geometry, float(targets[face_index])].reshape(1, 10)
            )
            probabilities.append(score)
        for collection, values in ((features, parts), (annotations, label_parts)):
            array = np.empty(count, dtype=object)
            for patch, value in enumerate(values):
                array[patch] = value
            collection[view] = array
            collection[f"face_order_{view}"] = np.arange(count)
        info[f"tri_{view}"] = faces[offset : offset + count]
        info[f"uvpx_{view}"] = uv
        for path, value in (
            (manual / f"origin/{mesh_id}_{index}.png", rgb),
            (manual / f"label/{mesh_id}_{index}.png", mask),
            (predictions / f"{mesh_id}_{index}.png", mock),
        ):
            if not cv2.imwrite(str(path), value):
                raise OSError(path)
        np.savez(
            predictions / f"3Dpred/{mesh_id}_{index}.npz",
            concatenated_array=probabilities,
        )
        offset += count
    np.savez(manual / f"info/{mesh_id}.npz", **info)
    np.savez(manual / f"SOTA_mesh/{mesh_id}.npz", **features)
    np.savez(manual / f"label_mesh/{mesh_id}.npz", **annotations)
    mesh_list = data / "mesh_ids.txt"
    mesh_list.write_text(mesh_id + "\n")
    contract = inspect_split(manual, participant_id_prefix_length=4)
    images, masks, _indices, mesh, labels = load_and_patchify_png_permesh(
        manual, mesh_id
    )
    sample = SAMDataset(images, masks, mesh, labels)[0]
    contract.update(
        scope=__doc__.splitlines()[0],
        image_batch_shape=list(images.shape),
        mesh_batch_shape=list(mesh.shape),
        model_image_shape=list(sample["pixel_values"].shape),
        probabilities="fabricated fixture; no model inference",
    )
    (output / "contracts.json").write_text(json.dumps(contract, indent=2) + "\n")
    evaluate_predictions(
        argparse.Namespace(
            data_dir=data,
            prediction_dir=predictions,
            mesh_list=mesh_list,
            output_dir=output / "evaluation",
            expected_count=1,
            fusion_weight_2d=0.5,
            threshold=0.5,
            bootstrap_reps=200,
            seed=42,
        )
    )
    print(f"Synthetic software check passed. NOT a paper result. Outputs: {output}")


if __name__ == "__main__":
    main()
