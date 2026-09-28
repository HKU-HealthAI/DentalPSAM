"""Regenerate aligned mesh targets and frozen 3D-branch feature predictions.

This is a safe replacement for the historical ``pred_to_2d_new.py`` path.
It requires a real checkpoint, writes outside the raw-data tree, keeps model
probabilities and ground truth in separate files, and verifies that the three
UV views form an exact partition of the original PLY faces.
"""


from __future__ import annotations


import argparse




import json


from pathlib import Path


import cv2


import numpy as np


import torch


from plyfile import PlyData


VIEWS = ("up", "in", "out")


VIEW_SHAPES = {"up": (512, 768), "in": (256, 2048), "out": (256, 2048)}


VIEW_PATCH_COUNTS = {"up": 6, "in": 8, "out": 8}


def read_mesh_ids(path: Path) -> list[str]:
    values = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Mesh list must be non-empty and contain no duplicates")
    return values


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    value = PlyData.read(str(path))
    vertex = value["vertex"].data
    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float64)
    faces = np.asarray([row[0] for row in value["face"].data], dtype=np.int64)
    colours = None
    if all(name in vertex.dtype.names for name in ("red", "green", "blue")):
        colours = np.stack([vertex[name] for name in ("red", "green", "blue")], axis=1)
    return xyz, faces, colours


def face_key(face: np.ndarray) -> tuple[int, int, int]:
    return tuple(sorted(int(value) for value in face))


def soft_face_targets(colours: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Preserve the continuous targets used by the original mesh-label export.

    Annotation PLY colours include intermediate grayscale values. The original
    exporter reads RGB in [0, 1], takes the per-channel vertex minimum on each
    triangle, inverts it, and retains channel 0. These targets supervise the
    DentalPSAM mesh BCE; they are NOT the binary targets used by the standalone
    3D classifier or the equal-triangle test evaluator.
    """
    colours = np.asarray(colours, dtype=np.float64)
    if colours.ndim != 2 or colours.shape[1] != 3:
        raise ValueError("Expected annotation RGB with shape [vertices, 3]")
    if not np.isfinite(colours).all() or np.any((colours < 0) | (colours > 255)):
        raise ValueError("Annotation RGB must contain finite values in [0, 255]")
    return (1.0 - np.min((colours / 255.0)[faces], axis=1))[:, 0]


def object_array(parts: list[np.ndarray]) -> np.ndarray:
    result = np.empty(len(parts), dtype=object)
    result[:] = parts
    return result


def original_face_scores(
    padded_faces: np.ndarray, probabilities: np.ndarray, original_faces: np.ndarray,
) -> np.ndarray:
    """Apply the source exporter's face-key lookup without adding padded faces.

    A short mesh repeats its final face. The historical dict(zip(...)) keeps
    the last score for that key, which can differ from the first occurrence.
    Preserve this export behavior even though standalone evaluation excludes
    repeated rows directly. Ground-truth targets never use this lookup.
    """
    if len(padded_faces) != len(probabilities):
        raise ValueError("Face indices and probabilities must have equal length")
    lookup = dict(zip(map(tuple, padded_faces), probabilities))
    return np.asarray([lookup[tuple(face)] for face in original_faces], dtype=np.float64)


def patch_rows(
    view: str,
    triangles: np.ndarray,
    uv_pixels: np.ndarray,
    vertices: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = VIEW_PATCH_COUNTS[view]
    patches: list[list[np.ndarray]] = [[] for _ in range(count)]
    orders: list[list[int]] = [[] for _ in range(count)]
    for order, (triangle, score) in enumerate(zip(triangles, scores)):
        row = np.empty(10, dtype=np.float64)
        row[:9] = vertices[triangle].reshape(-1)
        row[9] = score
        centre = np.mean(np.asarray(uv_pixels[triangle], dtype=np.float64), axis=0)
        if not np.all(np.isfinite(centre)):
            raise ValueError(f"{view}: non-finite UV centre")
        height, width = VIEW_SHAPES[view]
        # Compatibility with pred_to_2d_new.process_surface_triangles:
        # clip only the upper bounds. Negative centres use Python's negative
        # list indexing, including the historical factor 3 for every view.
        # Lower-bound clipping changes patch membership on real study meshes.
        column = int(min(centre[0], width - 1) // 256)
        patch_row = int(min(centre[1], height - 1) // 256)
        patch = column + 3 * patch_row
        if patch < -count or patch >= count:
            raise ValueError(f"{view}: invalid patch index {patch}")
        patches[patch].append(row)
        orders[patch].append(order)
    patch_arrays = [
        np.asarray(part, dtype=np.float64)
        for part in patches
    ]
    order_arrays = [np.asarray(part, dtype=np.int64) for part in orders]
    return object_array(patch_arrays), object_array(order_arrays)


def rasterize(view: str, triangles: np.ndarray, uv_pixels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    height, width = VIEW_SHAPES[view]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for triangle, score in zip(triangles, scores):
        # The source exporter truncates both polygon coordinates and intensity.
        # OpenCV clips polygons at the image boundary; do not clip vertices.
        polygon = uv_pixels[triangle].astype(np.int32)
        intensity = int(float(score) * 255.0)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], (intensity,) * 3)
    return image


def verify_patch_file(
    path: Path,
    vertices: np.ndarray,
    triangles_by_view: dict[str, np.ndarray],
    scores_by_view: dict[str, np.ndarray],
) -> None:
    """Reopen an export and check geometry, order, and scores face by face.

    Call separately with annotation-derived targets and model-derived scores.
    Equal prediction/target arrays alone are not an error: a perfect predictor
    could legitimately match the annotation. Each must match its own source.
    """
    with np.load(path, allow_pickle=True) as cache:
        for view in VIEWS:
            triangles = triangles_by_view[view]
            parts, orders = cache[view], cache[f"face_order_{view}"]
            if len(parts) != VIEW_PATCH_COUNTS[view] or len(orders) != len(parts):
                raise ValueError(f"Wrong patch count: {path.name}/{view}")
            rows, indices = [], []
            for part, order in zip(parts, orders):
                part = np.asarray(part)
                order = np.asarray(order)
                if part.shape == (0,):
                    part = part.reshape(0, 10)
                if part.ndim != 2 or part.shape[1] != 10 or len(part) > 6000:
                    raise ValueError(f"Invalid mesh patch shape: {path.name}/{view}")
                if order.ndim != 1 or order.dtype.kind not in "iu" or len(order) != len(part):
                    raise ValueError(f"Invalid face order: {path.name}/{view}")
                rows.append(part)
                indices.append(order)
            rows, indices = np.concatenate(rows), np.concatenate(indices)
            if not np.array_equal(np.sort(indices), np.arange(len(triangles))):
                raise ValueError(f"Incomplete face permutation: {path.name}/{view}")
            if not np.array_equal(rows[:, :9], vertices[triangles[indices]].reshape(-1, 9)):
                raise ValueError(f"Geometry mismatch: {path.name}/{view}")
            if not np.array_equal(rows[:, 9], scores_by_view[view][indices]):
                raise ValueError(f"Score/target mismatch: {path.name}/{view}")


def export_features(args) -> None:
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.info_dir = (
        args.info_dir.expanduser().resolve()
        if args.info_dir is not None
        else args.data_dir / "manual_2D" / "info"
    )
    args.mesh_list = args.mesh_list.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out_dir}")
    if args.out_dir == args.data_dir or args.data_dir in args.out_dir.parents:
        raise ValueError("Derived output must be outside the raw-data tree")
    for path in (args.data_dir, args.info_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not args.checkpoint.is_file() or not args.mesh_list.is_file():
        raise FileNotFoundError("Missing checkpoint or mesh list")

    from dentalpsam.branch3d.data import PlyDataset
    from dentalpsam.branch3d.checkpoints import load_branch3d

    ids = read_mesh_ids(args.mesh_list)
    wanted = {f"{mesh_id}.ply" for mesh_id in ids}
    dataset = PlyDataset(str(args.data_dir / "label"), enable_augmentation=False)
    dataset.file_list = sorted(name for name in dataset.file_list if name in wanted)
    if set(dataset.file_list) != wanted:
        raise ValueError("Requested mesh list does not match the TSGCNet dataset")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = load_branch3d(args.checkpoint, device, k=args.k)

    for directory in ("SOTA_mesh", "label_mesh", "SOTA_pred", "scores"):
        (args.out_dir / directory).mkdir(parents=True, exist_ok=(directory != "SOTA_mesh"))
    cases: list[dict[str, object]] = []

    with torch.inference_mode():
        for item in range(len(dataset)):
            index_face, points, legacy_target, _onehot, name, _raw_points = dataset[item]
            mesh_id = Path(name).stem
            origin_path = args.data_dir / "origin" / name
            label_path = args.data_dir / "label" / name
            info_path = args.info_dir / f"{mesh_id}.npz"
            vertices, faces, _origin_colours = read_ply(origin_path)
            _label_vertices, label_faces, label_colours = read_ply(label_path)
            if not np.array_equal(faces, label_faces) or label_colours is None:
                raise ValueError(f"Invalid label topology or colours: {mesh_id}")
            raw_count = len(faces)
            if not 0 < raw_count <= 16000:
                raise ValueError(f"Expected 1..16000 original faces: {mesh_id}")
            if not np.array_equal(index_face[:raw_count], faces):
                raise ValueError(f"TSGCNet face order differs from origin PLY: {mesh_id}")
            target = np.all(np.min(label_colours[faces], axis=1) == 0, axis=1).astype(np.float64)
            if not np.array_equal(legacy_target[:raw_count, 0], target):
                raise ValueError(f"TSGCNet and raw PLY targets differ: {mesh_id}")
            # Classification/evaluation targets stay binary. Mesh BCE must keep
            # the annotation's intermediate grayscale values, as in the source exporter.
            mesh_target = soft_face_targets(label_colours, faces)

            tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1).contiguous().to(torch.float32).to(device)
            log_probability = model(tensor, np.expand_dims(index_face, axis=0))
            # Match the source exporter's softmax conversion exactly, including
            # floating-point rounding; the model itself returns log-probability.
            probabilities = torch.softmax(log_probability.contiguous().view(-1, 2), dim=1)
            face_scores = original_face_scores(
                index_face, probabilities[:, 1].cpu().numpy(), faces,
            )
            if not np.all(np.isfinite(face_scores)) or np.any((face_scores < 0) | (face_scores > 1)):
                raise ValueError(f"Invalid model probability: {mesh_id}")
            score_path = args.out_dir / "scores" / f"{mesh_id}.npy"
            np.save(score_path, face_scores.astype(np.float32))

            mesh_index = {face_key(face): offset for offset, face in enumerate(faces)}
            if len(mesh_index) != raw_count:
                raise ValueError(f"Duplicate original face connectivity: {mesh_id}")
            prediction_npz: dict[str, np.ndarray] = {}
            target_npz: dict[str, np.ndarray] = {}
            triangles_by_view, scores_by_view, targets_by_view = {}, {}, {}
            used_indices: list[int] = []
            with np.load(info_path, allow_pickle=True) as info:
                for view in VIEWS:
                    triangles = np.asarray(info[f"tri_{view}"], dtype=np.int64)
                    uv_pixels = np.asarray(info[f"uvpx_{view}"], dtype=np.float64)
                    indices = np.asarray([mesh_index[face_key(face)] for face in triangles], dtype=np.int64)
                    used_indices.extend(indices.tolist())
                    view_scores = face_scores[indices]
                    view_target = mesh_target[indices]
                    triangles_by_view[view] = triangles
                    scores_by_view[view] = view_scores
                    targets_by_view[view] = view_target
                    prediction_npz[view], prediction_npz[f"face_order_{view}"] = patch_rows(
                        view, triangles, uv_pixels, vertices, view_scores,
                    )
                    target_npz[view], target_npz[f"face_order_{view}"] = patch_rows(
                        view, triangles, uv_pixels, vertices, view_target,
                    )
                    image_path = args.out_dir / "SOTA_pred" / f"{mesh_id}_{VIEWS.index(view)}.png"
                    if not cv2.imwrite(str(image_path), rasterize(view, triangles, uv_pixels, view_scores)):
                        raise OSError(f"Failed to write {image_path}")
            if sorted(used_indices) != list(range(raw_count)):
                raise ValueError(f"UV views are not an exact face partition: {mesh_id}")

            sota_path = args.out_dir / "SOTA_mesh" / f"{mesh_id}.npz"
            label_mesh_path = args.out_dir / "label_mesh" / f"{mesh_id}.npz"
            np.savez_compressed(sota_path, **prediction_npz)
            np.savez_compressed(label_mesh_path, **target_npz)
            verify_patch_file(sota_path, vertices, triangles_by_view, scores_by_view)
            verify_patch_file(label_mesh_path, vertices, triangles_by_view, targets_by_view)
            cases.append({
                "mesh_id": mesh_id,
                "raw_face_count": raw_count,
                "plaque_fraction": float(target.mean()),
                "continuous_target_face_count": int(np.sum((mesh_target > 0) & (mesh_target < 1))),
                "predicted_probability_mean": float(face_scores.mean()),
            })
            print(f"{mesh_id}: {raw_count} verified faces", flush=True)

    manifest = {
        "schema_version": 2,
        "status": "completed",
        "method": "frozen 3D-branch feature and annotation export",
        "normalization": model.normalization,
        "checkpoint": str(args.checkpoint),
        "mesh_list": str(args.mesh_list),
        "mesh_count": len(ids),
        "patient_count": len({mesh_id[:4] for mesh_id in ids}),
        "model_probability_location": "SOTA_mesh patch row column 9",
        "ground_truth_location": "label_mesh patch row column 9",
        "ground_truth_encoding": "1 - min(vertex annotation RGB / 255, per channel)[0]; continuous BCE target",
        "prediction_encoding": "softmax(model log-probabilities, dim=class)[:, 1]; P(plaque)",
        "patch_assignment": "source upper-only UV clipping with Python negative indexing",
        "padding_score_lookup": "original face tuples; last occurrence wins for repeated padding",
        "verification": "both NPZ files reopened; geometry, permutation and all scores/targets checked",
        "device": str(device),
        "k": args.k,
        "torch_version": torch.__version__,
        "source_png_gt_leakage_path_used": False,
        "raw_data_modified": False,
        "uv_info_dir": str(args.info_dir),
        "cases": cases,
    }
    (args.out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Regenerate both caches for one fixed split without touching source data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="One split with meshes/labels and processed/metadata, or origin/label and manual_2D/info.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen 3D-branch state dictionary.")
    parser.add_argument("--mesh-list", type=Path, help="Fixed mesh list; defaults to DATA/mesh_ids.txt.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory, outside the source data.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=12)
    args = parser.parse_args()
    from dentalpsam._layout import SplitLayout

    layout = SplitLayout.read(args.data)
    source = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output == source or source in output.parents:
        raise ValueError("Derived output must be outside the raw-data tree")
    mesh_list = args.mesh_list or source / "mesh_ids.txt"
    info = layout.processed / ("metadata" if layout.public else "info")
    for path in (mesh_list, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not info.is_dir():
        raise NotADirectoryError(f"Existing UV metadata required: {info}")
    # The model loader retains its checkpoint-sensitive historical layout.
    # For public names, create links in a new sibling directory, not in DATA.
    data = layout.legacy_view(output.with_name(output.name + "_inputs"), include_processed=False)
    export_features(argparse.Namespace(
        checkpoint=args.checkpoint, data_dir=data, info_dir=info,
        mesh_list=mesh_list, out_dir=output, device=args.device, k=args.k,
    ))


if __name__ == "__main__":
    main()
