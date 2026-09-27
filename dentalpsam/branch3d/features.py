"""Export a frozen TSGCNet checkpoint to DentalPSAM SOTA-mesh patches.

This is a safe replacement for the historical ``pred_to_2d_new.py`` path.
It requires a real checkpoint, writes outside the raw-data tree, keeps model
probabilities and ground truth in separate files, and verifies that the three
UV views form an exact partition of the original PLY faces.
"""


from __future__ import annotations


import argparse


import hashlib


import json


from pathlib import Path


import cv2


import numpy as np


import torch


from plyfile import PlyData


VIEWS = ("up", "in", "out")


VIEW_SHAPES = {"up": (512, 768), "in": (256, 2048), "out": (256, 2048)}


VIEW_PATCH_COUNTS = {"up": 6, "in": 8, "out": 8}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mesh_ids(path: Path) -> list[str]:
    values = [
        Path(line.strip()).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Mesh list must be non-empty and contain no duplicates")
    return values


def checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location=device)
    if isinstance(value, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in value and isinstance(value[key], dict):
                return value[key]
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    return value


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
        centre = np.mean(uv_pixels[triangle], axis=0)
        if not np.all(np.isfinite(centre)):
            raise ValueError(f"{view}: non-finite UV centre")
        if view == "up":
            column = int(np.clip(centre[0], 0, 767) // 256)
            patch_row = int(np.clip(centre[1], 0, 511) // 256)
            patch = column + 3 * patch_row
        else:
            patch = int(np.clip(centre[0], 0, 2047) // 256)
        if patch < 0 or patch >= count:
            raise ValueError(f"{view}: invalid patch index {patch}")
        patches[patch].append(row)
        orders[patch].append(order)
    patch_arrays = [
        np.asarray(part, dtype=np.float64).reshape(-1, 10) if part else np.empty((0, 10), dtype=np.float64)
        for part in patches
    ]
    order_arrays = [np.asarray(part, dtype=np.int64) for part in orders]
    return object_array(patch_arrays), object_array(order_arrays)


def rasterize(view: str, triangles: np.ndarray, uv_pixels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    height, width = VIEW_SHAPES[view]
    image = np.zeros((height, width), dtype=np.uint8)
    for triangle, score in zip(triangles, scores):
        polygon = np.rint(uv_pixels[triangle]).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(image, [polygon.reshape(-1, 1, 2)], int(round(float(score) * 255.0)))
    return image


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
    from dentalpsam.branch3d.model import TSGCNet

    ids = read_mesh_ids(args.mesh_list)
    wanted = {f"{mesh_id}.ply" for mesh_id in ids}
    dataset = PlyDataset(str(args.data_dir / "label"), enable_augmentation=False)
    dataset.file_list = sorted(name for name in dataset.file_list if name in wanted)
    if set(dataset.file_list) != wanted:
        raise ValueError("Requested mesh list does not match the TSGCNet dataset")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TSGCNet(in_channels=9, output_channels=2, k=args.k).to(device)
    model.load_state_dict(checkpoint_state(args.checkpoint, device))
    model.eval()

    for directory in ("SOTA_mesh", "label_mesh", "SOTA_pred", "scores"):
        (args.out_dir / directory).mkdir(parents=True, exist_ok=(directory != "SOTA_mesh"))
    source_root = Path(__file__).resolve().parent
    model_source = source_root / "model.py"
    data_source = source_root / "data.py"
    input_files = [args.checkpoint, args.mesh_list, model_source, data_source]
    output_files: list[Path] = []
    cases: list[dict[str, object]] = []

    with torch.inference_mode():
        for item in range(len(dataset)):
            index_face, points, legacy_target, _onehot, name, _raw_points = dataset[item]
            mesh_id = Path(name).stem
            origin_path = args.data_dir / "origin" / name
            label_path = args.data_dir / "label" / name
            info_path = args.info_dir / f"{mesh_id}.npz"
            input_files.extend((origin_path, label_path, info_path))
            vertices, faces, _origin_colours = read_ply(origin_path)
            _label_vertices, label_faces, label_colours = read_ply(label_path)
            if not np.array_equal(faces, label_faces) or label_colours is None:
                raise ValueError(f"Invalid label topology or colours: {mesh_id}")
            raw_count = len(faces)
            if not np.array_equal(index_face[:raw_count], faces):
                raise ValueError(f"TSGCNet face order differs from origin PLY: {mesh_id}")
            target = np.any(np.all(label_colours[faces] == 0, axis=2), axis=1).astype(np.float64)
            if not np.array_equal(legacy_target[:raw_count, 0], target):
                raise ValueError(f"TSGCNet and raw PLY targets differ: {mesh_id}")
            # Classification/evaluation targets stay binary. Mesh BCE must keep
            # the annotation's intermediate grayscale values, as in the source exporter.
            mesh_target = soft_face_targets(label_colours, faces)

            tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1).to(torch.float32).to(device)
            log_probability = model(tensor, np.expand_dims(index_face, axis=0))
            face_scores = log_probability[0, :raw_count, 1].exp().cpu().numpy().astype(np.float64)
            if not np.all(np.isfinite(face_scores)) or np.any((face_scores < 0) | (face_scores > 1)):
                raise ValueError(f"Invalid model probability: {mesh_id}")
            score_path = args.out_dir / "scores" / f"{mesh_id}.npy"
            np.save(score_path, face_scores.astype(np.float32))
            output_files.append(score_path)

            mesh_index = {face_key(face): offset for offset, face in enumerate(faces)}
            if len(mesh_index) != raw_count:
                raise ValueError(f"Duplicate original face connectivity: {mesh_id}")
            prediction_npz: dict[str, np.ndarray] = {}
            target_npz: dict[str, np.ndarray] = {}
            used_indices: list[int] = []
            with np.load(info_path, allow_pickle=True) as info:
                for view in VIEWS:
                    triangles = np.asarray(info[f"tri_{view}"], dtype=np.int64)
                    uv_pixels = np.asarray(info[f"uvpx_{view}"], dtype=np.float64)
                    indices = np.asarray([mesh_index[face_key(face)] for face in triangles], dtype=np.int64)
                    used_indices.extend(indices.tolist())
                    view_scores = face_scores[indices]
                    view_target = mesh_target[indices]
                    prediction_npz[view], prediction_npz[f"face_order_{view}"] = patch_rows(
                        view, triangles, uv_pixels, vertices, view_scores,
                    )
                    target_npz[view], target_npz[f"face_order_{view}"] = patch_rows(
                        view, triangles, uv_pixels, vertices, view_target,
                    )
                    image_path = args.out_dir / "SOTA_pred" / f"{mesh_id}_{VIEWS.index(view)}.png"
                    if not cv2.imwrite(str(image_path), rasterize(view, triangles, uv_pixels, view_scores)):
                        raise OSError(f"Failed to write {image_path}")
                    output_files.append(image_path)
            if sorted(used_indices) != list(range(raw_count)):
                raise ValueError(f"UV views are not an exact face partition: {mesh_id}")

            sota_path = args.out_dir / "SOTA_mesh" / f"{mesh_id}.npz"
            label_mesh_path = args.out_dir / "label_mesh" / f"{mesh_id}.npz"
            np.savez_compressed(sota_path, **prediction_npz)
            np.savez_compressed(label_mesh_path, **target_npz)
            output_files.extend((sota_path, label_mesh_path))
            cases.append({
                "mesh_id": mesh_id,
                "raw_face_count": raw_count,
                "plaque_fraction": float(target.mean()),
                "continuous_target_face_count": int(np.sum((mesh_target > 0) & (mesh_target < 1))),
                "predicted_probability_mean": float(face_scores.mean()),
            })
            print(f"{mesh_id}: {raw_count} verified faces", flush=True)

    manifest = {
        "schema_version": 1,
        "method": "TSGCNet frozen-checkpoint DentalPSAM SOTA-mesh export",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "mesh_list": str(args.mesh_list),
        "mesh_list_sha256": sha256(args.mesh_list),
        "mesh_count": len(ids),
        "patient_count": len({mesh_id[:4] for mesh_id in ids}),
        "model_probability_location": "SOTA_mesh patch row column 9",
        "ground_truth_location": "label_mesh patch row column 9",
        "ground_truth_encoding": "1 - min(vertex annotation RGB / 255, per channel)[0]; continuous BCE target",
        "source_png_gt_leakage_path_used": False,
        "raw_data_modified": False,
        "uv_info_dir": str(args.info_dir),
        "source": {
            "exporter_sha256": sha256(Path(__file__).resolve()),
            "model_sha256": sha256(model_source),
            "dataloader_sha256": sha256(data_source),
        },
        "input_file_set_sha256": hashlib.sha256(
            "\n".join(f"{path}:{sha256(path)}" for path in input_files).encode("utf-8")
        ).hexdigest(),
        "output_file_set_sha256": hashlib.sha256(
            "\n".join(f"{path.name}:{sha256(path)}" for path in output_files).encode("utf-8")
        ).hexdigest(),
        "cases": cases,
    }
    (args.out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
