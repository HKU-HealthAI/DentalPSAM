#!/usr/bin/env python3
"""Compare packaged models and original evaluation with trusted server source.

This is a one-mesh compatibility check, not a cohort performance evaluation.
Only run with trusted native Python source and trusted checkpoint files.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
from importlib.machinery import SourceFileLoader
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dentalpsam.checkpoints import extract_model_state, load_dentalpsam
from dentalpsam.data import SAMDataset, load_and_patchify_png_permesh
from dentalpsam.evaluation import equal_face_metrics, load_mesh_predictions
from tools.evaluate_mesh import sha256_file


def import_native(name, path):
    spec = importlib.util.spec_from_loader(name, SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-sam-dir", "native-tsgcnet-dir", "native-evaluator",
                 "sam-checkpoint", "dentalpsam-checkpoint", "tsgcnet-checkpoint",
                 "data-dir", "prediction-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--mesh-id", required=True)
    parser.add_argument("--native-model-file", type=Path,
                        help="Optional pinned historical model source, including .backup files")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for source in (args.data_dir, args.prediction_dir, args.native_sam_dir, args.native_tsgcnet_dir):
        if source.resolve() in args.output.resolve().parents:
            raise ValueError("Write compatibility evidence outside input/source directories")
    device = torch.device(args.device)
    report = {"scope": "one-mesh inference and evaluator compatibility; not paper reproduction",
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda}

    # Compare actual input tensors, not only checkpoint key names.
    sys.path.insert(0, str(args.native_sam_dir.parent))
    sys.path.insert(0, str(args.native_sam_dir))
    os.environ["SAMORAL_GPU"] = str(device.index or 0)
    native_data = import_native(f"{args.native_sam_dir.name}.dataloader", args.native_sam_dir / "dataloader.py")
    manual = args.data_dir / "manual_2D"
    clean_data = load_and_patchify_png_permesh(str(manual), args.mesh_id)
    old_data = native_data.load_and_patchify_png_permesh(str(manual), args.mesh_id)
    np.testing.assert_array_equal(clean_data[0], old_data[0])
    np.testing.assert_array_equal(clean_data[3], old_data[3])
    report["dentalpsam_inputs_exact"] = True
    sample = SAMDataset(clean_data[0], clean_data[1], clean_data[3], clean_data[4])[0]
    image = sample["pixel_values"].unsqueeze(0).to(device)
    mesh = torch.as_tensor(sample["SOTA_mesh"]).float().unsqueeze(0).to(device)
    model = load_dentalpsam(args.sam_checkpoint, args.dentalpsam_checkpoint, device)
    report["dentalpsam_mesh_fusion"] = model.mesh_decoder.fusion
    with torch.no_grad():
        output = model(image, mesh)
        expected = {key: output[key].cpu() for key in ("pred_masks", "pred_mesh")}
    del model, output
    gc.collect()
    torch.cuda.empty_cache()
    native_model_file = args.native_model_file or args.native_sam_dir / "model.py"
    native_model = import_native("native_sam_model", native_model_file)
    report["native_model_sha256"] = sha256_file(native_model_file)
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](image_size=1024, num_classes=1, checkpoint=str(args.sam_checkpoint)).to(device)
    model = native_model.DentalPSAM(sam).to(device).eval()
    model.load_state_dict(extract_model_state(torch.load(args.dentalpsam_checkpoint, map_location="cpu")), strict=True)
    with torch.no_grad():
        output = model(image, mesh)
        report["dentalpsam_max_abs_diff"] = {}
        for key in expected:
            actual = output[key].cpu()
            torch.testing.assert_close(actual, expected[key], rtol=1e-5, atol=1e-6)
            report["dentalpsam_max_abs_diff"][key] = float((actual - expected[key]).abs().max())
    del model, sam, output, image, mesh, clean_data, old_data
    gc.collect()
    torch.cuda.empty_cache()

    # TSGCNet preserves the legacy 16,000-face normalization contract.
    sys.path.insert(0, str(args.native_tsgcnet_dir))
    old_dataset_module = import_native("native_tsgc_data", args.native_tsgcnet_dir / "dataloader.py")
    from tsgcnet.data import PlyDataset
    from tsgcnet.model import TSGCNet
    datasets = [PlyDataset(str(args.data_dir / "label"), enable_augmentation=False),
                old_dataset_module.plydataset(str(args.data_dir / "label"), enable_augmentation=False)]
    batches = []
    for dataset in datasets:
        dataset.file_list = [args.mesh_id + ".ply"]
        batches.append(dataset[0])
    for index in (0, 1, 2, 3, 5):
        np.testing.assert_array_equal(batches[0][index], batches[1][index])
    report["tsgcnet_inputs_exact"] = True
    # Native adjacency construction is CPU/NumPy; only features move to GPU.
    faces = torch.as_tensor(batches[0][0]).unsqueeze(0)
    features = torch.as_tensor(batches[0][1]).float().T.unsqueeze(0).to(device)
    state = extract_model_state(torch.load(args.tsgcnet_checkpoint, map_location="cpu"))
    model = TSGCNet(in_channels=9, output_channels=2, k=12).to(device).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        expected = model(features, faces).cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    native_tsgc = import_native("native_tsgc_model", args.native_tsgcnet_dir / "TSGCNet.py")
    model = native_tsgc.TSGCNet(in_channels=9, output_channels=2, k=12).to(device).eval()
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = model(features, faces).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    report["tsgcnet_max_abs_diff"] = float((actual - expected).abs().max())

    # Native fixed fusion uses quantized PNG scores and strict > 0.5.
    native_eval = import_native("native_evaluation", args.native_evaluator)
    target, image, mesh = load_mesh_predictions(args.data_dir, args.prediction_dir, args.mesh_id)
    clean = equal_face_metrics(target, 0.5 * image + 0.5 * mesh)
    with contextlib.redirect_stdout(io.StringIO()):
        legacy = native_eval.process_single_mesh_with_strategies(
            args.mesh_id, str(args.prediction_dir / "3Dpred"), str(args.prediction_dir),
            str(manual / "info"), str(args.data_dir / "label"))
    if legacy is None:
        raise RuntimeError("Native evaluation did not return a result")
    np.testing.assert_equal(clean["plaque_iou"], legacy["fixed"]["iou"])
    np.testing.assert_equal(clean["plaque_dice"], legacy["fixed"]["dice"])
    report["native_fixed_fusion_exact"] = True
    report["checkpoint_sha256"] = {name: sha256_file(getattr(args, name)) for name in
                                   ("sam_checkpoint", "dentalpsam_checkpoint", "tsgcnet_checkpoint")}
    report["source_sha256"] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
        for directory in ("dentalpsam", "tsgcnet", "tools")
        for path in sorted((Path(__file__).resolve().parents[1] / directory).rglob("*.py"))
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
