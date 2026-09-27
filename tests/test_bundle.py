"""Reject valid-looking but scientifically mismatched weight/cache combinations."""

from argparse import Namespace
import json

import pytest
import torch

from dentalpsam.bundle import verify_bundle, write_training_bundle_manifest
from dentalpsam.mesh_io import sha256_file


def weights(root, variant="gated"):
    root.mkdir()
    model, branch, sam = [root / name for name in ("dentalpsam.pth", "branch3d.pth", "sam.pth")]
    branch.write_bytes(b"unit-test branch sentinel")
    sam.write_bytes(b"unit-test SAM sentinel")
    state = {"mesh_decoder.mlp.weight": torch.ones(1)}
    if variant == "concat":
        state["mesh_decoder.mlp_fusion.layers.0.weight"] = torch.ones(1)
    binding = {"branch3d_sha256": sha256_file(branch), "sam_sha256": sha256_file(sam),
               "mesh_fusion": variant}
    torch.save({"model_state_dict": state, "input_provenance": binding}, model)
    manifest = root / "manifest.json"
    write_training_bundle_manifest(manifest, model, branch, sam)
    return manifest, model, branch, sam


@pytest.mark.parametrize("variant", ["gated", "concat"])
def test_matching_bundle_roundtrip(tmp_path, variant):
    paths = weights(tmp_path / "weights", variant)
    assert verify_bundle(*paths)["mesh_fusion"] == variant


@pytest.mark.parametrize("index", [1, 2, 3])
def test_substituting_any_weight_is_rejected_before_loading(tmp_path, index):
    paths = weights(tmp_path / "weights")
    paths[index].write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Bundle mismatch"):
        verify_bundle(*paths)


def test_rehashing_wrong_branch_does_not_bypass_embedded_training_binding(tmp_path):
    manifest, model, branch, sam = weights(tmp_path / "weights")
    branch.write_bytes(b"another run")
    declared = json.loads(manifest.read_text())
    declared["branch3d_sha256"] = sha256_file(branch)
    manifest.write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="training binding"):
        verify_bundle(manifest, model, branch, sam)


def test_declared_variant_must_match_checkpoint(tmp_path):
    paths = weights(tmp_path / "weights", "concat")
    declared = json.loads(paths[0].read_text())
    declared["mesh_fusion"] = "gated"
    paths[0].write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="architecture"):
        verify_bundle(*paths)


def test_cannot_automatically_certify_unbound_historical_weights(tmp_path):
    paths = weights(tmp_path / "weights")
    value = torch.load(paths[1], map_location="cpu")
    torch.save(value["model_state_dict"], paths[1])
    with pytest.raises(ValueError, match="author verification"):
        write_training_bundle_manifest(tmp_path / "new.json", *paths[1:])
    declared = json.loads(paths[0].read_text())
    declared["dentalpsam_sha256"] = sha256_file(paths[1])
    paths[0].write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="lacks the training binding"):
        verify_bundle(*paths)


def test_bad_bundle_stops_before_preprocessing_and_output_creation(tmp_path, monkeypatch):
    from dentalpsam import workflows

    def forbidden(*args, **kwargs):
        pytest.fail("preprocessing must not run for an unverified bundle")

    monkeypatch.setattr(workflows, "prepare_split", forbidden)
    args = Namespace(data=tmp_path / "data", checkpoint=tmp_path / "dentalpsam.pth",
                     branch_checkpoint=tmp_path / "branch3d.pth", sam_checkpoint=tmp_path / "sam.pth",
                     bundle_manifest=tmp_path / "missing.json", output=tmp_path / "results")
    with pytest.raises(FileNotFoundError, match="manifest missing"):
        workflows.test_model(args)
    assert not args.output.exists()


@pytest.mark.parametrize("wrong_export", [False, True])
def test_existing_unknown_cache_is_never_used_by_public_test(tmp_path, monkeypatch, wrong_export):
    from dentalpsam import workflows, inference, reporting

    manifest, model, branch, sam = weights(tmp_path / "weights")
    data = tmp_path / "data"
    for name in ("meshes", "labels", "processed/mesh_features"):
        (data / name).mkdir(parents=True)
    poison = data / "processed/mesh_features/000101.npz"
    poison.write_bytes(b"unknown cached data must never be opened")
    (data / "mesh_ids.txt").write_text("000101\n")
    called = []

    def prepare(source, mesh_list, checkpoint, output, device):
        assert checkpoint == branch
        assert not (source / "manual_2D").exists()
        called.append(checkpoint)
        derived = output / "dataset"
        manual = derived / "manual_2D"
        for folder in ("origin", "label", "SOTA_mesh", "label_mesh", "info"):
            (manual / folder).mkdir(parents=True)
        for view in range(3):
            for folder in ("origin", "label"):
                (manual / folder / f"000101_{view}.png").write_bytes(b"test only")
        for folder in ("SOTA_mesh", "label_mesh"):
            (manual / folder / "000101.npz").write_bytes(b"fresh sentinel")
        (output / "features").mkdir()
        (output / "features/export_manifest.json").write_text(json.dumps({
            "status": "completed", "checkpoint_sha256": "0" * 64 if wrong_export else sha256_file(branch),
            "mesh_list_sha256": sha256_file(mesh_list), "mesh_count": 1}))
        return derived

    def predict(args):
        assert not wrong_export, "inference must not run for a mismatched export record"
        assert args.data_dir.is_relative_to(tmp_path / "results")
        assert (args.data_dir / "SOTA_mesh/000101.npz").read_bytes() == b"fresh sentinel"

    def evaluate(args):
        args.output_dir.mkdir()
        value = {key: {"mean": .5, "ci_low": .4, "ci_high": .6}
                 for key in ("plaque_iou", "plaque_dice", "nonplaque_iou", "nonplaque_dice", "accuracy")}
        (args.output_dir / "summary.json").write_text(json.dumps({"fusion": value}))

    monkeypatch.setattr(workflows, "prepare_split", prepare)
    monkeypatch.setattr(inference, "predict_split", predict)
    monkeypatch.setattr(reporting, "evaluate_predictions", evaluate)
    args = Namespace(data=data, checkpoint=model, branch_checkpoint=branch,
        sam_checkpoint=sam, bundle_manifest=manifest, mesh_list=None, expected_count=1,
        output=tmp_path / "results", device="cpu", num_workers=0, bootstrap_reps=10, seed=42)
    if wrong_export:
        with pytest.raises(ValueError, match="Generated features do not match"):
            workflows.test_model(args)
    else:
        workflows.test_model(args)
    assert called == [branch]
    assert poison.read_bytes() == b"unknown cached data must never be opened"


def test_training_rejects_unknown_cache_without_branch_checkpoint(tmp_path):
    from dentalpsam.workflows import train_model

    with pytest.raises(ValueError, match="requires --branch-checkpoint"):
        train_model(Namespace(stage="dentalpsam", train_dir=tmp_path / "train",
                              val_dir=tmp_path / "val", save_dir=tmp_path / "out",
                              branch_checkpoint=None))


def test_new_training_saves_binding_and_exports_its_bundle(tmp_path, monkeypatch):
    from dentalpsam import workflows, training
    from dentalpsam.cli import train_main

    data = tmp_path / "data"
    for split, mesh in (("train", "000101"), ("val", "000201")):
        for kind in ("meshes", "labels"):
            (data / split / kind).mkdir(parents=True)
            (data / split / kind / f"{mesh}.ply").write_bytes(b"layout-only sentinel")
    branch, sam = tmp_path / "branch.pth", tmp_path / "sam.pth"
    branch.write_bytes(b"frozen branch sentinel")
    sam.write_bytes(b"SAM sentinel")

    def prepare(source, mesh_list, checkpoint, output, device):
        derived = output / "dataset/manual_2D"
        (derived / "label_mesh").mkdir(parents=True)
        for mesh in mesh_list.read_text().splitlines():
            (derived / "label_mesh" / f"{mesh}.npz").write_bytes(b"fresh labels")
        (output / "features").mkdir()
        (output / "features/export_manifest.json").write_text(json.dumps({
            "status": "completed", "checkpoint_sha256": sha256_file(checkpoint)}))
        return derived.parent

    class TinyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(workflows, "prepare_split", prepare)
    monkeypatch.setattr(training, "make_loader", lambda *args: [])
    monkeypatch.setattr(training, "build_sam_vit_b", lambda *args: TinyModel())
    monkeypatch.setattr(training, "DentalPSAM", TinyModel)
    # Exercise optimizer/checkpoint plumbing with a tiny model, not clinical quality.
    def tiny_epoch(*args):
        model, optimizer = args[0], args[-1]
        if optimizer is not None:
            optimizer.zero_grad()
            model.weight.sum().backward()
            optimizer.step()
        return {"loss": .5, "mesh_loss": .25, "mesh_iou": .1}

    monkeypatch.setattr(training, "run_epoch", tiny_epoch)
    output = tmp_path / "trained"
    train_main(["--stage", "dentalpsam", "--data", str(data), "--output", str(output),
                "--branch-checkpoint", str(branch), "--sam-checkpoint", str(sam),
                "--epochs", "1", "--device", "cpu"])
    payload = torch.load(output / "best_model.pth", map_location="cpu")
    assert payload["input_provenance"]["branch3d_sha256"] == sha256_file(branch)
    assert set(payload["input_provenance"]["prepared_export_manifests"]) == {"train", "val"}
    manifest = verify_bundle(output / "manifest.json", output / "best_model.pth", branch, sam)
    assert manifest["provenance"]["kind"] == "training_checkpoint"
