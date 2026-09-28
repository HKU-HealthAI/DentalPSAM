"""Public workflows regenerate inputs and keep train/test responsibilities separate."""

from argparse import Namespace
import json

import pytest
import torch


def weights(root):
    root.mkdir()
    paths = [root / name for name in ("dentalpsam.pth", "branch3d.pth", "sam.pth")]
    for path in paths:
        path.write_bytes(b"test sentinel; model loading is mocked")
    return paths


def test_missing_weights_stop_before_preprocessing(tmp_path, monkeypatch):
    from dentalpsam import workflows

    monkeypatch.setattr(workflows, "prepare_split", lambda *a: pytest.fail("unexpected preprocessing"))
    args = Namespace(data=tmp_path / "data", checkpoint=tmp_path / "missing.pth",
                     branch_checkpoint=None, sam_checkpoint=tmp_path / "sam.pth",
                     mesh_list=None, output=tmp_path / "results")
    with pytest.raises(FileNotFoundError, match="DentalPSAM checkpoint"):
        workflows.test_model(args)
    assert not args.output.exists()


@pytest.mark.parametrize("wrong_membership", [False, True])
def test_testing_uses_fresh_features_without_a_manifest(tmp_path, monkeypatch, wrong_membership):
    from dentalpsam import workflows, inference, reporting

    model, branch, sam = weights(tmp_path / "weights")
    data = tmp_path / "data"
    for name in ("meshes", "labels", "processed/mesh_features"):
        (data / name).mkdir(parents=True)
    poison = data / "processed/mesh_features/000101.npz"
    poison.write_bytes(b"unknown cache must never be opened")
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
        mesh = "000201" if wrong_membership else "000101"
        for folder in ("SOTA_mesh", "label_mesh"):
            (manual / folder / f"{mesh}.npz").write_bytes(b"fresh features")
        return derived

    def predict(args):
        assert not wrong_membership
        assert args.data_dir.is_relative_to(tmp_path / "results")
        assert (args.data_dir / "SOTA_mesh/000101.npz").read_bytes() == b"fresh features"

    def evaluate(args):
        args.output_dir.mkdir()
        value = {key: {"mean": .5, "ci_low": .4, "ci_high": .6}
                 for key in ("plaque_iou", "plaque_dice", "nonplaque_iou", "nonplaque_dice", "accuracy")}
        (args.output_dir / "summary.json").write_text(json.dumps({"fusion": value}))

    monkeypatch.setattr(workflows, "prepare_split", prepare)
    monkeypatch.setattr(inference, "predict_split", predict)
    monkeypatch.setattr(reporting, "evaluate_predictions", evaluate)
    args = Namespace(data=data, checkpoint=model, branch_checkpoint=branch,
        sam_checkpoint=sam, mesh_list=None, expected_count=1,
        output=tmp_path / "results", device="cpu", num_workers=0, bootstrap_reps=10, seed=42)
    if wrong_membership:
        with pytest.raises(ValueError, match="membership"):
            workflows.test_model(args)
    else:
        workflows.test_model(args)
        assert (args.output / "summary.txt").is_file()
    assert called == [branch]
    assert poison.read_bytes() == b"unknown cache must never be opened"


def test_training_requires_the_branch_checkpoint(tmp_path):
    from dentalpsam.workflows import train_model
    with pytest.raises(ValueError, match="requires --branch-checkpoint"):
        train_model(Namespace(stage="dentalpsam", train_dir=tmp_path / "train",
                              val_dir=tmp_path / "val", save_dir=tmp_path / "out",
                              branch_checkpoint=None))


def test_training_prepares_both_splits_then_saves_checkpoint(tmp_path, monkeypatch):
    from dentalpsam import workflows, training
    from dentalpsam.cli import train_main

    data = tmp_path / "data"
    for split, mesh in (("train", "000101"), ("val", "000201")):
        for kind in ("meshes", "labels"):
            (data / split / kind).mkdir(parents=True)
            (data / split / kind / f"{mesh}.ply").write_bytes(b"layout-only sentinel")
    branch, sam = tmp_path / "branch.pth", tmp_path / "sam.pth"
    branch.write_bytes(b"branch sentinel")
    sam.write_bytes(b"SAM sentinel")
    prepared = []

    def prepare(source, mesh_list, checkpoint, output, device):
        prepared.append(mesh_list.read_text().splitlines())
        derived = output / "dataset/manual_2D"
        for folder in ("SOTA_mesh", "label_mesh"):
            (derived / folder).mkdir(parents=True)
            for mesh in prepared[-1]:
                (derived / folder / f"{mesh}.npz").write_bytes(b"fresh inputs")
        return derived.parent

    class TinyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setattr(workflows, "prepare_split", prepare)
    monkeypatch.setattr(training, "make_loader", lambda *args: [])
    monkeypatch.setattr(training, "build_sam_vit_b", lambda *args: TinyModel())
    monkeypatch.setattr(training, "DentalPSAM", TinyModel)

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
    assert prepared == [["000101"], ["000201"]]
    payload = torch.load(output / "best_model.pth", map_location="cpu")
    assert payload["mesh_fusion"] == "gated"
    assert payload["validation_metrics"]["mesh_loss"] == .25
    assert not (output / "manifest.json").exists()
