"""Check public argument routing without training a model."""

from pathlib import Path

import pytest


from dentalpsam.cli import train_main, test_main as run_test_command


@pytest.mark.parametrize("stage", ["3d", "dentalpsam"])
def test_training_stage_routes_to_library(monkeypatch, tmp_path, stage):
    from dentalpsam import workflows

    received = []
    monkeypatch.setattr(workflows, "train_model", received.append)
    command = [
        "--stage",
        stage,
        "--data",
        str(tmp_path / "data"),
        "--output",
        str(tmp_path / "run"),
    ]
    if stage == "dentalpsam":
        command += ["--sam-checkpoint", "sam.pth"]
    train_main(command)
    args = received[0]
    assert args.train_dir == tmp_path / "data/train"
    assert args.epochs == 50 and args.seed == 42 and args.learning_rate == 1e-4
    assert args.weight_decay == (1e-5 if stage == "3d" else 0)


def test_testing_has_no_threshold_tuning_option():
    with pytest.raises(SystemExit):
        run_test_command(
            [
                "--data",
                "data",
                "--checkpoint",
                "weights.pth",
                "--output",
                "out",
                "--threshold",
                "0.6",
            ]
        )


def test_weight_directory_resolves_named_files(monkeypatch):
    from dentalpsam import workflows

    received = []
    monkeypatch.setattr(workflows, "test_model", received.append)
    run_test_command(
        ["--data", "data/test", "--weights", "weights", "--output", "results"]
    )
    args = received[0]
    assert args.checkpoint == Path("weights/dentalpsam.pth")
    assert args.sam_checkpoint == Path("weights/sam.pth")
    assert args.branch_checkpoint is None  # Required only if inputs need preparation.


def test_joint_training_minimal_command(monkeypatch):
    from dentalpsam import workflows

    received = []
    monkeypatch.setattr(workflows, "train_model", received.append)
    train_main(
        [
            "--stage",
            "dentalpsam",
            "--data",
            "data",
            "--branch-checkpoint",
            "outputs/3d/best.pth",
            "--output",
            "outputs/model",
        ]
    )
    assert received[0].sam_checkpoint == Path("checkpoints/sam.pth")
    assert received[0].branch_checkpoint == Path("outputs/3d/best.pth")
