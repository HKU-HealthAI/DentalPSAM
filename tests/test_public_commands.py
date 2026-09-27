"""Check public argument routing without training a model."""
import importlib.util
from pathlib import Path

import pytest


def entry(name):
    path = Path(__file__).resolve().parents[1] / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"public_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stage", ["3d", "dentalpsam"])
def test_training_stage_routes_to_library(monkeypatch, tmp_path, stage):
    from dentalpsam import training as joint
    from tsgcnet import training as branch

    received = []
    monkeypatch.setattr(branch if stage == "3d" else joint, "run_training", received.append)
    command = ["--stage", stage, "--data", str(tmp_path / "data"), "--output", str(tmp_path / "run")]
    if stage == "dentalpsam":
        command += ["--sam-checkpoint", "sam.pth"]
    entry("train").main(command)
    args = received[0]
    suffix = "train" if stage == "3d" else "train/manual_2D"
    assert args.train_dir == tmp_path / "data" / suffix
    assert args.epochs == 50 and args.seed == 42 and args.learning_rate == 1e-4
    assert args.weight_decay == (1e-5 if stage == "3d" else 0)


def test_testing_has_no_threshold_tuning_option():
    with pytest.raises(SystemExit):
        entry("test").main(["--data", "data", "--checkpoint", "weights.pth",
                            "--output", "out", "--threshold", "0.6"])
