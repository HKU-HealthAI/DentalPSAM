"""Expose the original standalone scientific regressions through pytest."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "mesh_targets",
        "data",
        "probability_png",
        "sota_inputs",
        "prediction_io",
        "preflight_split",
        "evaluate_mesh",
        "original_evaluation",
    ],
)
def test_original_regression(name):
    command = [sys.executable, str(ROOT / "tests" / f"test_{name}.py")]
    if name == "prediction_io":
        command.append("--require_numpy")
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
