import json
from pathlib import Path
import subprocess
import sys


def test_public_synthetic_demo(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "demo"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/synthetic_sample/run.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    contract = json.loads((output / "contracts.json").read_text())
    assert contract["image_batch_shape"] == [22, 256, 256, 3]
    assert contract["mesh_batch_shape"] == [22, 6000, 10]
    assert contract["model_image_shape"] == [3, 256, 256]
    metrics = json.loads((output / "evaluation/summary.json").read_text())
    assert metrics["fusion"]["plaque_iou"]["mean"] == 1
