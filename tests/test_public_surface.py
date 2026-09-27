"""The installed code must not depend on maintainer-only utilities."""

import ast
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_single_public_package_and_no_unused_config():
    for name in ("tsgcnet", "scripts", "tools", "configs"):
        assert not (ROOT / name).is_dir() or not any((ROOT / name).iterdir())
    for path in (ROOT / "dentalpsam").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
            )
            assert not any(
                name.split(".")[0] in {"tools", "dev", "tsgcnet", "scripts"}
                for name in names
            ), path


@pytest.mark.parametrize(
    "args",
    [
        ["train.py", "--help"],
        ["test.py", "--help"],
        ["train.py", "--stage", "3d", "--help"],
        ["train.py", "--stage", "dentalpsam", "--help"],
    ],
)
def test_help_without_data(args):
    result = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "TSGCNet" not in result.stdout and "manual_2D" not in result.stdout
