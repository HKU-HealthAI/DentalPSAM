"""Pin numerical sources; separately reviewed corrections retain prior hashes.

The fixture records each deliberate source-compatibility correction and its
behavioral tests. A structural cleanup alone must never change these hashes.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = json.loads((ROOT / "tests/fixtures/refactor_contract.json").read_text())[
    "checks"
]


@pytest.mark.parametrize(
    "record", CHECKS, ids=lambda r: f"{r['module']}:{r['name'] or 'module'}"
)
def test_numerical_source_preserved(record):
    node = ast.parse((ROOT / record["module"]).read_text())
    if record["name"]:
        node = next(n for n in node.body if getattr(n, "name", None) == record["name"])
        node = ast.parse(ast.unparse(node)).body[0]
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
    # Python 3.12 adds empty type_params to AST nodes; it is absent in 3.9.
    # Ignore that representational difference, not executable source changes.
    for child in ast.walk(node):
        child._fields = tuple(key for key in child._fields if key != "type_params")
        # Normalize only the reviewed import relocations to the pinned source.
        # Function bodies, class names and all numerical expressions stay pinned.
        if isinstance(child, ast.ImportFrom) and child.module:
            child.module = child.module.replace("dentalpsam.branch3d", "tsgcnet")
            if child.module == "dentalpsam.mesh_io":
                child.module = "tools.evaluate_mesh"
    digest = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert (
        digest == record["ast_sha256"]
    ), "Scientific source changed; review separately from structural cleanup"
