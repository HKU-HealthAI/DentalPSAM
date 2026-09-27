"""Directory adapters must preserve files and reject ambiguous input layouts."""

from pathlib import Path
import pytest

from dentalpsam._layout import PROCESSED_NAMES, SplitLayout


def make_layout(root, public, prepared):
    for folder in ("meshes", "labels") if public else ("origin", "label"):
        (root / folder).mkdir(parents=True)
    if prepared:
        parent = root / ("processed" if public else "manual_2D")
        for old, new in PROCESSED_NAMES.items():
            (parent / (new if public else old)).mkdir(parents=True)
    return root


@pytest.mark.parametrize("public", [True, False])
@pytest.mark.parametrize("prepared", [True, False])
def test_adapter_keeps_input_files_and_order(tmp_path, public, prepared):
    root = make_layout(tmp_path / "input", public, prepared)
    layout = SplitLayout.read(root)
    source = layout.meshes / "unit_fixture.ply"
    source.write_bytes(b"identity-only unit-test sentinel")
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    adapted = layout.legacy_view(tmp_path / "view")
    assert (adapted / "origin" / source.name).samefile(source)
    assert source.read_bytes() == b"identity-only unit-test sentinel"
    assert layout.prepared() == prepared
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before
    if prepared:
        for old, new in PROCESSED_NAMES.items():
            assert (adapted / "manual_2D" / old).samefile(
                layout.processed / (new if public else old)
            )


def test_mixed_layout_is_rejected(tmp_path):
    root = make_layout(tmp_path / "input", True, False)
    (root / "origin").mkdir()
    with pytest.raises(ValueError, match="Mixed"):
        SplitLayout.read(root)


def test_partial_preparation_is_not_silently_ignored(tmp_path):
    root = make_layout(tmp_path / "input", True, False)
    (root / "processed/images").mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match="Incomplete"):
        SplitLayout.read(root).prepared()


def test_adapter_cannot_write_inside_input(tmp_path):
    root = make_layout(tmp_path / "input", True, True)
    with pytest.raises(ValueError, match="outside"):
        SplitLayout.read(root).legacy_view(root / "view")
