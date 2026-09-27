"""Test variant inference and strict loading without downloading model weights."""

from unittest.mock import Mock

import pytest
import torch

from dentalpsam import checkpoints


@pytest.mark.parametrize("variant", ["gated", "concat"])
def test_variant_and_strict_loading(monkeypatch, tmp_path, variant):
    state = {"mesh_decoder.mlp.weight": torch.ones(1)}
    if variant == "concat":
        state["mesh_decoder.mlp_fusion.layers.0.weight"] = torch.ones(1)
    path = tmp_path / "weights.pth"
    torch.save({"model_state_dict": state}, path)
    sam = Mock()
    sam.to.return_value = sam
    model = Mock()
    model.to.return_value = model
    constructor = Mock(return_value=model)
    monkeypatch.setattr(checkpoints, "build_sam_vit_b", Mock(return_value=sam))
    monkeypatch.setattr(checkpoints, "DentalPSAM", constructor)
    assert checkpoints.load_dentalpsam("unused.pth", path, torch.device("cpu")) is model
    constructor.assert_called_once_with(sam, mesh_fusion=variant)
    assert model.load_state_dict.call_args.kwargs == {"strict": True}
    model.eval.assert_called_once()


def test_invalid_checkpoint_container():
    with pytest.raises(TypeError):
        checkpoints.extract_model_state([1, 2])
