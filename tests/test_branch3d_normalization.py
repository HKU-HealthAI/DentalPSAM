"""Checkpoint-aware per-mesh BN must agree across training and inference."""

import pytest
import torch
from torch import nn

from dentalpsam.branch3d.checkpoints import checkpoint_normalization, load_branch3d
from dentalpsam.branch3d.model import TSGCNet


def model(mode="per_mesh"):
    torch.manual_seed(42)
    return TSGCNet(in_channels=9, output_channels=2, k=12, normalization=mode)


def test_current_mesh_moments_in_both_modes_without_buffer_updates():
    network = model()
    layer = network.bn1_c
    features = torch.randn(1, 64, 5, 12)
    buffers = {name: value.clone() for name, value in network.named_buffers()}
    expected = nn.functional.batch_norm(features, None, None, layer.weight,
                                       layer.bias, training=True, eps=layer.eps)
    training = layer(features)
    network.eval()
    assert not network.training and not network.dp1.training
    assert all(m.training and not m.track_running_stats for m in network.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm))
    torch.testing.assert_close(layer(features), expected, rtol=0, atol=0)
    torch.testing.assert_close(training, expected, rtol=0, atol=0)
    # A different mesh must not affect the next mesh's normalization.
    layer(features + 100)
    torch.testing.assert_close(layer(features), expected, rtol=0, atol=0)
    network.train().eval()
    for name, value in network.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


def test_initial_parameters_keys_and_training_gradients_are_preserved():
    current, historical = model(), model("running")
    assert current.state_dict().keys() == historical.state_dict().keys()
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, historical.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(1, 64, 5, 12)
    outputs = []
    for network in (current, historical):
        result = network.bn1_c(x)
        result.square().mean().backward()
        outputs.append(result)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(current.bn1_c.weight.grad, historical.bn1_c.weight.grad,
                               rtol=0, atol=0)
    historical.eval()
    assert not historical.bn1_c.training and historical.bn1_c.track_running_stats


@pytest.mark.parametrize("mode", ["per_mesh", "running"])
def test_checkpoint_roundtrip_restores_normalization(tmp_path, mode):
    original = model(mode).eval()
    path = tmp_path / "branch.pth"
    torch.save({"model_state_dict": original.state_dict(),
                "configuration": {"normalization": mode}}, path)
    restored = load_branch3d(path, "cpu")
    assert restored.normalization == mode
    x = torch.randn(1, 64, 5, 12)
    torch.testing.assert_close(original.bn1_c(x), restored.bn1_c(x), rtol=0, atol=0)


def test_unmarked_weights_keep_historical_inference(tmp_path):
    original = model("running").eval()
    path = tmp_path / "older.pth"
    torch.save(original.state_dict(), path)
    restored = load_branch3d(path, "cpu")
    assert restored.normalization == "running"
    assert not restored.bn1_c.training
    state = original.state_dict()
    del state["bn1_c.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_branch3d(path, "cpu")


def test_selected_training_checkpoint_is_recognized():
    arguments = {"coordinate_normalization": "legacy", "bn_statistics": "per_mesh",
                 "experiment_policy_version": "coordinate-bn-factorial-v2"}
    checkpoint = {"configuration": {"arguments": arguments}}
    assert checkpoint_normalization(checkpoint) == "per_mesh"
    arguments["coordinate_normalization"] = "xyz_standard"
    with pytest.raises(ValueError, match="coordinate"):
        checkpoint_normalization(checkpoint)


@pytest.mark.parametrize("configuration", [
    {"normalization": "unknown"},
    {"arguments": {"bn_statistics": "per_mesh"}},
    {"normalization": "running", "arguments": {
        "experiment_policy_version": "coordinate-bn-factorial-v2",
        "bn_statistics": "per_mesh"}},
])
def test_unknown_or_conflicting_contract_fails(configuration):
    with pytest.raises(ValueError, match="metadata"):
        checkpoint_normalization({"configuration": configuration})
