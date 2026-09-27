"""3D training contracts: class axis, padding, targets and model selection."""

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from dentalpsam.branch3d.training import (
    per_face_nll, read_mesh_target_and_area, validation_metrics,
)


def test_nll_reduces_over_classes_and_ignores_padding():
    logits = torch.tensor([[[1., 2.], [3., -1.], [-100., 100.]]], requires_grad=True)
    target = torch.tensor([1, 0])
    weights = torch.tensor([1., 2.])
    log_probability = logits.log_softmax(dim=-1)
    loss = per_face_nll(log_probability, target, weights)
    # Independent reference: per-face cross entropy on [faces, classes].
    reference = torch.nn.functional.cross_entropy(logits[0, :2], target, weight=weights)
    torch.testing.assert_close(loss, reference)
    loss.backward()
    torch.testing.assert_close(logits.grad[0, 2], torch.zeros(2))
    assert logits.grad[0, :2].abs().sum() > 0
    with pytest.raises(ValueError, match="shape"):
        per_face_nll(log_probability.transpose(1, 2), target, weights)
    with pytest.raises(ValueError, match="unpadded"):
        per_face_nll(log_probability, torch.empty(0, dtype=torch.long), weights)


def test_nonfinite_loss_fails_before_optimizer_step():
    with pytest.raises(FloatingPointError, match="Non-finite"):
        per_face_nll(torch.full((1, 3, 2), float("nan")), torch.tensor([1]), torch.ones(2))


def test_rgb_targets_follow_per_channel_minimum(tmp_path):
    vertices = np.zeros(3, dtype=[(key, "f4") for key in ("x", "y", "z")]
                        + [(key, "u1") for key in ("red", "green", "blue")])
    vertices["x"] = [0, 1, 0]
    vertices["y"] = [0, 0, 1]
    # No vertex is black; nevertheless every channel has a zero on this face.
    vertices["red"] = [0, 255, 255]
    vertices["green"] = [255, 0, 255]
    vertices["blue"] = [255, 255, 0]
    faces = np.array([([0, 1, 2],)], dtype=[("vertex_indices", "i4", (3,))])
    path = tmp_path / "mesh.ply"
    PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")]).write(path)
    target, area = read_mesh_target_and_area(path, path)
    np.testing.assert_array_equal(target, [1])
    np.testing.assert_allclose(area, [.5])


class FixedScores(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = iter(scores)

    def forward(self, points, indices):
        score = torch.tensor(next(self.scores), dtype=torch.float32)
        return torch.stack((1 - score, score), dim=-1).log().unsqueeze(0)


def batch(name, target):
    # One repeated padding face deliberately has an incorrect prediction.
    target = torch.tensor([*target, target[-1]], dtype=torch.int64)[None, :, None]
    count = target.shape[1]
    return (torch.zeros(1, count, 3, dtype=torch.int64), torch.zeros(1, count, 33),
            target, None, [name], None)


def test_validation_uses_equal_faces_mesh_macro_and_strict_threshold():
    names = ["000101.ply", "000102.ply", "000201.ply"]
    targets = [[1, 1], [1, 0], [1, 1]]
    metadata = {name: (np.array(target), np.array([1., 1000.]))
                for name, target in zip(names, targets)}
    model = FixedScores([[.9, .1, .99], [.5, .1, .99], [.9, .9, .01]])
    result = validation_metrics(model, [batch(n, t) for n, t in zip(names, targets)],
                                metadata, torch.device("cpu"))
    # Mesh IoUs are 1/2, 0, 1, independently of area or participant mesh count.
    assert result["mesh_macro_plaque_iou"] == pytest.approx(.5)
    assert result["mesh_macro_plaque_dice"] == pytest.approx((2/3 + 0 + 1)/3)


def test_empty_validation_is_rejected():
    with pytest.raises(ValueError, match="no meshes"):
        validation_metrics(FixedScores([]), [], {}, torch.device("cpu"))


def test_validation_rejects_disagreement_with_loader():
    with pytest.raises(ValueError, match="target differs"):
        validation_metrics(FixedScores([[.9, .1, .9]]), [batch("000101.ply", [1, 0])],
                           {"000101.ply": (np.array([0, 0]), np.ones(2))}, torch.device("cpu"))
