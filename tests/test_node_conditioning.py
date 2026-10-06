import pytest
import torch

from corrflow.models.node_conditioned import NodeConditionedMatrixVectorField


def test_simultaneous_node_and_matrix_permutation():
    torch.manual_seed(14)
    attributes = torch.randn(6, 3)
    model = NodeConditionedMatrixVectorField(
        node_features=attributes, hidden_dim=8, depth=2, condition_dim=2
    ).double()
    x = torch.randn(2, 6, 6, dtype=torch.float64)
    t = torch.rand(2, dtype=torch.float64)
    condition = torch.eye(2, dtype=torch.float64)
    p = torch.tensor([3, 0, 5, 2, 1, 4])
    original = model(x, t, condition)
    permuted = model(x[:, p][:, :, p], t, condition, model.node_features[p])
    torch.testing.assert_close(
        permuted, original[:, p][:, :, p], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(original, original.transpose(-1, -2))
    assert torch.count_nonzero(original.diagonal(dim1=-2, dim2=-1)) == 0
    # Named attributes must actually affect the field: leaving them fixed while
    # relabeling X is not the simultaneous scientific relabeling operation.
    wrong = model(x[:, p][:, :, p], t, condition)
    assert not torch.allclose(wrong, permuted, atol=1e-7, rtol=1e-7)
    original.square().mean().backward()
    assert model.node_input.weight.grad.abs().sum() > 0


def test_attributes_follow_changed_size_and_must_match_feature_dimension():
    model = NodeConditionedMatrixVectorField(
        node_features=torch.eye(5), hidden_dim=8, depth=1, condition_dim=2
    )
    x = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError, match="Node attributes"):
        model(x, torch.tensor([0.5]), torch.zeros(2, 2))
    result = model(x, torch.tensor([0.5]), torch.zeros(2, 2), model.node_features[:3])
    assert result.shape == x.shape
