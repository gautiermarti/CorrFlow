import torch

from corrflow.models import MLPMatrixVectorField


def test_mlp_vector_field_returns_hollow_symmetric_matrix() -> None:
    model = MLPMatrixVectorField(matrix_dim=6, hidden_dim=16, depth=2, condition_dim=4)
    matrix = torch.randn(3, 6, 6)
    output = model(matrix, torch.rand(3), torch.randn(3, 4))
    assert output.shape == matrix.shape
    assert torch.equal(output, output.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(output, dim1=-2, dim2=-1)) == 0
