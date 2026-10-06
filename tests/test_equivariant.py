import torch

from corrflow.models.equivariant import EquivariantMatrixVectorField


def test_vector_field_is_symmetric_hollow_and_permutation_equivariant() -> None:
    torch.manual_seed(17)
    model = EquivariantMatrixVectorField(
        hidden_dim=16,
        depth=2,
        condition_dim=3,
        time_frequencies=4,
        coordinate_skip=True,
    ).double()
    matrix = torch.randn(3, 8, 8, dtype=torch.float64)
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    matrix = matrix - torch.diag_embed(torch.diagonal(matrix, dim1=-2, dim2=-1))
    time = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    condition = torch.eye(3, dtype=torch.float64)
    permutation = torch.tensor([5, 2, 0, 7, 1, 6, 4, 3])

    output = model(matrix, time, condition)
    permuted_output = model(matrix[:, permutation][:, :, permutation], time, condition)
    expected = output[:, permutation][:, :, permutation]

    assert torch.allclose(output, output.transpose(-1, -2), atol=1e-12)
    assert torch.count_nonzero(torch.diagonal(output, dim1=-2, dim2=-1)) == 0
    assert torch.allclose(permuted_output, expected, atol=1e-10, rtol=1e-10)
