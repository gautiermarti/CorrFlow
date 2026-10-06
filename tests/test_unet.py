import torch

from corrflow.models import MatrixUNetDenoiser


def test_matrix_unet_preserves_shape() -> None:
    model = MatrixUNetDenoiser(base_dim=16, condition_dim=4, time_frequencies=4)
    matrix = torch.randn(2, 40, 40)
    output = model(matrix, torch.rand(2), torch.randn(2, 4))
    assert output.shape == matrix.shape
