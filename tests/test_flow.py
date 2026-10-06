import torch
from torch import nn

from corrflow.flow import hollow_symmetric_noise, sample_flow


def test_hollow_symmetric_noise_has_unit_coordinate_variance() -> None:
    noise = hollow_symmetric_noise(4000, 5, dtype=torch.float32, device="cpu")
    assert torch.equal(noise, noise.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(noise, dim1=-2, dim2=-1)) == 0
    indices = torch.tril_indices(5, 5, offset=-1)
    lower = noise[:, indices[0], indices[1]]
    assert abs(float(lower.var()) - 1) < 0.05


class _ZeroField(nn.Module):
    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(matrix)


def test_sample_flow_uses_condition_batch_not_feature_count() -> None:
    condition = torch.zeros(7, 4)
    samples = sample_flow(_ZeroField(), condition, matrix_dim=5, steps=2)
    assert samples.shape == (7, 5, 5)
