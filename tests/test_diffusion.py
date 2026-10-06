import torch
from torch import nn

from corrflow.diffusion import (
    cosine_schedule,
    ddpm_loss,
    equivariant_ddpm_loss,
    sample_ddim,
    sample_equivariant_ddim,
)
from corrflow.models import EquivariantMatrixVectorField


class _ZeroDenoiser(nn.Module):
    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(matrix)


class _FixedDenoiser(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        return self.output


def test_diffusion_schedule_and_shapes() -> None:
    cumulative = torch.cumprod(1 - cosine_schedule(100), dim=0)
    assert cumulative.shape == (100,)
    assert torch.all(cumulative[1:] < cumulative[:-1])
    target = torch.randn(3, 8, 8)
    condition = torch.randn(3, 4)
    assert ddpm_loss(_ZeroDenoiser(), target, condition, cumulative) > 0
    sampled = sample_ddim(_ZeroDenoiser(), condition, cumulative, matrix_dim=8, steps=5)
    assert sampled.shape == target.shape


def test_equivariant_diffusion_stays_hollow_and_symmetric() -> None:
    torch.manual_seed(9)
    model = EquivariantMatrixVectorField(
        hidden_dim=8, depth=1, condition_dim=3, time_frequencies=2
    )
    target = torch.randn(2, 5, 5)
    target = 0.5 * (target + target.transpose(-1, -2))
    target = target - torch.diag_embed(torch.diagonal(target, dim1=-2, dim2=-1))
    condition = torch.eye(3)[:2]
    cumulative = torch.cumprod(1 - cosine_schedule(12), dim=0)

    loss = equivariant_ddpm_loss(model, target, condition, cumulative)
    v_loss = equivariant_ddpm_loss(
        model, target, condition, cumulative, prediction_type="v"
    )
    sample = sample_equivariant_ddim(
        model, condition, cumulative, matrix_dim=5, steps=4
    )
    v_sample = sample_equivariant_ddim(
        model,
        condition,
        cumulative,
        matrix_dim=5,
        steps=4,
        prediction_type="v",
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(v_loss)
    torch.testing.assert_close(sample, sample.transpose(-1, -2))
    torch.testing.assert_close(
        torch.diagonal(sample, dim1=-2, dim2=-1), torch.zeros(2, 5)
    )
    torch.testing.assert_close(v_sample, v_sample.transpose(-1, -2))
    torch.testing.assert_close(
        torch.diagonal(v_sample, dim1=-2, dim2=-1), torch.zeros(2, 5)
    )


def test_velocity_prediction_target_is_exact() -> None:
    torch.manual_seed(11)
    target = torch.randn(2, 4, 4)
    target = 0.5 * (target + target.transpose(-1, -2))
    noise = torch.randn_like(target)
    noise = 0.5 * (noise + noise.transpose(-1, -2))
    condition = torch.zeros(2, 1)
    cumulative = torch.cumprod(1 - cosine_schedule(10), dim=0)
    timestep = torch.tensor([2, 7])
    alpha = cumulative[timestep][:, None, None]
    velocity = torch.sqrt(alpha) * noise - torch.sqrt(1 - alpha) * target

    loss = equivariant_ddpm_loss(
        _FixedDenoiser(velocity),
        target,
        condition,
        cumulative,
        prediction_type="v",
        noise=noise,
        timestep=timestep,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
