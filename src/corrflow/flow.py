"""Minimal rectified-flow training and deterministic ODE sampling."""

from __future__ import annotations

import torch
from torch import nn


def hollow_symmetric_noise(
    batch: int,
    matrix_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw iid N(0,1) strict-lower coordinates and reflect them exactly."""
    noise = torch.zeros(batch, matrix_dim, matrix_dim, dtype=dtype, device=device)
    indices = torch.tril_indices(matrix_dim, matrix_dim, offset=-1, device=device)
    values = torch.randn(
        batch,
        indices.shape[1],
        dtype=dtype,
        device=device,
        generator=generator,
    )
    noise[:, indices[0], indices[1]] = values
    return noise + noise.transpose(-1, -2)


def rectified_flow_loss(
    vector_field: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unconditional-pairing flow-matching objective in matrix coordinates."""
    batch = target.shape[0]
    if noise is None:
        noise = hollow_symmetric_noise(
            batch,
            target.shape[-1],
            dtype=target.dtype,
            device=target.device,
        )
    if time is None:
        time = torch.rand(batch, dtype=target.dtype, device=target.device)
    view = (batch,) + (1,) * (target.ndim - 1)
    interpolation = (1 - time.view(view)) * noise + time.view(view) * target
    velocity = target - noise
    prediction = vector_field(interpolation, time, condition)
    return torch.mean((prediction - velocity) ** 2)


@torch.no_grad()
def sample_flow(
    vector_field: nn.Module,
    condition: torch.Tensor,
    *,
    matrix_dim: int,
    steps: int = 50,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample with Heun's second-order method from t=0 to t=1."""
    batch = condition.shape[0]
    state = hollow_symmetric_noise(
        batch,
        matrix_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    step = 1.0 / steps

    for index in range(steps):
        time = torch.full((batch,), index * step, dtype=dtype, device=device)
        first = vector_field(state, time, condition)
        proposal = state + step * first
        second_time = torch.clamp(time + step, max=1.0)
        second = vector_field(proposal, second_time, condition)
        state = state + 0.5 * step * (first + second)

    state = 0.5 * (state + state.transpose(-1, -2))
    return state - torch.diag_embed(torch.diagonal(state, dim1=-2, dim2=-1))
