"""DDPM training and deterministic DDIM sampling for the image baseline."""

from __future__ import annotations

import math

import torch
from torch import nn

from corrflow.flow import hollow_symmetric_noise


def cosine_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    """Cosine cumulative-noise schedule from improved DDPM."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    cumulative = (
        torch.cos(((steps / timesteps + offset) / (1 + offset)) * math.pi / 2) ** 2
    )
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).float()


def ddpm_loss(
    denoiser: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard epsilon-prediction denoising objective."""
    batch = target.shape[0]
    if noise is None:
        noise = torch.randn_like(target)
    if timestep is None:
        timestep = torch.randint(
            cumulative_alphas.numel(), (batch,), device=target.device
        )
    alpha = cumulative_alphas[timestep][:, None, None]
    noisy = torch.sqrt(alpha) * target + torch.sqrt(1 - alpha) * noise
    normalized_time = timestep / (cumulative_alphas.numel() - 1)
    prediction = denoiser(noisy, normalized_time, condition)
    return torch.mean((prediction - noise) ** 2)


def equivariant_ddpm_loss(
    denoiser: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    *,
    prediction_type: str = "epsilon",
    noise: torch.Tensor | None = None,
    timestep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffusion prediction loss on hollow-symmetric GFT coordinates."""
    batch = target.shape[0]
    if noise is None:
        noise = hollow_symmetric_noise(
            batch,
            target.shape[-1],
            dtype=target.dtype,
            device=target.device,
        )
    if timestep is None:
        timestep = torch.randint(
            cumulative_alphas.numel(), (batch,), device=target.device
        )
    alpha = cumulative_alphas[timestep][:, None, None]
    noisy = torch.sqrt(alpha) * target + torch.sqrt(1 - alpha) * noise
    normalized_time = timestep / max(cumulative_alphas.numel() - 1, 1)
    prediction = denoiser(noisy, normalized_time, condition)
    if prediction_type == "epsilon":
        objective = noise
    elif prediction_type == "v":
        objective = torch.sqrt(alpha) * noise - torch.sqrt(1 - alpha) * target
    else:
        raise ValueError(f"Unsupported diffusion prediction type: {prediction_type}")
    return torch.mean((prediction - objective) ** 2)


@torch.no_grad()
def sample_ddim(
    denoiser: nn.Module,
    condition: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    *,
    matrix_dim: int,
    steps: int = 50,
    clip_clean: float | None = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM sampler (eta=0) on a subsampled training schedule."""
    device = condition.device
    batch = condition.shape[0]
    state = torch.randn(
        batch,
        matrix_dim,
        matrix_dim,
        device=device,
        generator=generator,
    )
    indices = (
        torch.linspace(cumulative_alphas.numel() - 1, 0, steps, device=device)
        .round()
        .long()
    )
    for position, timestep in enumerate(indices):
        alpha = cumulative_alphas[timestep]
        time = torch.full(
            (batch,),
            float(timestep) / (cumulative_alphas.numel() - 1),
            device=device,
        )
        predicted_noise = denoiser(state, time, condition)
        predicted_clean = (
            state - torch.sqrt(1 - alpha) * predicted_noise
        ) / torch.sqrt(alpha)
        if clip_clean is not None:
            predicted_clean = predicted_clean.clamp(-clip_clean, clip_clean)
        if position == len(indices) - 1:
            state = predicted_clean
        else:
            previous_alpha = cumulative_alphas[indices[position + 1]]
            state = (
                torch.sqrt(previous_alpha) * predicted_clean
                + torch.sqrt(1 - previous_alpha) * predicted_noise
            )
    return state


@torch.no_grad()
def sample_equivariant_ddim(
    denoiser: nn.Module,
    condition: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    *,
    matrix_dim: int,
    steps: int = 50,
    prediction_type: str = "epsilon",
    clip_clean: float | None = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Deterministic DDIM sampling in hollow-symmetric GFT matrix coordinates."""
    if steps < 2:
        raise ValueError("steps must be at least two.")
    device = condition.device
    batch = condition.shape[0]
    state = hollow_symmetric_noise(
        batch,
        matrix_dim,
        dtype=condition.dtype,
        device=device,
        generator=generator,
    )
    indices = (
        torch.linspace(cumulative_alphas.numel() - 1, 0, steps, device=device)
        .round()
        .long()
    )
    for position, timestep in enumerate(indices):
        alpha = cumulative_alphas[timestep]
        time = torch.full(
            (batch,),
            float(timestep) / max(cumulative_alphas.numel() - 1, 1),
            dtype=state.dtype,
            device=device,
        )
        prediction = denoiser(state, time, condition)
        if prediction_type == "epsilon":
            predicted_noise = prediction
            predicted_clean = (
                state - torch.sqrt(1 - alpha) * predicted_noise
            ) / torch.sqrt(alpha)
        elif prediction_type == "v":
            predicted_clean = (
                torch.sqrt(alpha) * state - torch.sqrt(1 - alpha) * prediction
            )
            predicted_noise = (
                torch.sqrt(1 - alpha) * state + torch.sqrt(alpha) * prediction
            )
        else:
            raise ValueError(
                f"Unsupported diffusion prediction type: {prediction_type}"
            )
        if clip_clean is not None:
            predicted_clean = predicted_clean.clamp(-clip_clean, clip_clean)
        if position == len(indices) - 1:
            state = predicted_clean
        else:
            previous_alpha = cumulative_alphas[indices[position + 1]]
            state = (
                torch.sqrt(previous_alpha) * predicted_clean
                + torch.sqrt(1 - previous_alpha) * predicted_noise
            )
    state = 0.5 * (state + state.transpose(-1, -2))
    return state - torch.diag_embed(torch.diagonal(state, dim1=-2, dim2=-1))
