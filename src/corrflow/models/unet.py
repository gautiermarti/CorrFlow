"""Compact conditional image U-Net for the raw-matrix DDPM baseline."""

from __future__ import annotations

import torch
from torch import nn

from corrflow.models.equivariant import FourierTimeEmbedding


class ConditionedResBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, context_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_dim)
        self.conv1 = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.context = nn.Linear(context_dim, 2 * output_dim)
        self.norm2 = nn.GroupNorm(8, output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.skip = (
            nn.Conv2d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(values)))
        scale, shift = self.context(context).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(torch.nn.functional.silu(hidden))
        return hidden + self.skip(values)


class MatrixUNetDenoiser(nn.Module):
    """Order-dependent DDPM noise predictor operating on matrices as images."""

    def __init__(
        self,
        *,
        base_dim: int = 32,
        condition_dim: int = 3,
        time_frequencies: int = 16,
    ) -> None:
        super().__init__()
        context_dim = 4 * base_dim
        self.condition_dim = condition_dim
        self.time = FourierTimeEmbedding(context_dim, time_frequencies)
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(2 * context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        self.input = nn.Conv2d(1, base_dim, 3, padding=1)
        self.block0 = ConditionedResBlock(base_dim, base_dim, context_dim)
        self.down1 = nn.Conv2d(base_dim, 2 * base_dim, 4, stride=2, padding=1)
        self.block1 = ConditionedResBlock(2 * base_dim, 2 * base_dim, context_dim)
        self.down2 = nn.Conv2d(2 * base_dim, 4 * base_dim, 4, stride=2, padding=1)
        self.block2 = ConditionedResBlock(4 * base_dim, 4 * base_dim, context_dim)
        self.down3 = nn.Conv2d(4 * base_dim, 8 * base_dim, 4, stride=2, padding=1)
        self.middle = ConditionedResBlock(8 * base_dim, 8 * base_dim, context_dim)

        self.up2 = nn.ConvTranspose2d(
            8 * base_dim, 4 * base_dim, 4, stride=2, padding=1
        )
        self.up_block2 = ConditionedResBlock(8 * base_dim, 4 * base_dim, context_dim)
        self.up1 = nn.ConvTranspose2d(
            4 * base_dim, 2 * base_dim, 4, stride=2, padding=1
        )
        self.up_block1 = ConditionedResBlock(4 * base_dim, 2 * base_dim, context_dim)
        self.up0 = nn.ConvTranspose2d(2 * base_dim, base_dim, 4, stride=2, padding=1)
        self.up_block0 = ConditionedResBlock(2 * base_dim, base_dim, context_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, 1, 3, padding=1),
        )

    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition.unsqueeze(-1)
        context = self.context(
            torch.cat([self.time(time), self.condition(condition)], dim=-1)
        )
        hidden0 = self.block0(self.input(matrix.unsqueeze(1)), context)
        hidden1 = self.block1(self.down1(hidden0), context)
        hidden2 = self.block2(self.down2(hidden1), context)
        hidden3 = self.middle(self.down3(hidden2), context)
        hidden = self.up2(hidden3)
        hidden = self.up_block2(torch.cat([hidden, hidden2], dim=1), context)
        hidden = self.up1(hidden)
        hidden = self.up_block1(torch.cat([hidden, hidden1], dim=1), context)
        hidden = self.up0(hidden)
        hidden = self.up_block0(torch.cat([hidden, hidden0], dim=1), context)
        return self.output(hidden).squeeze(1)
