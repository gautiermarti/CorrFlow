"""Order-dependent MLP vector field used as an equivariance ablation."""

from __future__ import annotations

import torch
from corrbench.geometry import lower_to_symmetric, symmetric_to_lower
from torch import nn

from corrflow.models.equivariant import FourierTimeEmbedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class MLPMatrixVectorField(nn.Module):
    """Dense GFT flow that deliberately does not enforce permutation equivariance."""

    def __init__(
        self,
        *,
        matrix_dim: int = 80,
        hidden_dim: int = 512,
        depth: int = 4,
        condition_dim: int = 3,
        time_frequencies: int = 16,
    ) -> None:
        super().__init__()
        self.matrix_dim = matrix_dim
        self.condition_dim = condition_dim
        vector_dim = matrix_dim * (matrix_dim - 1) // 2
        self.time_embedding = FourierTimeEmbedding(hidden_dim, time_frequencies)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input = nn.Linear(vector_dim + 2 * hidden_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim) for _ in range(depth)]
        )
        self.output = nn.Linear(hidden_dim, vector_dim)
        self.coordinate_skip = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        batch = matrix.shape[0]
        if time.numel() == 1:
            time = time.expand(batch)
        if condition.ndim == 1:
            condition = condition.unsqueeze(-1)
        vector = symmetric_to_lower(matrix)
        context = torch.cat(
            [self.time_embedding(time), self.condition_embedding(condition)], dim=-1
        )
        hidden = self.input(torch.cat([vector, context], dim=-1))
        output = self.output(self.blocks(hidden))
        output = output + self.coordinate_skip(context) * vector
        return lower_to_symmetric(output)
