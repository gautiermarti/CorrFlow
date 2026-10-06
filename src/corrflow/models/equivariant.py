"""Permutation-equivariant vector fields on hollow symmetric matrices."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierTimeEmbedding(nn.Module):
    """Deterministic Fourier features followed by a learned projection."""

    def __init__(self, hidden_dim: int, frequencies: int = 16) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.projection = nn.Sequential(
            nn.Linear(2 * frequencies, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        time = time.reshape(-1, 1)
        frequencies = 2.0 ** torch.arange(
            self.frequencies, dtype=time.dtype, device=time.device
        )
        angles = 2 * math.pi * time * frequencies.reshape(1, -1)
        return self.projection(torch.cat([angles.sin(), angles.cos()], dim=-1))


class EdgeNodeBlock(nn.Module):
    """Update node and symmetric edge states using invariant aggregations."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(5 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        edges: torch.Tensor,
        nodes: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = edges.shape[1]
        degree = max(n - 1, 1)

        edge_mean = (edges * mask).sum(dim=2) / degree
        edge_second = ((edges**2) * mask).sum(dim=2) / degree
        global_state = edge_mean.mean(dim=1, keepdim=True).expand(-1, n, -1)
        node_context = context[:, None, :].expand(-1, n, -1)
        node_update = self.node_mlp(
            torch.cat([edge_mean, edge_second, global_state, node_context], dim=-1)
        )
        nodes = self.node_norm(nodes + node_update)

        node_i = nodes[:, :, None, :].expand(-1, -1, n, -1)
        node_j = nodes[:, None, :, :].expand(-1, n, -1, -1)
        edge_context = context[:, None, None, :].expand(-1, n, n, -1)
        edge_update = self.edge_mlp(
            torch.cat(
                [
                    edges,
                    node_i + node_j,
                    node_i * node_j,
                    torch.abs(node_i - node_j),
                    edge_context,
                ],
                dim=-1,
            )
        )
        edges = self.edge_norm(edges + edge_update)
        edges = 0.5 * (edges + edges.transpose(1, 2))
        edges = edges * mask
        return edges, nodes


class EquivariantMatrixVectorField(nn.Module):
    """A symmetric, permutation-equivariant conditional matrix vector field.

    For any permutation matrix P, the network satisfies

        f(P X P^T, t, y) = P f(X, t, y) P^T

    up to floating-point roundoff. Inputs and outputs are hollow symmetric
    matrices. Conditions are global continuous or categorical feature vectors
    and remain unchanged by asset permutations.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        depth: int = 4,
        condition_dim: int = 3,
        time_frequencies: int = 16,
        coordinate_skip: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.coordinate_skip_enabled = coordinate_skip
        self.edge_input = nn.Linear(1, hidden_dim)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = FourierTimeEmbedding(hidden_dim, time_frequencies)
        self.context_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([EdgeNodeBlock(hidden_dim) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        if coordinate_skip:
            self.coordinate_skip = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self, matrix: torch.Tensor, time: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("matrix must have shape (batch, n, n)")
        batch, n, _ = matrix.shape
        if time.numel() == 1:
            time = time.expand(batch)
        if condition.ndim == 1:
            condition = condition.unsqueeze(-1)
        if condition.shape[0] == 1:
            condition = condition.expand(batch, -1)
        if condition.shape != (batch, self.condition_dim):
            raise ValueError(f"condition must have shape (batch, {self.condition_dim})")

        matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
        mask = (1 - torch.eye(n, dtype=matrix.dtype, device=matrix.device))[
            None, :, :, None
        ]
        edges = self.edge_input(matrix.unsqueeze(-1)) * mask
        nodes = torch.zeros(
            batch, n, self.hidden_dim, dtype=edges.dtype, device=edges.device
        )
        context = self.context_mlp(
            torch.cat(
                [self.time_embedding(time), self.condition_embedding(condition)],
                dim=-1,
            )
        )

        for block in self.blocks:
            edges, nodes = block(edges, nodes, context, mask)

        output = self.output(edges).squeeze(-1)
        if self.coordinate_skip_enabled:
            output = output + self.coordinate_skip(context)[:, None] * matrix
        output = 0.5 * (output + output.transpose(-1, -2))
        return output * mask.squeeze(-1)
