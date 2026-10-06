"""Experimental CorrFlow field conditioned on attributes attached to matrix nodes."""

from __future__ import annotations

import torch
from torch import nn

from corrflow.models.equivariant import EquivariantMatrixVectorField


class NodeConditionedMatrixVectorField(EquivariantMatrixVectorField):
    """Equivariant under simultaneous relabeling of matrices and node attributes.

    F(P X P^T, P A, t, y) = P F(X, A, t, y) P^T. Holding named ROI attributes
    fixed intentionally allows a nonexchangeable distribution on a fixed atlas.
    This is a research extension, separate from the categorical financial model.
    """

    def __init__(self, *, node_features: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        attributes = torch.as_tensor(node_features, dtype=torch.float32)
        if attributes.ndim != 2 or min(attributes.shape) < 1:
            raise ValueError("Expected nonempty nodes-by-features attributes")
        if not bool(torch.isfinite(attributes).all()):
            raise ValueError("Nonfinite node attributes")
        self.register_buffer("node_features", attributes.clone())
        self.node_input = nn.Linear(attributes.shape[1], self.hidden_dim)

    def forward(self, matrix, time, condition, node_features=None):
        if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("Expected batched square matrices")
        batch, n, _ = matrix.shape
        attributes = self.node_features if node_features is None else node_features
        if attributes.ndim == 2:
            attributes = attributes[None].expand(batch, -1, -1)
        if attributes.shape != (batch, n, self.node_input.in_features):
            raise ValueError("Node attributes do not match matrix nodes")
        if time.numel() == 1:
            time = time.expand(batch)
        if condition.ndim == 1:
            condition = condition.unsqueeze(-1)
        if condition.shape[0] == 1:
            condition = condition.expand(batch, -1)
        if condition.shape != (batch, self.condition_dim):
            raise ValueError("Condition dimensions differ")
        matrix = (matrix + matrix.transpose(-1, -2)) / 2
        mask = (1 - torch.eye(n, dtype=matrix.dtype, device=matrix.device))[
            None, :, :, None
        ]
        edges = self.edge_input(matrix.unsqueeze(-1)) * mask
        nodes = self.node_input(attributes)
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
        output = (output + output.transpose(-1, -2)) / 2
        return output * mask.squeeze(-1)
