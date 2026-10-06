"""Geometry/API smoke with an untrained network, not learned scenarios."""

import torch
from corrbench.geometry import gft_to_corr, symmetric_to_lower

from corrflow.flow import sample_flow
from corrflow.models import EquivariantMatrixVectorField

torch.manual_seed(7)
torch.set_num_threads(2)
model = EquivariantMatrixVectorField(hidden_dim=8, depth=1, condition_dim=4).eval()
with torch.no_grad():
    coordinates = sample_flow(model, torch.zeros(3, 4), matrix_dim=8, steps=2)
    matrices, info = gft_to_corr(
        symmetric_to_lower(coordinates).double(),
        tolerance=1e-8,
        max_iterations=2000,
        return_info=True,
    )
assert bool(torch.all(info["converged"]))
assert bool(torch.all(torch.linalg.eigvalsh(matrices) > 0))
print("Untrained API smoke:", tuple(matrices.shape), "all valid")
