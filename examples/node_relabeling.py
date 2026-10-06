"""Check that node identities move with a relabeled matrix."""

import json

import torch

from corrflow.models.node_conditioned import NodeConditionedMatrixVectorField

torch.set_num_threads(2)
torch.manual_seed(17)
attributes = torch.eye(6, dtype=torch.float64)
model = NodeConditionedMatrixVectorField(
    node_features=attributes, hidden_dim=16, depth=2, condition_dim=2
).double()
x = torch.randn(1, 6, 6, dtype=torch.float64)
x = (x + x.transpose(-1, -2)) / 2
t = torch.tensor([0.4], dtype=torch.float64)
condition = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
permutation = torch.tensor([3, 1, 5, 0, 2, 4])
with torch.no_grad():
    original = model(x, t, condition)
    permuted = model(
        x[:, permutation][:, :, permutation],
        t,
        condition,
        model.node_features[permutation],
    )
    incorrect = model(x[:, permutation][:, :, permutation], t, condition)
    error = (permuted - original[:, permutation][:, :, permutation]).abs().max()
    wrong_error = (incorrect - permuted).abs().max()
assert error < 1e-12
assert wrong_error > 1e-7
print(
    json.dumps(
        {
            "simultaneous_relabeling_max_error": float(error),
            "leaving_node_identities_fixed_max_difference": float(wrong_error),
            "scope": "Random-weight field identity, not empirical model fidelity",
        },
        indent=2,
    )
)
