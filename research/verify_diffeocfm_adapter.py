"""Numerically compare adapter components with pinned upstream MIT source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from corrflow.diffeocfm import (
    UPSTREAM_REVISION,
    CorrelationGaussian,
    DiffeoVectorField,
    flow_loss,
)

SOURCE_HASHES = {
    "diffeo.py": "43e21370085aa9630c3cf3f5e59164416430807f902b13ddf8569f387cd4da76",
    "gaussian.py": "8c84061e26abeec64d57e29a454d7e41bf5c34c0dce2f8aea4a1e7318580c5c2",
    "model.py": "1024f03c860e52d231863d3f1c6cce822cbc5012e482c2e241c248756c2b3ebe",
    "LICENCE": "9cdcd6ab0442036aa61fb37d969bc327146080f96e2e2ddc22aecdf39375dfd8",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.upstream.resolve()
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream file changed: {name}")
    if any(name in sys.modules for name in ["diffeo", "gaussian", "model"]):
        raise RuntimeError("Run in a fresh process")
    sys.path.insert(0, str(root))
    from gaussian import DiffeoGauss
    from model import MLP
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    rng = np.random.default_rng(20261005)
    observations = rng.normal(size=(120, 40, 8))
    observations[:, :, 1] += 0.7 * observations[:, :, 0]
    matrices = np.stack([np.corrcoef(x.T) for x in observations])
    labels = np.tile(np.arange(3), 40)
    official = DiffeoGauss({"DIFFEO": "corrcholesky", "RNG": 20261005})
    official.fit(matrices, labels)
    prior, z = CorrelationGaussian.fit(matrices, labels)
    differences = {}

    def compare(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-12)
        differences[name] = float(np.max(np.abs(a - b)))

    compare("transformed_training", z, official._transform_features(matrices))
    compare("class_means", prior.means.numpy(), official._prior.means_)
    compare(
        "full_covariance_cholesky",
        prior.factors.numpy(),
        official._prior.covariances_cholesky_,
    )
    compare(
        "inverse",
        prior.inverse(torch.from_numpy(z)).numpy(),
        official._inverse_transform_features(z),
    )
    # Identical weights and inputs isolate architecture/objective from RNG order.
    model = DiffeoVectorField(vector_dim=28, classes=3, hidden=[32])
    original = MLP(input_dim=28, cond_dim=3, hidden_dim=[32])
    original.load_state_dict(model.state_dict())
    y = torch.from_numpy(labels)
    x1 = torch.from_numpy(z)
    x0 = prior.sample(y, generator=torch.Generator().manual_seed(17))
    times = torch.linspace(0.01, 0.99, len(y), dtype=torch.float64)
    condition = torch.nn.functional.one_hot(y, num_classes=3)
    _, xt, ut = ConditionalFlowMatcher(sigma=0).sample_location_and_conditional_flow(
        x0, x1, t=times
    )
    compare(
        "vector_field",
        model(xt, y, times).detach().numpy(),
        original(xt, condition, times).detach().numpy(),
    )
    local_loss = flow_loss(model, x0, x1, y, times)
    original_loss = torch.nn.functional.mse_loss(original(xt, condition, times), ut)
    compare("flow_loss", local_loss.detach().numpy(), original_loss.detach().numpy())
    local_loss.backward()
    original_loss.backward()
    compare(
        "parameter_gradients",
        torch.cat([p.grad.flatten() for p in model.parameters()]).numpy(),
        torch.cat([p.grad.flatten() for p in original.parameters()]).numpy(),
    )
    result = {
        "scope": "synthetic component parity; excludes upstream training RNG/results",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_hashes": SOURCE_HASHES,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_absolute_differences": differences,
        "versions": {
            p: version(p)
            for p in ["torch", "torchcfm", "torchdiffeq", "pyriemann", "scikit-learn"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
