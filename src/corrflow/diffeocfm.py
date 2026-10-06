"""Correlation DiffeoCFM adapted from upstream 902282a (MIT).

See THIRD_PARTY_NOTICES.md. Retains the full class-wise OAS Gaussian prior and
float64 SELU MLP. All fitted quantities use training data only. No SPD code.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from corrbench.geometry import corr_to_normalized_cholesky, normalized_cholesky_to_corr
from sklearn.covariance import OAS
from sklearn.preprocessing import StandardScaler
from torch import nn

UPSTREAM_REVISION = "902282acd13a67784e2096d09ee93477b268c079"


def checked_correlations(matrices):
    x = np.asarray(matrices, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] != x.shape[-2] or not len(x):
        raise ValueError("Expected nonempty (samples, n, n) correlations")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite correlations")
    if not np.allclose(x, x.swapaxes(-1, -2), atol=1e-6, rtol=0):
        raise ValueError("Asymmetric correlations")
    if not np.allclose(np.diagonal(x, axis1=-2, axis2=-1), 1, atol=1e-6, rtol=0):
        raise ValueError("Nonunit diagonal")
    return (x + x.swapaxes(-1, -2)) / 2


class CorrelationGaussian(nn.Module):
    """Train-standardized normalized Cholesky coordinates and full OAS prior."""

    def __init__(self, *, matrix_dim, classes):
        super().__init__()
        self.matrix_dim, self.classes = matrix_dim, classes
        d = matrix_dim * (matrix_dim - 1) // 2
        self.register_buffer("center", torch.zeros(d, dtype=torch.float64))
        self.register_buffer("scale", torch.ones(d, dtype=torch.float64))
        self.register_buffer("means", torch.zeros(classes, d, dtype=torch.float64))
        self.register_buffer("factors", torch.zeros(classes, d, d, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))
        self.shrinkage = []

    @classmethod
    def fit(cls, matrices, labels):
        x, y = checked_correlations(matrices), np.asarray(labels)
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Integer labels must match matrix rows")
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Classes must start at zero and be contiguous")
        scaler = StandardScaler()
        z = scaler.fit_transform(corr_to_normalized_cholesky(x))
        prior = cls(matrix_dim=x.shape[-1], classes=len(classes))
        prior.center.copy_(torch.from_numpy(scaler.mean_))
        prior.scale.copy_(torch.from_numpy(scaler.scale_))
        for c in classes:
            group = z[y == c]
            if len(group) < 2:
                raise ValueError("Each class needs at least two training cases")
            estimator = OAS(store_precision=False).fit(group)
            prior.means[c].copy_(torch.from_numpy(group.mean(0)))
            prior.factors[c].copy_(
                torch.from_numpy(np.linalg.cholesky(estimator.covariance_))
            )
            prior.shrinkage.append(float(estimator.shrinkage_))
        prior.fitted.fill_(True)
        return prior, z

    def transform(self, matrices):
        x = checked_correlations(matrices)
        if x.shape[-1] != self.matrix_dim:
            raise ValueError("Wrong matrix dimension")
        z = torch.as_tensor(corr_to_normalized_cholesky(x), device=self.center.device)
        return (z - self.center) / self.scale

    def inverse(self, z):
        return normalized_cholesky_to_corr(z * self.scale + self.center)

    def sample(self, labels, *, generator):
        if not bool(self.fitted):
            raise RuntimeError("Fit the prior before sampling")
        if labels.ndim != 1 or labels.dtype != torch.int64:
            raise ValueError("Expected int64 label vector")
        if bool(torch.any((labels < 0) | (labels >= self.classes))):
            raise ValueError("Unknown class")
        noise = torch.randn(
            len(labels),
            len(self.center),
            dtype=torch.float64,
            device=self.center.device,
            generator=generator,
        )
        result = torch.empty_like(noise)
        # Do not duplicate d-by-d factors for every sample in the batch.
        for c in range(self.classes):
            mask = labels == c
            result[mask] = noise[mask] @ self.factors[c].T + self.means[c]
        return result


class DiffeoVectorField(nn.Module):
    """Upstream float64 SELU MLP with time and categorical one-hot inputs."""

    def __init__(self, *, vector_dim, classes, hidden=(512,)):
        super().__init__()
        if not hidden or any(x < 1 for x in hidden):
            raise ValueError("Positive hidden widths required")
        self.classes = classes
        widths = [vector_dim + classes + 1, *hidden]
        layers = []
        for left, right in zip(widths[:-1], widths[1:], strict=True):
            layers.extend([nn.Linear(left, right, dtype=torch.float64), nn.SELU()])
        layers.append(nn.Linear(hidden[-1], vector_dim, dtype=torch.float64))
        self.net = nn.Sequential(*layers)

    def forward(self, x, labels, t):
        condition = torch.nn.functional.one_hot(labels, num_classes=self.classes)
        t = t.reshape(-1, 1).expand(len(x), 1)
        return self.net(torch.cat([x, condition, t], dim=-1).double())


def flow_loss(model, x0, x1, labels, times):
    """Classic flow matching with sigma=0, as in the upstream default."""
    xt = (1 - times[:, None]) * x0 + times[:, None] * x1
    return torch.nn.functional.mse_loss(model(xt, labels, times), x1 - x0)


@torch.no_grad()
def sample_correlations(
    prior, model, labels, *, seed, batch_size=64, atol=1e-5, rtol=1e-5
):
    """Gaussian (model=None) or flow samples, including measured RHS counts."""
    if batch_size < 1 or not len(labels) or atol <= 0 or rtol <= 0:
        raise ValueError("Positive counts and tolerances required")
    from torchdiffeq import odeint

    device = prior.center.device
    labels = torch.as_tensor(labels, device=device, dtype=torch.int64)
    rng = torch.Generator(device).manual_seed(seed)
    if model is not None:
        model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    start = time.perf_counter()
    initial = prior.sample(labels, generator=rng)
    sync()
    prior_seconds = time.perf_counter() - start
    ode_seconds, inverse_seconds = 0.0, 0.0
    outputs, evaluations = [], []
    for begin in range(0, len(labels), batch_size):
        end = begin + batch_size
        y, z = labels[begin:end], initial[begin:end]
        nfe = 0

        def rhs(t, x, batch_labels=y):
            nonlocal nfe
            nfe += 1
            return model(x, batch_labels, t)

        started = time.perf_counter()
        if model is not None:
            z = odeint(
                rhs,
                z,
                torch.tensor([0.0, 1.0], device=device, dtype=torch.float64),
                method="dopri5",
                atol=atol,
                rtol=rtol,
                options={"min_step": 1e-5},
            )[-1]
        sync()
        ode_seconds += time.perf_counter() - started
        started = time.perf_counter()
        matrices = prior.inverse(z).cpu().numpy()
        inverse_seconds += time.perf_counter() - started
        if not np.isfinite(matrices).all():
            raise FloatingPointError("Nonfinite generated correlations")
        outputs.append(matrices)
        evaluations.append(nfe)
    counts = [len(x) for x in outputs]
    return np.concatenate(outputs), {
        "sampling_seconds": time.perf_counter() - start,
        "prior_seconds": prior_seconds,
        "ode_seconds": ode_seconds,
        "inverse_seconds": inverse_seconds,
        "rhs_calls_by_batch": evaluations,
        "batch_sizes": counts,
        "rhs_calls_total": sum(evaluations),
        "mean_rhs_evaluations_per_sample": float(
            np.average(evaluations, weights=counts)
        ),
        "method": "dopri5" if model is not None else "Gaussian",
        "atol": atol,
        "rtol": rtol,
    }
