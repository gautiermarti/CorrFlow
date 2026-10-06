"""Audited flow sampling with independent RNG and converged geometric inverse."""

from __future__ import annotations

import time

import numpy as np
import torch
from corrbench.geometry import (
    gft_to_corr,
    normalized_cholesky_to_corr,
    symmetric_to_lower,
)

from corrflow.flow import sample_flow


@torch.no_grad()
def sample_correlations(
    model,
    conditions,
    *,
    matrix_dim,
    coordinate_mean,
    coordinate_std,
    seed,
    steps=25,
    batch_size=32,
    precision="bfloat16",
    coordinate_transform="gft",
    inverse_tolerance=1e-8,
    inverse_max_iterations=1000,
):
    if min(steps, batch_size, matrix_dim, len(conditions)) < 1:
        raise ValueError("Positive dimensions, steps and sample counts required")
    if coordinate_transform not in ["gft", "corrcholesky"]:
        raise ValueError("Unsupported coordinate transform")
    if precision not in ["float32", "bfloat16"]:
        raise ValueError("Unsupported sampling precision")
    device = next(model.parameters()).device
    conditions = torch.as_tensor(conditions, device=device, dtype=torch.float32)
    generator = torch.Generator(device).manual_seed(seed)
    model.eval()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    class CountedField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, *args):
            self.calls += 1
            return model(*args)

    counted = CountedField()
    sync()
    started = time.perf_counter()
    coordinates, calls, counts = [], [], []
    for begin in range(0, len(conditions), batch_size):
        condition = conditions[begin : begin + batch_size]
        previous = counted.calls
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            matrix = sample_flow(
                counted,
                condition,
                matrix_dim=matrix_dim,
                steps=steps,
                device=device,
                generator=generator,
            )
        coordinates.append(
            symmetric_to_lower(matrix.float()) * coordinate_std + coordinate_mean
        )
        calls.append(counted.calls - previous)
        counts.append(len(condition))
    sync()
    flow_seconds = time.perf_counter() - started
    inverse_started = time.perf_counter()
    outputs, iterations = [], []
    for vectors in coordinates:
        if coordinate_transform == "gft":
            correlations, info = gft_to_corr(
                vectors.double(),
                tolerance=inverse_tolerance,
                max_iterations=inverse_max_iterations,
                return_info=True,
            )
            if not bool(torch.all(info["converged"])):
                raise RuntimeError("GFT inverse did not converge; refusing output")
            iterations.append(int(info["iterations"]))
        else:
            correlations = normalized_cholesky_to_corr(vectors.double())
        if not bool(torch.all(torch.isfinite(correlations))):
            raise FloatingPointError("Nonfinite generated matrices")
        outputs.append(correlations.cpu().numpy())
    sync()
    return np.concatenate(outputs), {
        "sampling_seconds": time.perf_counter() - started,
        "flow_seconds": flow_seconds,
        "inverse_seconds": time.perf_counter() - inverse_started,
        "inverse_dtype": "float64",
        "inverse_converged": True,
        "inverse_tolerance": inverse_tolerance,
        "inverse_max_iterations": max(iterations, default=0),
        "rhs_calls_by_batch": calls,
        "batch_sizes": counts,
        "rhs_calls_total": sum(calls),
        "mean_rhs_evaluations_per_sample": float(np.average(calls, weights=counts)),
        "method": "Heun",
        "steps": steps,
        "precision": precision,
    }
