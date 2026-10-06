#!/usr/bin/env python3
"""Measure numerical permutation equivariance of a flow checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from corrflow.flow import hollow_symmetric_noise
from corrflow.models import EquivariantMatrixVectorField, MLPMatrixVectorField


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new evaluation output: {args.output}")

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    architecture = checkpoint.get("architecture", "equivariant")
    model_class = (
        EquivariantMatrixVectorField
        if architecture == "equivariant"
        else MLPMatrixVectorField
    )
    model = model_class(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    matrix_dim = int(checkpoint["model_config"].get("matrix_dim", 80))
    condition_dim = int(checkpoint["model_config"]["condition_dim"])
    generator = torch.Generator(args.device).manual_seed(args.seed)
    matrices = hollow_symmetric_noise(
        args.batch_size,
        matrix_dim,
        dtype=torch.float32,
        device=args.device,
        generator=generator,
    )
    times = torch.rand(args.batch_size, device=args.device, generator=generator)
    conditions = torch.randn(
        args.batch_size,
        condition_dim,
        device=args.device,
        generator=generator,
    )
    permutation = torch.randperm(matrix_dim, device=args.device, generator=generator)
    permuted = matrices[:, permutation][:, :, permutation]

    with torch.no_grad():
        expected = model(matrices, times, conditions)[:, permutation][:, :, permutation]
        observed = model(permuted, times, conditions)
    error = (expected - observed).abs()
    denominator = torch.linalg.vector_norm(expected.reshape(args.batch_size, -1), dim=1)
    relative = torch.linalg.vector_norm(
        error.reshape(args.batch_size, -1), dim=1
    ) / denominator.clamp_min(1e-12)
    report = {
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "matrix_dim": matrix_dim,
        "batch_size": args.batch_size,
        "maximum_absolute_error": float(error.max()),
        "mean_absolute_error": float(error.mean()),
        "mean_relative_frobenius_error": float(relative.mean()),
        "maximum_relative_frobenius_error": float(relative.max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
