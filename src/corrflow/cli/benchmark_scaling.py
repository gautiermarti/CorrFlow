#!/usr/bin/env python3
"""Measure CorrFlow compute capacity separately from learned size generalization."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from corrbench.geometry import corr_to_gft, gft_to_corr, symmetric_to_lower

from corrflow.flow import sample_flow
from corrflow.models import EquivariantMatrixVectorField


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[80, 160, 200, 500, 1000, 2000]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--max-memory-gib", type=float, default=16.0)
    args = parser.parse_args()
    if min(args.dimensions + [args.batch_size, args.repeats, args.steps]) < 1:
        parser.error("Dimensions, batch size, repeats and steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(1701)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device)
    model = EquivariantMatrixVectorField(
        hidden_dim=48,
        depth=3,
        condition_dim=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    condition = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(
        args.batch_size, 1
    )

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def timed(fn):
        sync()
        start = time.perf_counter()
        value = fn()
        sync()
        return time.perf_counter() - start, value

    report = {
        "purpose": "synthetic compute benchmark; no learned fidelity claim",
        "model": "48 hidden channels, 3 blocks; random weights, synthetic edges",
        "device": torch.cuda.get_device_name(device) if cuda else "cpu",
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "steps": args.steps,
        "model_evaluations": 2 * args.steps,
        "precision": "bfloat16 autocast" if cuda else "float32",
        "inverse_input": "synthetic sample correlation, ceil(3.2*n) observations, "
        "Gaussian common factor; float32",
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    previous_peak = None
    previous_n = None
    for n in args.dimensions:
        row = {"n": n}
        # Stop conservatively before allocating a size that is likely too large.
        if (
            cuda
            and previous_peak is not None
            and previous_peak * (n / previous_n) ** 2 * 1.25
            > args.max_memory_gib * 2**30
        ):
            row["status"] = "skipped: projected peak exceeds memory budget"
            report["rows"].append(row)
            save()
            continue
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(args.batch_size, n, n, device=device)
        x = (x + x.transpose(-1, -2)) / math.sqrt(2)
        x.diagonal(dim1=-2, dim2=-1).zero_()
        t = torch.full((args.batch_size,), 0.5, device=device)

        def train_step(x=x, t=t):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=cuda
            ):
                loss = (model(x, t, condition) - x).square().mean()
            loss.backward()
            optimizer.step()

        try:
            train_step()  # Warm up kernels and optimizer state.
            durations = [timed(train_step)[0] for _ in range(args.repeats)]
            row["train_step_seconds_median"] = float(np.median(durations))
            if cuda:
                previous_peak = torch.cuda.max_memory_allocated(device)
                previous_n = n
                row["training_peak_allocated_gib"] = previous_peak / 2**30
                row["training_peak_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / 2**30
                )
            optimizer.zero_grad(set_to_none=True)
            model.eval()

            def sample(n=n):
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=cuda,
                    ),
                ):
                    return sample_flow(
                        model,
                        condition,
                        matrix_dim=n,
                        device=device,
                        steps=args.steps,
                    )

            sample()
            times = [timed(sample)[0] for _ in range(args.repeats)]
            row["ode_seconds_per_matrix_median"] = (
                float(np.median(times)) / args.batch_size
            )
            # A known valid endpoint makes inverse scaling measurable without
            # pretending a model trained at 80 is statistically valid at n.
            observations = max(n + 2, math.ceil(3.2 * n))
            returns = torch.randn(args.batch_size, observations, n, device=device)
            returns += 0.5 * torch.randn(
                args.batch_size, observations, 1, device=device
            )
            returns -= returns.mean(dim=1, keepdim=True)
            covariance = returns.transpose(-1, -2) @ returns
            scale = covariance.diagonal(dim1=-2, dim2=-1).sqrt()
            target = covariance / scale[:, :, None] / scale[:, None, :]
            coordinates = corr_to_gft(target)
            del returns, covariance, scale

            def invert(coordinates=coordinates):
                return gft_to_corr(
                    coordinates,
                    tolerance=1e-6,
                    max_iterations=1000,
                    return_info=True,
                )

            invert()
            inverse_times = []
            for _ in range(args.repeats):
                elapsed, (corr, info) = timed(invert)
                inverse_times.append(elapsed)
            row.update(
                {
                    "inverse_seconds_per_matrix_median": float(np.median(inverse_times))
                    / args.batch_size,
                    "inverse_converged": bool(torch.all(info["converged"])),
                    "inverse_iterations": int(info["iterations"]),
                    "inverse_max_entry_error": float((corr - target).abs().max()),
                    "inverse_minimum_eigenvalue": float(
                        torch.linalg.eigvalsh(corr).min()
                    ),
                    "distinct_edges": int(symmetric_to_lower(x).shape[-1]),
                    "status": "measured",
                }
            )
            del coordinates, corr, target
        except torch.cuda.OutOfMemoryError:
            row["status"] = "out of memory; stopped"
            report["rows"].append(row)
            save()
            break
        report["rows"].append(row)
        save()
        print(json.dumps(row), flush=True)
        del x, t, train_step, sample, invert


if __name__ == "__main__":
    main()
