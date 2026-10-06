#!/usr/bin/env python3
"""Compare direct small-universe samples with full-sample principal submatrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from corrbench.geometry import corr_to_gft
from corrbench.metrics import (
    energy_distance,
    orbit_sliced_wasserstein,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--direct-small", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.direct_small) as source:
        direct = source["matrices"]
        direct_indices = source["selected_test_indices"]
    with np.load(args.full) as source:
        full = source["matrices"]
        full_indices = source["selected_test_indices"]
    if not np.array_equal(direct_indices, full_indices):
        raise ValueError("The direct and full samples use different test selections.")
    if direct.shape[-1] != args.matrix_dim or full.shape[-1] < args.matrix_dim:
        raise ValueError("Sample dimensions do not match --matrix-dim.")
    principal = full[:, : args.matrix_dim, : args.matrix_dim]

    with np.load(args.data / "train.npz") as source:
        train = source["matrices"][:, : args.matrix_dim, : args.matrix_dim]
    train_features, feature_names = structural_features(train)
    feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    direct_features, _ = structural_features(direct)
    principal_features, _ = structural_features(principal)

    gft = []
    for matrices in (train, direct, principal):
        batches = []
        for start in range(0, matrices.shape[0], 256):
            batch = torch.from_numpy(matrices[start : start + 256]).to(args.device)
            batches.append(corr_to_gft(batch).cpu().numpy())
        gft.append(np.concatenate(batches))
    train_gft, direct_gft, principal_gft = gft
    gft_scale = float(train_gft.std())

    report = {
        "matrix_dim": args.matrix_dim,
        "samples": int(direct.shape[0]),
        "feature_names": feature_names,
        "direct_validity": validity_summary(direct),
        "principal_validity": validity_summary(principal),
        "gft_sliced_wasserstein": sliced_wasserstein(
            direct_gft,
            principal_gft,
            scale=gft_scale,
            seed=args.seed,
        ),
        "orbit_gft_sliced_wasserstein": orbit_sliced_wasserstein(
            direct_gft,
            principal_gft,
            scale=gft_scale,
            seed=args.seed,
        ),
        "feature_sliced_wasserstein": sliced_wasserstein(
            direct_features,
            principal_features,
            scale=feature_scale,
            seed=args.seed,
            device="cpu",
        ),
        "feature_energy_distance": energy_distance(
            direct_features / feature_scale,
            principal_features / feature_scale,
            seed=args.seed,
        ),
        "feature_mean_absolute_z_difference": float(
            np.mean(
                np.abs(direct_features.mean(0) - principal_features.mean(0))
                / feature_scale
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
