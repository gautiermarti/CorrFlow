#!/usr/bin/env python3
"""Sample a trained CorrFlow checkpoint and evaluate a frozen test subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from corrbench.geometry import (
    corr_to_gft,
    gft_to_corr,
    normalized_cholesky_to_corr,
    symmetric_to_lower,
)
from corrbench.metrics import (
    classifier_two_sample_auc,
    energy_distance,
    manifold_precision_recall,
    nearest_train_diagnostic,
    orbit_sliced_wasserstein,
    portfolio_functionals,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)

from corrflow.diffusion import cosine_schedule, sample_equivariant_ddim
from corrflow.flow import sample_flow
from corrflow.models import EquivariantMatrixVectorField, MLPMatrixVectorField


def stratified_subset(
    labels: np.ndarray, per_class: int, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        selected.extend(
            rng.choice(candidates, min(per_class, candidates.size), replace=False)
        )
    selected = np.asarray(selected)
    return selected[rng.permutation(selected.size)]


def make_conditions(
    labels: np.ndarray,
    sharpes: np.ndarray | None,
    scaler: dict[str, float | None],
) -> np.ndarray:
    one_hot = np.eye(3, dtype=np.float32)[labels]
    if sharpes is None:
        return one_hot
    standardized = (
        (sharpes - float(scaler["sharpe_mean"])) / float(scaler["sharpe_std"])
    ).astype(np.float32)
    return np.column_stack([one_hot, standardized])


def metrics(
    real_gft: np.ndarray,
    generated_gft: np.ndarray,
    real_corr: np.ndarray,
    generated_corr: np.ndarray,
    *,
    coordinate_scale: float,
    feature_scale: np.ndarray,
    portfolio_scale: np.ndarray,
    seed: int,
    train_features: np.ndarray | None = None,
) -> dict[str, object]:
    real_features, _ = structural_features(real_corr)
    generated_features, _ = structural_features(generated_corr)
    standardized_real = real_features / feature_scale
    standardized_generated = generated_features / feature_scale
    real_portfolio, portfolio_names = portfolio_functionals(real_corr)
    generated_portfolio, _ = portfolio_functionals(generated_corr)
    report: dict[str, object] = {
        "gft_sliced_wasserstein": sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=coordinate_scale,
            seed=seed,
        ),
        "orbit_gft_sliced_wasserstein": orbit_sliced_wasserstein(
            real_gft,
            generated_gft,
            scale=coordinate_scale,
            seed=seed,
        ),
        "feature_sliced_wasserstein": sliced_wasserstein(
            real_features,
            generated_features,
            scale=feature_scale,
            seed=seed,
            device="cpu",
        ),
        "feature_energy_distance": energy_distance(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_manifold": manifold_precision_recall(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_classifier_two_sample": classifier_two_sample_auc(
            standardized_real,
            standardized_generated,
            seed=seed,
        ),
        "feature_mean_absolute_z_error": float(
            np.mean(
                np.abs(real_features.mean(axis=0) - generated_features.mean(axis=0))
                / feature_scale
            )
        ),
        "portfolio_functional_names": portfolio_names,
        "portfolio_sliced_wasserstein": sliced_wasserstein(
            real_portfolio,
            generated_portfolio,
            scale=portfolio_scale,
            seed=seed,
            device="cpu",
        ),
        "portfolio_energy_distance": energy_distance(
            real_portfolio / portfolio_scale,
            generated_portfolio / portfolio_scale,
            seed=seed,
        ),
        "validity": validity_summary(generated_corr),
    }
    if train_features is not None:
        report["nearest_train"] = nearest_train_diagnostic(
            train_features,
            real_features,
            generated_features,
            scale=feature_scale,
            seed=seed,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=600)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Evaluate the complete natural test distribution instead of stratifying.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument(
        "--matrix-dim",
        type=int,
        help="Evaluate zero-shot at a smaller dimension using principal submatrices.",
    )
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json; arrays use the matching .npz path")
    for target in (args.output, args.output.with_suffix(".npz")):
        if target.exists():
            raise FileExistsError(f"Choose a new evaluation output: {target}")
    device_type = torch.device(args.device).type

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model_class = (
        EquivariantMatrixVectorField
        if checkpoint.get("architecture", "equivariant") == "equivariant"
        else MLPMatrixVectorField
    )
    model = model_class(**checkpoint["model_config"]).to(args.device)
    state = checkpoint["ema_model"] if args.weights == "ema" else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    generative_objective = checkpoint.get("generative_objective", "flow")
    if generative_objective == "diffusion":
        betas = cosine_schedule(int(checkpoint["diffusion_config"]["timesteps"])).to(
            args.device
        )
        cumulative_alphas = torch.cumprod(1 - betas, dim=0)
    elif generative_objective != "flow":
        raise ValueError(f"Unsupported generative objective: {generative_objective}")

    coordinate_transform = checkpoint.get("coordinate_transform", "gft")
    coordinate_key = "gft" if coordinate_transform == "gft" else "coordinates"
    with np.load(args.data / coordinate_transform / "test.npz") as source:
        if coordinate_key not in source.files:
            raise KeyError(f"Missing {coordinate_key} in coordinate cache.")
        all_labels = source["labels"]
        all_sharpes = source["sharpes"] if "sharpes" in source.files else None
    with np.load(args.data / "gft" / "test.npz") as source:
        all_gft = source["gft"]
    with np.load(args.data / "test.npz") as source:
        all_corr = source["matrices"]
    with np.load(args.data / "train.npz") as source:
        train_corr = source["matrices"]

    source_dim = int(all_corr.shape[-1])
    matrix_dim = source_dim if args.matrix_dim is None else args.matrix_dim
    if not 2 <= matrix_dim <= source_dim:
        raise ValueError(f"matrix_dim must be between 2 and {source_dim}")
    if (
        matrix_dim != source_dim
        and checkpoint.get("architecture", "equivariant") != "equivariant"
    ):
        raise ValueError("zero-shot dimension transfer requires an equivariant model")
    if matrix_dim != source_dim and coordinate_transform != "gft":
        raise ValueError("zero-shot dimension transfer is currently defined for GFT")

    rng = np.random.default_rng(args.seed)
    selected = (
        np.arange(all_labels.size)
        if args.all_samples
        else stratified_subset(all_labels, args.per_class, rng)
    )
    labels = all_labels[selected]
    real_corr = all_corr[selected, :matrix_dim, :matrix_dim]
    train_corr = train_corr[:, :matrix_dim, :matrix_dim]
    if matrix_dim == source_dim:
        real_gft = all_gft[selected]
    else:
        real_gft = (
            corr_to_gft(torch.from_numpy(real_corr).to(args.device)).cpu().numpy()
        )
    sharpes = None if all_sharpes is None else all_sharpes[selected]
    conditions = make_conditions(labels, sharpes, checkpoint["condition_scaler"])

    generated_vectors = []
    generator = torch.Generator(args.device).manual_seed(args.seed)
    sampling_started = time.perf_counter()
    for start in range(0, labels.size, args.batch_size):
        condition = torch.from_numpy(conditions[start : start + args.batch_size]).to(
            args.device
        )
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=device_type == "cuda",
        ):
            if generative_objective == "diffusion":
                normalized_matrix = sample_equivariant_ddim(
                    model,
                    condition,
                    cumulative_alphas,
                    matrix_dim=matrix_dim,
                    steps=args.steps,
                    prediction_type=checkpoint["diffusion_config"].get(
                        "prediction", "epsilon"
                    ),
                    generator=generator,
                )
            else:
                normalized_matrix = sample_flow(
                    model,
                    condition,
                    matrix_dim=matrix_dim,
                    steps=args.steps,
                    device=args.device,
                    generator=generator,
                )
        normalized_vector = symmetric_to_lower(normalized_matrix.float())
        generated_vectors.append(
            (
                normalized_vector * checkpoint["coordinate_std"]
                + checkpoint["coordinate_mean"]
            )
            .cpu()
            .numpy()
        )
        print(f"sampled {min(start + args.batch_size, labels.size)}/{labels.size}")
    generated_coordinates = np.concatenate(generated_vectors).astype(np.float32)
    sampling_seconds = time.perf_counter() - sampling_started

    generated_correlations = []
    inverse_converged = True
    inverse_failed_samples = 0
    inverse_retried_samples = 0
    inverse_maximum_iterations = 0
    inverse_started = time.perf_counter()
    for start in range(0, labels.size, 256):
        batch = torch.from_numpy(generated_coordinates[start : start + 256]).to(
            args.device
        )
        if coordinate_transform == "gft":
            corr, info = gft_to_corr(
                batch,
                tolerance=1e-6,
                max_iterations=200,
                return_info=True,
            )
            failed = ~info["converged"]
            inverse_retried_samples += int(failed.sum())
            if bool(torch.any(failed)):
                retry_corr, retry_info = gft_to_corr(
                    batch[failed],
                    tolerance=1e-6,
                    max_iterations=1000,
                    return_info=True,
                )
                corr[failed] = retry_corr
                inverse_failed_samples += int((~retry_info["converged"]).sum())
                inverse_converged &= bool(torch.all(retry_info["converged"]))
                inverse_maximum_iterations = max(
                    inverse_maximum_iterations, int(retry_info["iterations"])
                )
            inverse_maximum_iterations = max(
                inverse_maximum_iterations, int(info["iterations"])
            )
        else:
            corr = normalized_cholesky_to_corr(batch)
        generated_correlations.append(corr.cpu().numpy().astype(np.float32))
    generated_corr = np.concatenate(generated_correlations)
    inverse_seconds = time.perf_counter() - inverse_started
    generated_gft = (
        corr_to_gft(torch.from_numpy(generated_corr).to(args.device)).cpu().numpy()
    )

    train_features, _ = structural_features(train_corr)
    feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    train_portfolio, _ = portfolio_functionals(train_corr)
    portfolio_scale = np.maximum(train_portfolio.std(axis=0, ddof=1), 1e-6)
    report = {
        "checkpoint": str(args.checkpoint),
        "generative_objective": generative_objective,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_loss": checkpoint["validation_loss"],
        "steps": args.steps,
        "weights": args.weights,
        "coordinate_transform": coordinate_transform,
        "samples": int(labels.size),
        "sample_selection": "all" if args.all_samples else "stratified",
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "sampling_algorithm": "DDIM" if generative_objective == "diffusion" else "Heun",
        "model_evaluations_per_sample": int(
            args.steps if generative_objective == "diffusion" else 2 * args.steps
        ),
        "sampling_seconds": sampling_seconds,
        "sampling_seconds_per_1000": sampling_seconds / labels.size * 1000,
        "inverse_seconds": inverse_seconds,
        "training_matrix_dim": source_dim,
        "evaluation_matrix_dim": matrix_dim,
        "inverse_converged": inverse_converged,
        "inverse_failed_samples": inverse_failed_samples,
        "inverse_retried_samples": inverse_retried_samples,
        "inverse_maximum_iterations": inverse_maximum_iterations,
        "pooled": metrics(
            real_gft,
            generated_gft,
            real_corr,
            generated_corr,
            coordinate_scale=float(checkpoint["coordinate_std"]),
            feature_scale=feature_scale,
            portfolio_scale=portfolio_scale,
            seed=args.seed,
            train_features=train_features,
        ),
        "by_class": {},
    }
    for label in range(3):
        mask = labels == label
        report["by_class"][str(label)] = metrics(
            real_gft[mask],
            generated_gft[mask],
            real_corr[mask],
            generated_corr[mask],
            coordinate_scale=float(checkpoint["coordinate_std"]),
            feature_scale=feature_scale,
            portfolio_scale=portfolio_scale,
            seed=args.seed + label + 1,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        gft=generated_gft,
        coordinates=generated_coordinates,
        matrices=generated_corr,
        labels=labels,
        selected_test_indices=selected,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
