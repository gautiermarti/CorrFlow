#!/usr/bin/env python3
"""Train conditional permutation-equivariant rectified flow in GFT space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from corrbench.geometry import lower_to_symmetric
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary

from corrflow.flow import hollow_symmetric_noise, rectified_flow_loss
from corrflow.models import EquivariantMatrixVectorField, MLPMatrixVectorField
from corrflow.sampling import sample_correlations


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def make_conditions(
    labels: np.ndarray,
    sharpes: np.ndarray | None,
    *,
    sharpe_mean: float | None = None,
    sharpe_std: float | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    one_hot = np.eye(3, dtype=np.float32)[labels]
    if sharpes is None:
        return one_hot, {"sharpe_mean": None, "sharpe_std": None}
    mean = float(sharpes.mean()) if sharpe_mean is None else sharpe_mean
    std = float(sharpes.std()) if sharpe_std is None else sharpe_std
    standardized = ((sharpes - mean) / max(std, 1e-8)).astype(np.float32)
    return np.column_stack([one_hot, standardized]), {
        "sharpe_mean": mean,
        "sharpe_std": std,
    }


def update_ema(
    ema: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                ema[name].lerp_(value.detach(), 1 - decay)
            else:
                ema[name].copy_(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    parser.add_argument("--epochs", type=int, help="Override configured epochs.")
    parser.add_argument("--coordinates", choices=["gft", "corrcholesky"], default="gft")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)

    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"] if args.seed is None else args.seed)
    training["seed"] = seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    categorical_only = config.get("conditioning") == "categorical_only"
    structural_every = int(training.get("structural_validation_every", 0))
    selection_seed = int(training.get("validation_seed", 44301))

    coordinate_key = "gft" if args.coordinates == "gft" else "coordinates"
    with np.load(args.data / args.coordinates / "train.npz") as source:
        train_gft_np = source[coordinate_key]
        train_labels = source["labels"]
        train_sharpes = source["sharpes"] if "sharpes" in source.files else None
    with np.load(args.data / args.coordinates / "validation.npz") as source:
        validation_gft_np = source[coordinate_key]
        validation_labels = source["labels"]
        validation_sharpes = source["sharpes"] if "sharpes" in source.files else None
    if categorical_only:
        train_sharpes = validation_sharpes = None

    coordinate_mean = float(train_gft_np.mean())
    coordinate_std = float(train_gft_np.std())
    train_conditions_np, condition_scaler = make_conditions(train_labels, train_sharpes)
    validation_conditions_np, _ = make_conditions(
        validation_labels,
        validation_sharpes,
        sharpe_mean=condition_scaler["sharpe_mean"],
        sharpe_std=condition_scaler["sharpe_std"],
    )

    train_gft = torch.from_numpy((train_gft_np - coordinate_mean) / coordinate_std).to(
        device
    )
    train_conditions = torch.from_numpy(train_conditions_np).to(device)
    validation_gft = torch.from_numpy(
        (validation_gft_np - coordinate_mean) / coordinate_std
    ).to(device)
    validation_conditions = torch.from_numpy(validation_conditions_np).to(device)
    del train_gft_np, validation_gft_np

    model_config = dict(config["model"])
    architecture = model_config.pop("architecture", "equivariant")
    model_class = (
        EquivariantMatrixVectorField
        if architecture == "equivariant"
        else MLPMatrixVectorField
    )
    model = model_class(**model_config, condition_dim=train_conditions.shape[1]).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    precision = training.get("precision", "float32")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    validation_count = min(512, validation_gft.shape[0])
    validation_indices = torch.randperm(validation_gft.shape[0], device=device)[
        :validation_count
    ]
    validation_target = lower_to_symmetric(validation_gft[validation_indices])
    validation_condition = validation_conditions[validation_indices]
    validation_noise = hollow_symmetric_noise(
        validation_count,
        validation_target.shape[-1],
        dtype=validation_target.dtype,
        device=device,
        generator=torch.Generator(device).manual_seed(seed + 1),
    )
    validation_time = torch.rand(
        validation_count,
        device=device,
        generator=torch.Generator(device).manual_seed(seed + 2),
    )

    args.output.mkdir(parents=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    data_paths = [
        args.data / args.coordinates / f"{split}.npz"
        for split in ["train", "validation"]
    ]
    if structural_every:
        data_paths += [args.data / f"{split}.npz" for split in ["train", "validation"]]
        with np.load(args.data / "train.npz") as a:
            if not np.array_equal(a["labels"], train_labels):
                raise ValueError("Training coordinate cache labels differ")
            train_features, _ = structural_features(a["matrices"])
        feature_scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
        rng = np.random.default_rng(selection_seed)
        per_class = int(training["validation_per_class"])
        selected = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(validation_labels == c), per_class, replace=False
                )
                for c in np.unique(validation_labels)
            ]
        )
        with np.load(args.data / "validation.npz") as a:
            if not np.array_equal(a["labels"], validation_labels):
                raise ValueError("Validation coordinate cache labels differ")
            real_features, _ = structural_features(a["matrices"][selected])
        structural_conditions = validation_conditions[selected]
        np.save(args.output / "selected_validation_indices.npy", selected)
    import inspect

    import corrflow.flow as flow_module
    import corrflow.sampling as sampling_module

    provenance = {
        "protocol": config.get("protocol", "legacy_flow"),
        "conditioning": "categorical_only" if categorical_only else "available_scores",
        "source_hashes": {
            str(path.relative_to(args.data)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in data_paths
        },
        "code_hashes": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [
                ("trainer", __file__),
                ("flow", flow_module.__file__),
                ("sampling", sampling_module.__file__),
                ("model", inspect.getfile(model_class)),
            ]
        },
        "selection": "minimum fixed-noise structural SWD, raw weights"
        if structural_every
        else "validation flow loss",
        "torch": torch.__version__,
        "device": str(device),
        "training_rows": len(train_labels),
        "validation_rows": len(validation_labels),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    history = []
    best_validation = math.inf
    best_structural = math.inf
    global_step = 0
    started = time.monotonic()
    stop = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_gft.shape[0], device=device)
        losses = []
        for start in range(0, train_gft.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            target = lower_to_symmetric(train_gft[indices])
            condition = train_conditions[indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ):
                loss = rectified_flow_loss(model, target, condition)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            update_ema(ema, model, float(training["ema_decay"]))
            losses.append(float(loss.detach()))
            global_step += 1
            if global_step % 100 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"step={global_step} loss={np.mean(losses[-100:]):.6f} "
                    f"grad={float(gradient_norm):.3f} seconds={elapsed:.1f}",
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        model.eval()
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ),
        ):
            validation_loss = float(
                rectified_flow_loss(
                    model,
                    validation_target,
                    validation_condition,
                    noise=validation_noise,
                    time=validation_time,
                )
            )
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "elapsed_seconds": time.monotonic() - started,
        }
        if structural_every and (
            epoch == 0 or (epoch + 1) % structural_every == 0 or epoch + 1 == epochs
        ):
            generated, timing = sample_correlations(
                model,
                structural_conditions,
                matrix_dim=validation_target.shape[-1],
                coordinate_mean=coordinate_mean,
                coordinate_std=coordinate_std,
                seed=selection_seed,
                coordinate_transform=args.coordinates,
                **config.get("sampling", {}),
            )
            features, _ = structural_features(generated)
            row["structural"] = {
                "feature_swd": sliced_wasserstein(
                    real_features,
                    features,
                    scale=feature_scale,
                    seed=selection_seed,
                    device="cpu",
                ),
                "validity": validity_summary(generated),
                "sampling": timing,
            }
            row["elapsed_seconds"] = time.monotonic() - started
        if device.type == "cuda":
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        history.append(row)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema,
            "model_config": {
                **model_config,
                "condition_dim": int(train_conditions.shape[1]),
            },
            "architecture": architecture,
            "coordinate_transform": args.coordinates,
            "training_config": training,
            "coordinate_mean": coordinate_mean,
            "coordinate_std": coordinate_std,
            "condition_scaler": condition_scaler,
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation_loss": validation_loss,
            "history_row": row,
            "sampling_config": config.get("sampling", {}),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, args.output / "best.pt")
        score = row.get("structural", {}).get("feature_swd", math.inf)
        if score < best_structural:
            best_structural = score
            torch.save(checkpoint, args.output / "best_structural.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stop:
            break
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "epochs_completed": len(history),
                "steps_completed": global_step,
                "stopped_at_max_steps": stop,
                "training_seconds": time.monotonic() - started,
                "best_structural_swd": best_structural if structural_every else None,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
