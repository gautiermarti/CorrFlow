#!/usr/bin/env python3
"""Train conditional permutation-equivariant diffusion in GFT space."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from corrbench.geometry import lower_to_symmetric

from corrflow.diffusion import cosine_schedule, equivariant_ddpm_loss
from corrflow.flow import hollow_symmetric_noise
from corrflow.models import EquivariantMatrixVectorField


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


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
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    seed = int(training["seed"] if args.seed is None else args.seed)
    training["seed"] = seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    arrays = {}
    for split in ("train", "validation"):
        with np.load(args.data / "gft" / f"{split}.npz") as source:
            arrays[split] = {
                "gft": source["gft"],
                "labels": source["labels"],
                "sharpes": source["sharpes"] if "sharpes" in source.files else None,
            }

    coordinate_mean = float(arrays["train"]["gft"].mean())
    coordinate_std = float(arrays["train"]["gft"].std())
    train_conditions_np, condition_scaler = make_conditions(
        arrays["train"]["labels"], arrays["train"]["sharpes"]
    )
    validation_conditions_np, _ = make_conditions(
        arrays["validation"]["labels"],
        arrays["validation"]["sharpes"],
        sharpe_mean=condition_scaler["sharpe_mean"],
        sharpe_std=condition_scaler["sharpe_std"],
    )

    train_gft = torch.from_numpy(
        (arrays["train"]["gft"] - coordinate_mean) / coordinate_std
    ).to(device)
    validation_gft = torch.from_numpy(
        (arrays["validation"]["gft"] - coordinate_mean) / coordinate_std
    ).to(device)
    train_conditions = torch.from_numpy(train_conditions_np).to(device)
    validation_conditions = torch.from_numpy(validation_conditions_np).to(device)
    del arrays

    model = EquivariantMatrixVectorField(
        **config["model"], condition_dim=train_conditions.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    betas = cosine_schedule(int(config["diffusion"]["timesteps"])).to(device)
    cumulative_alphas = torch.cumprod(1 - betas, dim=0)
    prediction_type = str(config["diffusion"].get("prediction", "epsilon"))
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
    validation_timestep = torch.randint(
        cumulative_alphas.numel(),
        (validation_count,),
        device=device,
        generator=torch.Generator(device).manual_seed(seed + 2),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_validation = math.inf
    global_step = 0
    started = time.monotonic()
    stop = False
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
                loss = equivariant_ddpm_loss(
                    model,
                    target,
                    condition,
                    cumulative_alphas,
                    prediction_type=prediction_type,
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
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
                equivariant_ddpm_loss(
                    model,
                    validation_target,
                    validation_condition,
                    cumulative_alphas,
                    prediction_type=prediction_type,
                    noise=validation_noise,
                    timestep=validation_timestep,
                )
            )
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema,
            "model_config": {
                **config["model"],
                "condition_dim": int(train_conditions.shape[1]),
            },
            "architecture": "equivariant",
            "generative_objective": "diffusion",
            "coordinate_transform": "gft",
            "diffusion_config": config["diffusion"],
            "training_config": training,
            "coordinate_mean": coordinate_mean,
            "coordinate_std": coordinate_std,
            "condition_scaler": condition_scaler,
            "epoch": epoch + 1,
            "global_step": global_step,
            "validation_loss": validation_loss,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stop:
            break


if __name__ == "__main__":
    main()
