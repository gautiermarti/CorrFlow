#!/usr/bin/env python3
"""Train the conditional raw-matrix image DDPM baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from corrflow.diffusion import cosine_schedule, ddpm_loss
from corrflow.models import MatrixUNetDenoiser


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_state(state, output):
    """Recover derived files from the last atomically completed training epoch."""
    atomic_save(state["checkpoint"], output / "last.pt")
    atomic_save(state["best_checkpoint"], output / "best.pt")
    history = output / "history.json"
    temporary = history.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state["history"], indent=2) + "\n")
    os.replace(temporary, history)
    epoch = state["checkpoint"]["epoch"]
    if epoch in state["signature"]["budget_epochs"]:
        budget = output / "budgets" / f"epoch_{epoch:04d}"
        budget.mkdir(parents=True, exist_ok=True)
        atomic_save(state["checkpoint"], budget / "last.pt")
        atomic_save(state["best_checkpoint"], budget / "best.pt")
        (budget / "selection.json").write_text(
            json.dumps(
                {
                    "budget_epochs": epoch,
                    "selected_epoch": state["best_checkpoint"]["epoch"],
                    "selection": "minimum fixed-noise raw validation denoising loss",
                    "validation_loss": state["best_checkpoint"]["validation_loss"],
                },
                indent=2,
            )
            + "\n"
        )


def conditions(
    labels: np.ndarray,
    sharpes: np.ndarray | None,
    scaler: dict[str, float | None] | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    one_hot = np.eye(3, dtype=np.float32)[labels]
    if sharpes is None:
        return one_hot, {"sharpe_mean": None, "sharpe_std": None}
    if scaler is None:
        scaler = {
            "sharpe_mean": float(sharpes.mean()),
            "sharpe_std": float(sharpes.std()),
        }
    values = (
        (sharpes - float(scaler["sharpe_mean"])) / float(scaler["sharpe_std"])
    ).astype(np.float32)
    return np.column_stack([one_hot, values]), scaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    parser.add_argument("--epochs", type=int, help="Override configured epochs.")
    parser.add_argument("--budget-epochs", type=int, nargs="*", default=[])
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    training = config["training"]
    seed = int(training["seed"] if args.seed is None else args.seed)
    training["seed"] = seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    epochs = int(training["epochs"])
    budgets = sorted(set(args.budget_epochs))
    if epochs < 1 or any(epoch < 1 or epoch > epochs for epoch in budgets):
        raise ValueError("Budget epochs must lie within the positive training budget")
    stop = epochs if args.stop_after_epoch is None else args.stop_after_epoch
    if not 1 <= stop <= epochs:
        raise ValueError("stop-after-epoch must lie within the training budget")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
    stateful = bool(args.save_state or args.resume or budgets)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Refusing to overwrite a run; use --resume")
    source_root = Path(__file__).resolve().parents[1]
    signature = {
        "schema": "corrflow.image-ddpm-training.v1",
        "config": config,
        "budget_epochs": budgets,
        "data_sha256": {
            split: fingerprint(args.data / f"{split}.npz")
            for split in ["train", "validation"]
        },
        "code_sha256": {
            str(path.relative_to(source_root)): fingerprint(path)
            for path in sorted(source_root.rglob("*.py"))
        },
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(torch.device(args.device)),
        "threads": torch.get_num_threads(),
    }
    resumed = None
    if args.resume:
        resumed = torch.load(
            args.output / "training_state.pt",
            weights_only=True,
            map_location=args.device,
        )
        if resumed["signature"] != signature:
            raise ValueError("Resume requires unchanged data, code, config and runtime")
        if resumed["checkpoint"]["epoch"] > stop:
            raise ValueError("Cannot resume to an earlier epoch")
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    arrays = {}
    for split in ("train", "validation"):
        with np.load(args.data / f"{split}.npz") as source:
            arrays[split] = {
                "matrices": source["matrices"],
                "labels": source["labels"],
                "sharpes": source["sharpes"] if "sharpes" in source.files else None,
            }
    mean = float(arrays["train"]["matrices"].mean())
    std = float(arrays["train"]["matrices"].std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training matrices need a finite positive scale")
    train_condition, scaler = conditions(
        arrays["train"]["labels"], arrays["train"]["sharpes"]
    )
    validation_condition, _ = conditions(
        arrays["validation"]["labels"],
        arrays["validation"]["sharpes"],
        scaler,
    )
    if not all(np.isfinite(x).all() for x in [train_condition, validation_condition]):
        raise ValueError("Conditions must be finite with nonconstant score scaling")
    train = torch.from_numpy((arrays["train"]["matrices"] - mean) / std).to(device)
    validation = torch.from_numpy((arrays["validation"]["matrices"] - mean) / std).to(
        device
    )
    train_condition = torch.from_numpy(train_condition).to(device)
    validation_condition = torch.from_numpy(validation_condition).to(device)

    model = MatrixUNetDenoiser(
        **config["model"], condition_dim=train_condition.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    betas = cosine_schedule(int(config["diffusion"]["timesteps"])).to(device)
    cumulative_alphas = torch.cumprod(1 - betas, dim=0)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    precision = training.get("precision", "float32")
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    batch_size = int(training["batch_size"])

    validation_count = min(256, validation.shape[0])
    validation_indices = torch.randperm(validation.shape[0], device=device)[
        :validation_count
    ]
    validation_noise = torch.randn_like(validation[validation_indices])
    validation_timestep = torch.randint(
        cumulative_alphas.numel(), (validation_count,), device=device
    )

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = math.inf
    best_checkpoint = None
    global_step = 0
    first_epoch = 0
    previous_elapsed = 0.0
    if resumed is not None:
        model.load_state_dict(resumed["checkpoint"]["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        ema = resumed["checkpoint"]["ema_model"]
        best_checkpoint = resumed["best_checkpoint"]
        best = best_checkpoint["validation_loss"]
        history = resumed["history"]
        first_epoch = resumed["checkpoint"]["epoch"]
        global_step = history[-1]["step"]
        previous_elapsed = history[-1]["elapsed_seconds"]
        validation_indices = resumed["validation"]["indices"]
        validation_noise = resumed["validation"]["noise"]
        validation_timestep = resumed["validation"]["timestep"]
        torch.random.set_rng_state(resumed["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(resumed["device_rng"].cpu(), device)
        numpy_state = resumed["numpy_rng"]
        np.random.set_state(
            (
                numpy_state[0],
                np.array(numpy_state[1], dtype=np.uint32),
                *numpy_state[2:],
            )
        )
        publish_state(resumed, args.output)
    (args.output / "run_manifest.json").write_text(
        json.dumps(signature, indent=2) + "\n"
    )
    started = time.monotonic()
    for epoch in range(first_epoch, stop):
        if epoch > 0 and args.stop_file is not None and args.stop_file.exists():
            print("Yielding at completed epoch", epoch, flush=True)
            break
        model.train()
        order = torch.randperm(train.shape[0], device=device)
        losses = []
        for start in range(0, train.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=precision != "float32",
            ):
                loss = ddpm_loss(
                    model, train[index], train_condition[index], cumulative_alphas
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            optimizer.step()
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if value.is_floating_point():
                        ema[name].lerp_(
                            value.detach(), 1 - float(training["ema_decay"])
                        )
                    else:
                        ema[name].copy_(value)
            losses.append(float(loss.detach()))
            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"step={global_step} loss={np.mean(losses[-100:]):.6f}",
                    flush=True,
                )

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
                ddpm_loss(
                    model,
                    validation[validation_indices],
                    validation_condition[validation_indices],
                    cumulative_alphas,
                    noise=validation_noise,
                    timestep=validation_timestep,
                )
            )
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite validation loss")
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "ema_model": ema,
            "model_config": {
                **config["model"],
                "condition_dim": int(train_condition.shape[1]),
            },
            "diffusion_config": config["diffusion"],
            "matrix_mean": mean,
            "matrix_std": std,
            "matrix_dim": int(train.shape[-1]),
            "condition_scaler": scaler,
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
        }
        if validation_loss < best:
            best = validation_loss
            best_checkpoint = copy.deepcopy(checkpoint)
        numpy_state = np.random.get_state()
        state = {
            "signature": signature,
            "checkpoint": checkpoint,
            "best_checkpoint": best_checkpoint,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "validation": {
                "indices": validation_indices,
                "noise": validation_noise,
                "timestep": validation_timestep,
            },
            "cpu_rng": torch.random.get_rng_state(),
            "device_rng": torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None,
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        }
        if stateful:
            atomic_save(state, args.output / "training_state.pt")
        publish_state(state, args.output)


if __name__ == "__main__":
    main()
