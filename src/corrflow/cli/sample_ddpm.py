#!/usr/bin/env python3
"""Generate raw matrix samples from the image DDPM baseline."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from corrflow.diffusion import cosine_schedule, sample_ddim
from corrflow.models import MatrixUNetDenoiser


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--samples-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=True
    )
    model = MatrixUNetDenoiser(**checkpoint["model_config"]).to(args.device)
    state = checkpoint["ema_model"] if args.weights == "ema" else checkpoint["model"]
    model.load_state_dict(state)
    model.eval()
    selection_seed = args.seed if args.selection_seed is None else args.selection_seed
    rng = np.random.default_rng(selection_seed)
    selected = None
    sharpes = None
    with np.load(args.data / f"{args.split}.npz") as source:
        candidate_labels = source["labels"]
        matrix_dim = int(source["matrices"].shape[-1])
        if checkpoint.get("matrix_dim", matrix_dim) != matrix_dim:
            raise ValueError("Checkpoint and evaluation matrix dimensions differ")
        if matrix_dim % 8:
            raise ValueError("The image U-Net requires dimensions divisible by eight")
        selected = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(candidate_labels == label),
                    args.samples_per_class,
                    replace=False,
                )
                for label in range(3)
            ]
        )
        labels = candidate_labels[selected].astype(np.uint8)
        if "sharpes" in source.files:
            sharpes = source["sharpes"][selected]
    condition = np.eye(3, dtype=np.float32)[labels]
    if sharpes is not None:
        scaler = checkpoint["condition_scaler"]
        standardized = (
            (sharpes - scaler["sharpe_mean"]) / scaler["sharpe_std"]
        ).astype(np.float32)
        condition = np.column_stack([condition, standardized])

    timesteps = int(checkpoint["diffusion_config"]["timesteps"])
    cumulative_alphas = torch.cumprod(
        1 - cosine_schedule(timesteps).to(args.device), dim=0
    )
    generated = []
    generator = torch.Generator(args.device).manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, labels.size, args.batch_size):
        batch_condition = torch.from_numpy(
            condition[start : start + args.batch_size]
        ).to(args.device)
        with torch.autocast(
            device_type=torch.device(args.device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(args.device).type == "cuda",
        ):
            normalized = sample_ddim(
                model,
                batch_condition,
                cumulative_alphas,
                matrix_dim=matrix_dim,
                steps=args.steps,
                generator=generator,
            )
        generated.append(
            (normalized.float() * checkpoint["matrix_std"] + checkpoint["matrix_mean"])
            .cpu()
            .numpy()
        )
    matrices = np.concatenate(generated).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrices": matrices,
        "labels": labels,
        "selected_test_indices": selected,
        "split": np.asarray(args.split),
        "sampling_seconds": np.asarray(time.perf_counter() - started),
        "generation_seed": np.asarray(args.seed),
        "selection_seed": np.asarray(selection_seed),
        "weights": np.asarray(args.weights),
        "checkpoint_sha256": np.asarray(
            hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        ),
    }
    if sharpes is not None:
        payload["sharpes"] = sharpes
    np.savez_compressed(args.output, **payload)
    print(f"wrote {matrices.shape} to {args.output}")


if __name__ == "__main__":
    main()
