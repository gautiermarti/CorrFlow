"""Save CorrFlow samples for common CorrBench scoring on an explicit split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrflow.cli.train_diffeocfm import select_validation
from corrflow.cli.train_flow import make_conditions
from corrflow.models import EquivariantMatrixVectorField, MLPMatrixVectorField
from corrflow.sampling import sample_correlations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=44301)
    p.add_argument("--selection-seed", type=int)
    p.add_argument("--weights", choices=["raw", "ema"], default="raw")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_class = (
        EquivariantMatrixVectorField
        if checkpoint.get("architecture", "equivariant") == "equivariant"
        else MLPMatrixVectorField
    )
    model = model_class(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model" if args.weights == "raw" else "ema_model"])
    with np.load(args.data / f"{args.split}.npz") as a:
        selection_seed = (
            args.seed if args.selection_seed is None else args.selection_seed
        )
        selected = select_validation(a["labels"], args.per_class, selection_seed)
        labels = a["labels"][selected]
        matrix_dim = a["matrices"].shape[-1]
        sharpes = (
            a["sharpes"][selected]
            if checkpoint["condition_scaler"]["sharpe_mean"] is not None
            else None
        )
    conditions, _ = make_conditions(labels, sharpes, **checkpoint["condition_scaler"])
    config = dict(checkpoint.get("sampling_config", {}))
    if args.steps is not None:
        config["steps"] = args.steps
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    matrices, timing = sample_correlations(
        model,
        conditions,
        matrix_dim=matrix_dim,
        coordinate_mean=checkpoint["coordinate_mean"],
        coordinate_std=checkpoint["coordinate_std"],
        seed=args.seed,
        coordinate_transform=checkpoint.get("coordinate_transform", "gft"),
        **config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=matrices,
        labels=labels,
        selected_test_indices=selected,
        split=args.split,
        checkpoint_epoch=checkpoint["epoch"],
        sampling_seconds=timing["sampling_seconds"],
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "epoch": checkpoint["epoch"],
                "weights": args.weights,
                "seed": args.seed,
                "selection_seed": selection_seed,
                "split": args.split,
                "samples": len(labels),
                "timing": timing,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(labels)} {args.split} samples to {args.output}")


if __name__ == "__main__":
    main()
