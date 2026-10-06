"""Sample a fitted DiffeoCFM or its Gaussian prior for common CorrBench scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from corrflow.cli.train_diffeocfm import select_validation
from corrflow.diffeocfm import (
    CorrelationGaussian,
    DiffeoVectorField,
    sample_correlations,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument(
        "--weights",
        choices=["best_structural", "best_loss", "last", "gaussian"],
        default="best_structural",
    )
    p.add_argument("--per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=44301)
    p.add_argument("--selection-seed", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    torch.set_num_threads(4)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    sampling = {**config["sampling"], "batch_size": args.batch_size}
    stored = torch.load(args.run / "prior.pt", map_location="cpu", weights_only=True)
    prior = CorrelationGaussian(
        matrix_dim=stored["matrix_dim"], classes=stored["classes"]
    )
    prior.load_state_dict(stored["state"])
    prior = prior.to(args.device)
    model, epoch = None, 0
    if args.weights != "gaussian":
        checkpoint = torch.load(
            args.run / f"{args.weights}.pt", map_location="cpu", weights_only=True
        )
        model = DiffeoVectorField(**checkpoint["model_config"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
    with np.load(args.data / f"{args.split}.npz") as a:
        selection_seed = (
            args.seed if args.selection_seed is None else args.selection_seed
        )
        indices = select_validation(a["labels"], args.per_class, selection_seed)
        labels = a["labels"][indices].astype(np.int64)
    matrices, timing = sample_correlations(
        prior, model, labels, seed=args.seed, **sampling
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        matrices=matrices,
        labels=labels,
        selected_test_indices=indices,
        split=args.split,
        checkpoint_epoch=epoch,
        sampling_seconds=timing["sampling_seconds"],
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "weights": args.weights,
                "epoch": epoch,
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
