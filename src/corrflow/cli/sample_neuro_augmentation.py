"""Generate fixed-atlas, diagnosis-balanced matrices for the ABIDE utility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from corrflow.diffeocfm import CorrelationGaussian, DiffeoVectorField
from corrflow.diffeocfm import sample_correlations as sample_diffeo
from corrflow.models.equivariant import EquivariantMatrixVectorField
from corrflow.models.node_conditioned import NodeConditionedMatrixVectorField
from corrflow.sampling import sample_correlations as sample_gft


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--family", choices=["diffeocfm", "gaussian", "gft"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-class", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--weights", choices=["best_structural", "last"], default="best_structural"
    )
    args = p.parse_args()
    if args.output.suffix != ".npz" or args.per_class < 1:
        p.error("Positive per-class count and .npz output required")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    labels = np.repeat(np.arange(2), args.per_class)
    source_paths = [args.run / "config.yaml", args.run / "provenance.json"]
    config = yaml.safe_load(source_paths[0].read_text())
    epoch = 0
    if args.family in {"diffeocfm", "gaussian"}:
        prior_path = args.run / "prior.pt"
        stored = torch.load(prior_path, map_location="cpu", weights_only=True)
        prior = CorrelationGaussian(
            matrix_dim=stored["matrix_dim"], classes=stored["classes"]
        )
        prior.load_state_dict(stored["state"])
        prior = prior.to(args.device)
        source_paths.append(prior_path)
        model = None
        if args.family == "diffeocfm":
            path = args.run / f"{args.weights}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = DiffeoVectorField(**checkpoint["model_config"]).to(args.device)
            model.load_state_dict(checkpoint["model"])
            epoch = checkpoint["epoch"]
            source_paths.append(path)
        matrices, timing = sample_diffeo(
            prior, model, labels, seed=args.seed, **config["sampling"]
        )
    else:
        path = args.run / f"{args.weights}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        n = checkpoint["matrix_dim"]
        if checkpoint["node_identity"]:
            model = NodeConditionedMatrixVectorField(
                node_features=torch.eye(n), **checkpoint["model_config"]
            )
        else:
            model = EquivariantMatrixVectorField(**checkpoint["model_config"])
        model = model.to(args.device)
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        source_paths.append(path)
        matrices, timing = sample_gft(
            model,
            np.eye(2, dtype=np.float32)[labels],
            matrix_dim=n,
            coordinate_mean=checkpoint["coordinate_mean"],
            coordinate_std=checkpoint["coordinate_std"],
            seed=args.seed,
            **checkpoint["sampling_config"],
        )
    if not np.isfinite(matrices).all() or np.linalg.eigvalsh(matrices).min() <= 0:
        raise ValueError("Invalid generated correlations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, matrices=matrices, labels=labels, seed=args.seed, epoch=epoch
    )
    report = {
        "family": args.family,
        "weights": args.weights,
        "epoch": epoch,
        "generation_seed": args.seed,
        "per_class": args.per_class,
        "training_seed_count": 1,
        "timing": timing,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "epoch": epoch,
                "samples": len(labels),
                "seconds": timing["sampling_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
