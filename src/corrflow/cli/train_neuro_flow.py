"""Experimental fixed-atlas CorrFlow with optional explicit ROI identity features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.geometry import corr_to_gft, lower_to_symmetric
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary

from corrflow.cli.train_diffeocfm import select_validation
from corrflow.flow import rectified_flow_loss
from corrflow.models.equivariant import EquivariantMatrixVectorField
from corrflow.models.node_conditioned import NodeConditionedMatrixVectorField
from corrflow.sampling import sample_correlations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--anonymous", action="store_true", help="Ablate node identity")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    cfg = config["training"]
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    arrays = {}
    for split in ["train", "validation"]:
        with np.load(args.data / f"{split}.npz") as a:
            arrays[split] = {
                k: a[k] for k in ["matrices", "labels", "node_ids", "subject_ids"]
            }
    train, val = arrays["train"], arrays["validation"]
    if not np.array_equal(train["node_ids"], val["node_ids"]):
        raise ValueError("Atlas IDs/order differ")
    if set(train["subject_ids"]) & set(val["subject_ids"]):
        raise ValueError("Subject overlap")
    n = len(train["node_ids"])
    if not np.array_equal(np.unique(train["labels"]), [0, 1]):
        raise ValueError("This pilot requires binary diagnosis labels")
    selected = select_validation(
        val["labels"], cfg["validation_per_class"], cfg["validation_seed"]
    )
    args.output.mkdir(parents=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config))
    np.save(args.output / "selected_validation_indices.npy", selected)
    import corrflow.flow as flow_module
    import corrflow.models.equivariant as base_module
    import corrflow.models.node_conditioned as node_module
    import corrflow.sampling as sampling_module

    provenance = {
        "scope": "Experimental fixed-atlas pilot; not original paper CorrFlow",
        "node_attributes": "none" if args.anonymous else "one-hot fixed atlas identity",
        "limitation": "Identity features do not establish transfer to unseen atlases",
        "source_sha256": {s: sha(args.data / f"{s}.npz") for s in arrays},
        "code_sha256": {
            "trainer": sha(__file__),
            **{
                m.__name__: sha(m.__file__)
                for m in [flow_module, base_module, node_module, sampling_module]
            },
        },
        "torch": torch.__version__,
        "device": args.device,
        "test_opened": False,
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    started = time.perf_counter()
    vectors = corr_to_gft(torch.from_numpy(train["matrices"]).double()).float()
    mean, std = float(vectors.mean()), float(vectors.std(unbiased=False))
    if std <= 0:
        raise ValueError("Degenerate coordinates")
    target = lower_to_symmetric((vectors - mean) / std).to(args.device)
    conditions = (
        torch.nn.functional.one_hot(torch.from_numpy(train["labels"]), 2)
        .float()
        .to(args.device)
    )
    val_conditions = np.eye(2, dtype=np.float32)[val["labels"][selected]]
    if args.anonymous:
        model = EquivariantMatrixVectorField(**config["model"]).to(args.device)
    else:
        model = NodeConditionedMatrixVectorField(
            node_features=torch.eye(n), **config["model"]
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    features, _ = structural_features(train["matrices"])
    scale = np.maximum(features.std(0, ddof=1), 1e-6)
    real_features, _ = structural_features(val["matrices"][selected])
    history, best = [], float("inf")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(len(target), device=args.device)
        losses = []
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start : start + cfg["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = rectified_flow_loss(model, target[idx], conditions[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite flow loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        matrices, timing = sample_correlations(
            model,
            val_conditions,
            matrix_dim=n,
            coordinate_mean=mean,
            coordinate_std=std,
            seed=cfg["validation_seed"],
            **config["sampling"],
        )
        generated_features, _ = structural_features(matrices)
        score = sliced_wasserstein(
            real_features,
            generated_features,
            scale=scale,
            seed=cfg["validation_seed"],
            device="cpu",
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_feature_swd": score,
            "sampling": timing,
            "validity": validity_summary(matrices),
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        checkpoint = {
            "model": model.state_dict(),
            "model_config": config["model"],
            "node_identity": not args.anonymous,
            "node_ids": torch.from_numpy(train["node_ids"]),
            "matrix_dim": n,
            "coordinate_mean": mean,
            "coordinate_std": std,
            "epoch": epoch,
            "history_row": row,
            "sampling_config": config["sampling"],
        }
        torch.save(checkpoint, args.output / "last.pt")
        if score < best:
            best = score
            torch.save(checkpoint, args.output / "best_structural.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row), flush=True)
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "epochs": len(history),
                "best_validation_swd": best,
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
