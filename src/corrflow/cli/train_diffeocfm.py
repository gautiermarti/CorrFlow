"""Train categorical DiffeoCFM using explicit CorrBench train/validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary

from corrflow.diffeocfm import (
    UPSTREAM_REVISION,
    CorrelationGaussian,
    DiffeoVectorField,
    flow_loss,
    sample_correlations,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_validation(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
            for c in np.unique(labels)
        ]
    )


def validate_samples(prior, model, labels, real_features, scale, config):
    matrices, timing = sample_correlations(
        prior,
        model,
        labels,
        seed=config["training"]["validation_seed"],
        **config["sampling"],
    )
    features, _ = structural_features(matrices)
    return {
        "feature_swd": sliced_wasserstein(
            real_features,
            features,
            scale=scale,
            seed=config["training"]["validation_seed"],
            device="cpu",
        ),
        "validity": validity_summary(matrices),
        "sampling": timing,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    if config["conditioning"] != "categorical_only":
        raise ValueError("This adapter supports categorical information only")
    training = config["training"]
    if args.seed is not None:
        training["seed"] = args.seed
    if args.epochs is not None:
        training["epochs"] = args.epochs
    epochs, batch_size = int(training["epochs"]), int(training["batch_size"])
    every = int(training["structural_validation_every"])
    if min(epochs, batch_size, every, int(training["validation_per_class"])) < 1:
        raise ValueError("Training and validation counts must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(training["seed"])
    arrays, hashes = {}, {}
    for split in ["train", "validation"]:
        path = args.data / f"{split}.npz"
        hashes[split] = sha256(path)
        with np.load(path) as a:
            arrays[split] = {k: a[k] for k in ["matrices", "labels"]}
    train, val = arrays["train"], arrays["validation"]
    if batch_size > len(train["labels"]):
        raise ValueError("Batch size exceeds training count")
    if not np.array_equal(np.unique(train["labels"]), np.unique(val["labels"])):
        raise ValueError("Train/validation class sets differ")
    selected = select_validation(
        val["labels"], training["validation_per_class"], training["validation_seed"]
    )
    args.output.mkdir(parents=True)
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    np.save(args.output / "selected_validation_indices.npy", selected)
    provenance = {
        "protocol": config["protocol"],
        "source_hashes": hashes,
        "upstream_revision": UPSTREAM_REVISION,
        "scope": "categorical-only, frozen train/validation; no test data opened",
        "data_rows": {s: len(a["labels"]) for s, a in arrays.items()},
        "code_hashes": {"trainer": sha256(Path(__file__))},
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "selection": "minimum fixed-noise structural validation SWD; ties retain first",
    }
    import corrflow.diffeocfm as implementation

    provenance["code_hashes"]["implementation"] = sha256(Path(implementation.__file__))
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    started = time.perf_counter()
    prior, ztrain = CorrelationGaussian.fit(train["matrices"], train["labels"])
    torch.save(
        {
            "state": prior.state_dict(),
            "matrix_dim": prior.matrix_dim,
            "classes": prior.classes,
            "shrinkage": prior.shrinkage,
        },
        args.output / "prior.pt",
    )
    prior = prior.to(device)
    ztrain = torch.as_tensor(ztrain, device=device)
    zval = prior.transform(val["matrices"][selected])
    ytrain = torch.as_tensor(train["labels"], device=device, dtype=torch.int64)
    yval = torch.as_tensor(val["labels"][selected], device=device, dtype=torch.int64)
    preprocessing_seconds = time.perf_counter() - started
    features, _ = structural_features(train["matrices"])
    scale = np.maximum(features.std(axis=0, ddof=1), 1e-6)
    real_features, _ = structural_features(val["matrices"][selected])
    reference = validate_samples(prior, None, yval, real_features, scale, config)
    reference["preprocessing_seconds"] = preprocessing_seconds
    reference["shrinkage"] = prior.shrinkage
    (args.output / "gaussian_validation.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    print(json.dumps({"gaussian": reference}), flush=True)
    model_config = {
        "vector_dim": ztrain.shape[1],
        "classes": prior.classes,
        "hidden": config["model"]["hidden"],
    }
    model = DiffeoVectorField(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=0
    )
    # The official fMRI factor_lr=1 schedule is constant; retain that behavior.
    train_rng = torch.Generator(device).manual_seed(training["seed"])
    validation_rng = torch.Generator(device).manual_seed(
        training["validation_seed"] + 1
    )
    validation_initial = prior.sample(yval, generator=validation_rng)
    validation_time = torch.rand(
        len(yval), device=device, dtype=torch.float64, generator=validation_rng
    )
    history = []
    best_loss, best_structural = float("inf"), float("inf")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        order = torch.randperm(len(ztrain), device=device, generator=train_rng)
        losses = []
        for start in range(0, len(order) - batch_size + 1, batch_size):
            idx = order[start : start + batch_size]
            labels = ytrain[idx]
            x0 = prior.sample(labels, generator=train_rng)
            times = torch.rand(
                batch_size, device=device, dtype=torch.float64, generator=train_rng
            )
            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model, x0, ztrain[idx], labels, times)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                flow_loss(model, validation_initial, zval, yval, validation_time)
            )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "training_epoch_seconds": time.perf_counter() - epoch_start,
        }
        if epoch == 1 or epoch % every == 0 or epoch == epochs:
            row["structural"] = validate_samples(
                prior, model, yval, real_features, scale, config
            )
        row["elapsed_seconds"] = time.perf_counter() - training_started
        if device.type == "cuda":
            row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        history.append(row)
        checkpoint = {
            "model": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "training_seed": training["seed"],
            "history_row": row,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(checkpoint, args.output / "best_loss.pt")
        score = row.get("structural", {}).get("feature_swd", float("inf"))
        if score < best_structural:
            best_structural = score
            torch.save(checkpoint, args.output / "best_structural.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(row), flush=True)
    final = {
        "epochs_completed": epochs,
        "best_structural_swd": best_structural,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": time.perf_counter() - training_started,
    }
    (args.output / "complete.json").write_text(json.dumps(final, indent=2) + "\n")


if __name__ == "__main__":
    main()
