"""Validation-only common scoring of DiffeoCFM, its exact prior and replay."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from corrbench.cli.evaluate_matrix_samples import distribution_metrics
from corrbench.geometry import corr_to_gft
from corrbench.metrics import (
    portfolio_functionals,
    sliced_wasserstein,
    structural_features,
    validity_summary,
)

from corrflow.diffeocfm import (
    CorrelationGaussian,
    DiffeoVectorField,
    sample_correlations,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.run / "complete.json").exists():
        raise ValueError("Training run is incomplete")
    torch.set_num_threads(4)
    config = yaml.safe_load((args.run / "config.yaml").read_text())
    seed = config["training"]["validation_seed"]
    provenance = json.loads((args.run / "provenance.json").read_text())
    arrays, cache, hashes = {}, {}, {}
    for split in ["train", "validation"]:
        path = args.data / f"{split}.npz"
        hashes[split] = sha256(path)
        if hashes[split] != provenance["source_hashes"][split]:
            raise ValueError(f"Training source changed: {split}")
        with np.load(path) as a:
            arrays[split] = {k: a[k] for k in ["matrices", "labels"]}
        path = args.data / "gft" / f"{split}.npz"
        hashes[f"gft/{split}"] = sha256(path)
        with np.load(path) as a:
            cache[split] = a["gft"]
            if not np.array_equal(a["labels"], arrays[split]["labels"]):
                raise ValueError("Coordinate cache label order differs")
    selected = np.load(args.run / "selected_validation_indices.npy")
    real = arrays["validation"]["matrices"][selected]
    labels = arrays["validation"]["labels"][selected]
    real_gft = cache["validation"][selected]
    train = arrays["train"]["matrices"]
    train_features, _ = structural_features(train)
    scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
    train_portfolio, _ = portfolio_functionals(train)
    portfolio_scale = np.maximum(train_portfolio.std(axis=0, ddof=1), 1e-6)
    train_gft_scale = float(cache["train"].std())
    del cache
    stored = torch.load(args.run / "prior.pt", map_location="cpu", weights_only=True)
    prior = CorrelationGaussian(
        matrix_dim=stored["matrix_dim"], classes=stored["classes"]
    )
    prior.load_state_dict(stored["state"])
    prior = prior.to(args.device)
    args.output.mkdir(parents=True)
    summary = {
        "protocol": config["protocol"],
        "scope": "single training seed, previously inspected validation periods only",
        "split": "validation",
        "training_seed": config["training"]["seed"],
        "sampling_and_metric_seed": seed,
        "evaluation_device": args.device,
        "torch": torch.__version__,
        "samples_per_candidate": len(labels),
        "source_hashes": hashes,
        "run_provenance": provenance,
        "implementation_sha256": sha256(Path(__file__)),
        "benchmark_code_hashes": {
            name: sha256(Path(inspect.getfile(function)))
            for name, function in [
                ("geometry", corr_to_gft),
                ("metrics", structural_features),
                ("common_evaluator", distribution_metrics),
            ]
        },
        "versions": {
            name: version(name)
            for name in ["numpy", "scipy", "scikit-learn", "torch", "torchdiffeq"]
        },
        "config": config,
        "completion": json.loads((args.run / "complete.json").read_text()),
        "prior_sha256": sha256(args.run / "prior.pt"),
        "scoring": (
            "common CorrBench metrics, float32 symmetry/unit-diagonal preparation; "
            "no PD repair"
        ),
        "candidates": {},
    }
    for name in ["gaussian", "best_structural", "last", "replay"]:
        model, epoch, checkpoint_hash = None, 0, None
        if name == "replay":
            rng = np.random.default_rng(seed)
            indices = np.array(
                [
                    rng.choice(np.flatnonzero(arrays["train"]["labels"] == y))
                    for y in labels
                ]
            )
            native = train[indices]
            timing = {"method": "class-conditional replay", "rhs_calls_total": 0}
        else:
            if name != "gaussian":
                path = args.run / f"{name}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                model = DiffeoVectorField(**checkpoint["model_config"]).to(args.device)
                model.load_state_dict(checkpoint["model"])
                epoch = checkpoint["epoch"]
                checkpoint_hash = sha256(path)
            native, timing = sample_correlations(
                prior, model, labels, seed=seed, **config["sampling"]
            )
        # Match the frozen common CLI's scored representation; retain native validity.
        raw = native.astype(np.float32)
        generated = (raw + raw.swapaxes(-1, -2)) * 0.5
        diagonal = np.arange(generated.shape[-1])
        generated[:, diagonal, diagonal] = 1
        with torch.no_grad():
            generated_gft = (
                corr_to_gft(torch.from_numpy(generated).to(args.device)).cpu().numpy()
            )
        report = {
            "epoch": epoch,
            "checkpoint_sha256": checkpoint_hash,
            "native_validity": validity_summary(native),
            "sampling": timing,
            "by_class": {},
        }
        for label in [None, *np.unique(labels)]:
            mask = (
                np.ones(len(labels), dtype=bool) if label is None else labels == label
            )
            metrics = distribution_metrics(
                real[mask],
                generated[mask],
                real_gft[mask],
                generated_gft[mask],
                train_gft_scale=train_gft_scale,
                train_features=train_features,
                feature_scale=scale,
                train_portfolio=train_portfolio,
                portfolio_scale=portfolio_scale,
                seed=seed if label is None else seed + int(label) + 1,
                include_diagnostics=label is None,
                device=args.device,
            )
            if label is None:
                rf, _ = structural_features(real)
                gf, _ = structural_features(generated)
                metrics["entry_swd"] = sliced_wasserstein(
                    rf[:, :9], gf[:, :9], scale=scale[:9], seed=seed, device="cpu"
                )
                metrics["spectral_swd"] = sliced_wasserstein(
                    rf[:, 9:], gf[:, 9:], scale=scale[9:], seed=seed, device="cpu"
                )
                report["pooled"] = metrics
            else:
                report["by_class"][str(label)] = metrics
        sample_path = args.output / f"{name}_samples.npz"
        np.savez_compressed(
            sample_path,
            matrices=native,
            labels=labels,
            selected_test_indices=selected,
            split="validation",
        )
        report["samples_sha256"] = sha256(sample_path)
        (args.output / f"{name}_metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        summary["candidates"][name] = {
            k: v for k, v in report.items() if k != "by_class"
        }
        print(
            json.dumps(
                {
                    "candidate": name,
                    "epoch": epoch,
                    "feature_swd": report["pooled"]["feature_sliced_wasserstein"],
                }
            ),
            flush=True,
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
