"""Reproduce each completed budget's native selection before any test scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import corrflow
from corrflow.cli.train_ddpm import conditions, fingerprint
from corrflow.diffusion import cosine_schedule, ddpm_loss
from corrflow.models import MatrixUNetDenoiser


@torch.no_grad()
def audit(run, data, device):
    state = torch.load(
        run / "training_state.pt", map_location=device, weights_only=True
    )
    signature = state["signature"]
    config = signature["config"]
    training = config["training"]
    epochs = int(training["epochs"])
    assert state["checkpoint"]["epoch"] == epochs, "Declared budget is incomplete"
    assert signature == json.loads((run / "run_manifest.json").read_text())
    assert signature["torch"] == str(torch.__version__), "Native Torch version differs"
    assert signature["numpy"] == str(np.__version__), "Native NumPy version differs"
    assert signature["device"] == str(torch.device(device)), "Native device differs"
    torch.set_num_threads(signature["threads"])
    package = Path(corrflow.__file__).parent
    for relative, expected in signature["code_sha256"].items():
        assert fingerprint(package / relative) == expected, relative
    for split, expected in signature["data_sha256"].items():
        assert fingerprint(data / f"{split}.npz") == expected, split
    history = json.loads((run / "history.json").read_text())
    assert history == state["history"]
    assert [row["epoch"] for row in history] == list(range(1, epochs + 1))
    with np.load(data / "train.npz") as source:
        mean, std = float(source["matrices"].mean()), float(source["matrices"].std())
        _, scaler = conditions(source["labels"], source.get("sharpes"))
        steps = math.ceil(len(source["labels"]) / int(training["batch_size"]))
    assert [row["step"] for row in history] == [steps * i for i in range(1, epochs + 1)]
    with np.load(data / "validation.npz") as source:
        validation = torch.from_numpy((source["matrices"] - mean) / std).to(device)
        context, _ = conditions(source["labels"], source.get("sharpes"), scaler)
    context = torch.from_numpy(context).to(device)
    torch.manual_seed(int(training["seed"]))
    model = MatrixUNetDenoiser(**state["checkpoint"]["model_config"]).to(device)
    cumulative = torch.cumprod(
        1 - cosine_schedule(config["diffusion"]["timesteps"]).to(device), dim=0
    )
    indices = torch.randperm(len(validation), device=device)[
        : min(256, len(validation))
    ]
    noise = torch.randn_like(validation[indices])
    timestep = torch.randint(cumulative.numel(), (len(indices),), device=device)
    for key, value in {
        "indices": indices,
        "noise": noise,
        "timestep": timestep,
    }.items():
        assert torch.equal(value, state["validation"][key]), "Validation RNG differs"
    rows = []
    for budget in signature["budget_epochs"]:
        selected = min(history[:budget], key=lambda row: row["validation_loss"])
        for choice, expected in [("best", selected), ("last", history[budget - 1])]:
            path = run / f"budgets/epoch_{budget:04d}/{choice}.pt"
            checkpoint = torch.load(path, weights_only=True, map_location=device)
            assert checkpoint["epoch"] == expected["epoch"]
            assert checkpoint["validation_loss"] == expected["validation_loss"]
            assert checkpoint["matrix_mean"] == mean and checkpoint["matrix_std"] == std
            assert checkpoint["condition_scaler"] == scaler
            model.load_state_dict(checkpoint["model"])
            model.eval()
            with torch.autocast(
                torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=training["precision"] == "bfloat16",
            ):
                value = float(
                    ddpm_loss(
                        model,
                        validation[indices],
                        context[indices],
                        cumulative,
                        noise=noise,
                        timestep=timestep,
                    )
                )
            assert value == expected["validation_loss"], (
                "Native score did not reproduce"
            )
            rows.append(
                {
                    "budget": budget,
                    "choice": choice,
                    "epoch": checkpoint["epoch"],
                    "validation_loss": value,
                    "native_exact": True,
                    "checkpoint_sha256": fingerprint(path),
                }
            )
    return {
        "audit_schema": "corrflow.image-ddpm-native-audit.v1",
        "completed_epochs": epochs,
        "training_seed": training["seed"],
        "native_validation_checks": rows,
        "validation_rng_exact": True,
        "test_data_read": False,
        "training_state_sha256": fingerprint(run / "training_state.pt"),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.run, args.data, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
