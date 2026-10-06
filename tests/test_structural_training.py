import json
import sys

import numpy as np
import pytest
import torch
import yaml
from corrbench.geometry import corr_to_gft

from corrflow.cli.sample_flow import main as sample_main
from corrflow.cli.train_flow import main as train_main
from corrflow.models import EquivariantMatrixVectorField
from corrflow.sampling import sample_correlations


def test_sampling_preserves_rng_counts_calls_and_requires_convergence():
    torch.set_num_threads(2)
    model = EquivariantMatrixVectorField(hidden_dim=4, depth=1, time_frequencies=2)
    before = torch.random.get_rng_state().clone()
    matrices, info = sample_correlations(
        model,
        np.eye(3, dtype=np.float32),
        matrix_dim=4,
        coordinate_mean=0,
        coordinate_std=0.2,
        seed=17,
        steps=2,
        precision="float32",
        batch_size=2,
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert np.linalg.eigvalsh(matrices).min() > 0
    assert info["rhs_calls_by_batch"] == [4, 4]
    assert info["mean_rhs_evaluations_per_sample"] == 4
    with pytest.raises(RuntimeError, match="did not converge"):
        sample_correlations(
            model,
            np.eye(3, dtype=np.float32),
            matrix_dim=4,
            coordinate_mean=0,
            coordinate_std=0.2,
            seed=17,
            steps=2,
            precision="float32",
            inverse_max_iterations=1,
        )


def test_categorical_training_ignores_scores_and_validation_preserves_weights(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(36, 30, 4))
    matrices = np.stack([np.corrcoef(a.T) for a in x]).astype(np.float32)
    labels = np.tile(np.arange(3), 12)
    coordinates = corr_to_gft(matrices).astype(np.float32)
    root = tmp_path / "data"
    (root / "gft").mkdir(parents=True)
    for split, indices in [("train", slice(0, 24)), ("validation", slice(24, 36))]:
        common = {
            "labels": labels[indices],
            "sharpes": np.full(len(labels[indices]), np.nan),
        }
        np.savez(root / f"{split}.npz", matrices=matrices[indices], **common)
        np.savez(root / "gft" / f"{split}.npz", gft=coordinates[indices], **common)
    config = {
        "conditioning": "categorical_only",
        "protocol": "test",
        "model": {"hidden_dim": 4, "depth": 1, "time_frequencies": 2},
        "training": {
            "seed": 17,
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0,
            "ema_decay": 0.9,
            "gradient_clip": 1,
            "precision": "float32",
            "structural_validation_every": 1,
            "validation_per_class": 2,
            "validation_seed": 44301,
        },
        "sampling": {"steps": 2, "batch_size": 3, "precision": "float32"},
    }
    path = tmp_path / "config.yaml"
    states = []
    for every in [1, 2]:
        config["training"]["structural_validation_every"] = every
        path.write_text(yaml.safe_dump(config))
        output = tmp_path / f"run{every}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data",
                str(root),
                "--config",
                str(path),
                "--output",
                str(output),
                "--device",
                "cpu",
            ],
        )
        train_main()
        last = torch.load(output / "last.pt", weights_only=True)
        states.append(last["model"])
        assert last["model_config"]["condition_dim"] == 3
        assert last["condition_scaler"]["sharpe_mean"] is None
        assert (
            json.loads((output / "complete.json").read_text())["epochs_completed"] == 3
        )
        with pytest.raises(FileExistsError):
            train_main()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    output = tmp_path / "samples.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample",
            "--data",
            str(root),
            "--checkpoint",
            str(tmp_path / "run1" / "best_structural.pt"),
            "--output",
            str(output),
            "--per-class",
            "2",
            "--device",
            "cpu",
        ],
    )
    sample_main()
    with np.load(output) as a:
        assert a["split"].item() == "validation"
        assert a["matrices"].shape == (6, 4, 4)
        assert np.linalg.eigvalsh(a["matrices"]).min() > 0
