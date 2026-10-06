import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from corrflow.cli.sample_ddpm import main as sample_main
from corrflow.cli.train_ddpm import main


def test_budget_checkpoints_resume_exactly_and_refuse_changed_data(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    rng = np.random.default_rng(19)
    data = tmp_path / "data"
    data.mkdir()
    for split, count in [("train", 12), ("validation", 6)]:
        observations = rng.normal(size=(count, 24, 8))
        matrices = np.stack([np.corrcoef(x.T) for x in observations]).astype("float32")
        np.savez(
            data / f"{split}.npz",
            matrices=matrices,
            labels=np.arange(count) % 3,
            sharpes=np.linspace(-1, 1, count),
        )
    # A trainer must not open test data at any point, including resume.
    (data / "test.npz").write_bytes(b"test data must remain unopened")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"base_dim": 8, "time_frequencies": 2},
                "diffusion": {"timesteps": 12},
                "training": {
                    "seed": 19,
                    "epochs": 4,
                    "batch_size": 4,
                    "learning_rate": 0.0002,
                    "weight_decay": 0.0001,
                    "ema_decay": 0.995,
                    "gradient_clip": 1,
                    "precision": "float32",
                },
            }
        )
    )

    def run(output, *extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--data",
                str(data),
                "--config",
                str(config),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--threads",
                "2",
                "--budget-epochs",
                "2",
                "4",
                *extra,
            ],
        )
        main()

    complete, resumed = tmp_path / "complete", tmp_path / "resumed"
    run(complete)
    run(resumed, "--stop-after-epoch", "2")
    # Simulate interruption between the canonical state and derived file writes.
    (resumed / "last.pt").write_bytes(b"interrupted derived write")
    (resumed / "history.json").unlink()
    run(resumed, "--resume")
    for file in [
        "best.pt",
        "last.pt",
        "budgets/epoch_0002/best.pt",
        "budgets/epoch_0002/last.pt",
        "budgets/epoch_0004/best.pt",
    ]:
        left = torch.load(complete / file, weights_only=True)
        right = torch.load(resumed / file, weights_only=True)
        assert left["epoch"] == right["epoch"]
        assert left["validation_loss"] == right["validation_loss"]
        for weights in ["model", "ema_model"]:
            assert all(
                torch.equal(x, right[weights][k]) for k, x in left[weights].items()
            )
    left = json.loads((complete / "history.json").read_text())
    right = json.loads((resumed / "history.json").read_text())
    assert [{k: v for k, v in r.items() if k != "elapsed_seconds"} for r in left] == [
        {k: v for k, v in r.items() if k != "elapsed_seconds"} for r in right
    ]
    for epoch in [2, 4]:
        selected = json.loads(
            (resumed / f"budgets/epoch_{epoch:04d}/selection.json").read_text()
        )
        expected = min(right[:epoch], key=lambda r: r["validation_loss"])
        assert selected["selected_epoch"] == expected["epoch"]
        assert selected["validation_loss"] == expected["validation_loss"]
    samples = tmp_path / "samples.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample",
            "--data",
            str(data),
            "--checkpoint",
            str(resumed / "best.pt"),
            "--output",
            str(samples),
            "--split",
            "validation",
            "--samples-per-class",
            "2",
            "--steps",
            "3",
            "--weights",
            "raw",
            "--device",
            "cpu",
        ],
    )
    sample_main()
    with np.load(samples) as result:
        assert result["matrices"].shape == (6, 8, 8)
        assert np.isfinite(result["matrices"]).all()
        assert str(result["split"]) == "validation"
    generated = []
    for seed in [101, 102]:
        path = tmp_path / f"sample_{seed}.npz"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sample",
                "--data",
                str(data),
                "--checkpoint",
                str(resumed / "best.pt"),
                "--output",
                str(path),
                "--split",
                "validation",
                "--samples-per-class",
                "2",
                "--steps",
                "3",
                "--device",
                "cpu",
                "--seed",
                str(seed),
                "--selection-seed",
                "77",
            ],
        )
        sample_main()
        with np.load(path) as result:
            generated.append({k: result[k] for k in result.files})
    np.testing.assert_array_equal(
        generated[0]["selected_test_indices"], generated[1]["selected_test_indices"]
    )
    assert not np.array_equal(generated[0]["matrices"], generated[1]["matrices"])
    auditor_path = Path(__file__).parents[1] / "research/audit_ddpm_budget.py"
    spec = importlib.util.spec_from_file_location("ddpm_auditor", auditor_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    report = auditor.audit(resumed, data, "cpu")
    assert report["validation_rng_exact"]
    assert len(report["native_validation_checks"]) == 4
    damaged = resumed / "budgets/epoch_0002/best.pt"
    checkpoint = torch.load(damaged, weights_only=True)
    checkpoint["model"]["output.2.bias"] += 0.125
    torch.save(checkpoint, damaged)
    with pytest.raises(AssertionError, match="Native score did not reproduce"):
        auditor.audit(resumed, data, "cpu")
    with pytest.raises(FileExistsError, match="overwrite"):
        run(resumed)
    with (data / "train.npz").open("ab") as handle:
        handle.write(b"changed source bytes")
    with pytest.raises(ValueError, match="unchanged data"):
        run(resumed, "--resume")
