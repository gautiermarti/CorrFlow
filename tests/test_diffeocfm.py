import numpy as np
import pytest
import torch

from corrflow.diffeocfm import (
    CorrelationGaussian,
    DiffeoVectorField,
    sample_correlations,
)


def data(seed=91):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(60, 30, 4))
    x[:, :, 1] += 0.7 * x[:, :, 0]
    return np.stack([np.corrcoef(a.T) for a in x]), np.tile(np.arange(3), 20)


def test_external_validation_does_not_change_fit_and_inverse_restores_input():
    x, y = data()
    prior, z = CorrelationGaussian.fit(x, y)
    state = {k: v.clone() for k, v in prior.state_dict().items()}
    other, _ = data(100)
    restored = prior.inverse(prior.transform(other)).numpy()
    np.testing.assert_allclose(restored, other, atol=1e-12)
    for k, v in state.items():
        assert torch.equal(v, prior.state_dict()[k])
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-12)
    covariance = (prior.factors[0] @ prior.factors[0].T).numpy()
    assert np.max(np.abs(covariance - np.diag(np.diag(covariance)))) > 1e-3


def test_zero_flow_equals_gaussian_preserves_rng_and_counts_evaluations():
    pytest.importorskip("torchdiffeq")
    x, y = data()
    prior, _ = CorrelationGaussian.fit(x, y)
    model = DiffeoVectorField(vector_dim=6, classes=3, hidden=[8])
    for parameter in model.parameters():
        parameter.data.zero_()
    before = torch.random.get_rng_state().clone()
    gaussian, ginfo = sample_correlations(prior, None, y[:9], seed=17, batch_size=4)
    actual, info = sample_correlations(prior, model, y[:9], seed=17, batch_size=3)
    assert torch.equal(before, torch.random.get_rng_state())
    np.testing.assert_allclose(actual, gaussian, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > 0
    assert ginfo["rhs_calls_total"] == 0
    assert info["rhs_calls_total"] == sum(info["rhs_calls_by_batch"]) > 0
    assert info["mean_rhs_evaluations_per_sample"] == np.average(
        info["rhs_calls_by_batch"], weights=info["batch_sizes"]
    )


def test_prior_rejects_unknown_labels_and_invalid_matrices():
    x, y = data()
    with pytest.raises(ValueError, match="start at zero"):
        CorrelationGaussian.fit(x, y + 1)
    invalid = x.copy()
    invalid[:, 0, 1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        CorrelationGaussian.fit(invalid, y)
    prior, _ = CorrelationGaussian.fit(x, y)
    with pytest.raises(ValueError, match="Unknown class"):
        prior.sample(torch.tensor([3]), generator=torch.Generator().manual_seed(1))


def test_training_uses_only_explicit_train_validation_and_saves_reloadable_model(
    tmp_path, monkeypatch
):
    pytest.importorskip("torchdiffeq")
    import json
    import sys

    import yaml

    from corrflow.cli.train_diffeocfm import main

    x, y = data()
    root = tmp_path / "data"
    root.mkdir()
    for name, a, b in [("train", x[:48], y[:48]), ("validation", x[48:], y[48:])]:
        np.savez(root / f"{name}.npz", matrices=a, labels=b)
    cfg = {
        "protocol": "unit-test",
        "conditioning": "categorical_only",
        "model": {"hidden": [8]},
        "training": {
            "seed": 17,
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.001,
            "structural_validation_every": 1,
            "validation_per_class": 3,
            "validation_seed": 44301,
        },
        "sampling": {"batch_size": 4, "atol": 1e-5, "rtol": 1e-5},
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(root),
            "--config",
            str(config),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    main()
    checkpoint = torch.load(output / "best_structural.pt", weights_only=True)
    assert checkpoint["epoch"] in [1, 2, 3]
    assert (output / "complete.json").exists()
    stored = torch.load(output / "prior.pt", weights_only=True)
    expected, _ = CorrelationGaussian.fit(x[:48], y[:48])
    for key, value in stored["state"].items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        main()

    # Changing how often ODE validation runs must not change learned weights.
    first = torch.load(output / "last.pt", weights_only=True)["model"]
    cfg["training"]["structural_validation_every"] = 2
    config.write_text(yaml.safe_dump(cfg))
    alternate = tmp_path / "run-frequency2"
    sys.argv[sys.argv.index("--output") + 1] = str(alternate)
    main()
    second = torch.load(alternate / "last.pt", weights_only=True)["model"]
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    from corrflow.cli.sample_diffeocfm import main as sample_main

    samples = tmp_path / "samples.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample",
            "--run",
            str(output),
            "--data",
            str(root),
            "--output",
            str(samples),
            "--device",
            "cpu",
            "--per-class",
            "3",
        ],
    )
    sample_main()
    with np.load(samples) as a:
        assert a["split"].item() == "validation"
        assert a["matrices"].shape == (9, 4, 4)
        assert np.linalg.eigvalsh(a["matrices"]).min() > 0
    assert (
        json.loads(samples.with_suffix(".json").read_text())["timing"]["atol"] == 1e-5
    )
