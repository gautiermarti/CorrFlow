# %% [markdown]
# # Generate, validate, and compare
# This lesson uses a small pretrained CorrFlow trained only on a synthetic 6D
# mixture. Its three labels are toy regimes, not financial regimes. It demonstrates
# the public sampling API and honest distributional evaluation on CPU.
#
# Run from this tutorials directory. Download/extract the accompanying teaching
# assets, or create them with build_toy.py and the documented training command.
# %%
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from corrbench.metrics import sliced_wasserstein, structural_features, validity_summary

from corrflow.models import EquivariantMatrixVectorField
from corrflow.sampling import sample_correlations

started = time.perf_counter()
torch.set_num_threads(2)
base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
assets = base / "assets"
manifest = json.loads((assets / "manifest.json").read_text())
for name, digest in manifest["sha256"].items():
    assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == digest, name
checkpoint = torch.load(
    assets / "toy_corrflow.pt", map_location="cpu", weights_only=True
)
model = EquivariantMatrixVectorField(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model"])
labels = np.repeat(np.arange(3), 30)
generated, timing = sample_correlations(
    model,
    np.eye(3, dtype=np.float32)[labels],
    matrix_dim=6,
    coordinate_mean=checkpoint["coordinate_mean"],
    coordinate_std=checkpoint["coordinate_std"],
    seed=3106,
    **checkpoint["sampling_config"],
)
print(json.dumps(validity_summary(generated), indent=2))
assert np.linalg.eigvalsh(generated).min() > 0
# %% [markdown]
# Evaluate distributions within each condition. Arbitrarily pairing generated and
# real rows would penalize two independent draws from the same distribution.
# Normalize features using training data only, and include conditional replay.
# %%
with np.load(assets / "train.npz") as source:
    train, train_labels = source["matrices"], source["labels"]
with np.load(assets / "test.npz") as source:
    test, test_labels = source["matrices"], source["labels"]
train_features, feature_names = structural_features(train)
scale = np.maximum(train_features.std(axis=0, ddof=1), 1e-6)
rng = np.random.default_rng(3107)
replay = np.stack(
    [train[rng.choice(np.flatnonzero(train_labels == label))] for label in labels]
)
scores = []
for label in range(3):
    real, _ = structural_features(test[test_labels == label])
    row = {"toy_regime": label}
    for name, samples in [
        ("CorrFlow", generated),
        ("Replay", replay),
        ("Identity", np.tile(np.eye(6), (len(labels), 1, 1))),
    ]:
        features, _ = structural_features(samples[labels == label])
        row[name] = sliced_wasserstein(
            real, features, scale=scale, seed=3108, device="cpu"
        )
    scores.append(row)
print(
    json.dumps(
        {
            "scores": scores,
            "selected_epoch": checkpoint["epoch"],
            "timing": timing,
            "elapsed_seconds": time.perf_counter() - started,
        },
        indent=2,
    )
)
# %% [markdown]
# These are teaching results from one fit of a synthetic model. They are not
# financial benchmark results and do not measure uncertainty across training fits.
# Every method above produces valid matrices; the distributional scores ask a
# different question. The next lesson reconstructs the audited financial table.
