"""Create a deterministic, openly reproducible synthetic teaching dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from corrbench.geometry import gft_to_corr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "gft").mkdir()
    hashes = {}
    for split, seed, per_class in [
        ("train", 3101, 120),
        ("validation", 3102, 30),
        ("test", 3103, 30),
    ]:
        rng = np.random.default_rng(seed)
        labels = np.repeat(np.arange(3), per_class)
        gamma = (
            rng.normal(0.0, 0.10, (len(labels), 15))
            + np.array([-0.08, 0.10, 0.30])[labels, None]
        )
        matrices, info = gft_to_corr(
            gamma, tolerance=1e-10, max_iterations=1000, return_info=True
        )
        assert info["converged"].all()
        np.savez_compressed(
            args.output / f"{split}.npz", matrices=matrices, labels=labels
        )
        np.savez_compressed(
            args.output / "gft" / f"{split}.npz",
            gft=gamma.astype(np.float32),
            labels=labels,
        )
    for path in sorted(args.output.rglob("*.npz")):
        hashes[str(path.relative_to(args.output))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "name": "Synthetic 6D GFT teaching mixture",
                "license": "CC0-1.0",
                "scope": (
                    "Independent synthetic draws; no market or participant data; "
                    "no empirical paper claim"
                ),
                "construction": (
                    "15 independent Normal(class_mean, .10^2) GFT coordinates; "
                    "means [-.08, .10, .30]"
                ),
                "split_seeds": {"train": 3101, "validation": 3102, "test": 3103},
                "sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print("Created synthetic train/validation/test splits:", args.output)


if __name__ == "__main__":
    main()
