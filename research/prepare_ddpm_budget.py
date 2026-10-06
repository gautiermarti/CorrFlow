"""Freeze the declared five-seed Taiwan DDPM budget commands without executing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    bench = repo.parent / "corrbench"
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = repo / "configs/ddpm_image.yaml"
    sources = [
        *sorted((repo / "src").rglob("*.py")),
        *sorted((bench / "src").rglob("*.py")),
        config,
        Path(__file__).resolve(),
        repo / "research/run_ddpm_budget.py",
        repo / "docs/ddpm_budget_protocol.md",
    ]
    sources += [data / f"{split}.npz" for split in ["train", "validation"]]
    jobs = []
    for seed in range(1701, 1706):
        destination = output / f"seed{seed}"
        artifacts = [
            destination / "history.json",
            destination / "run_manifest.json",
            destination / "training_state.pt",
        ]
        artifacts += [
            destination / f"budgets/epoch_{epoch:04d}" / file
            for epoch in [20, 50, 100, 200]
            for file in ["best.pt", "last.pt", "selection.json"]
        ]
        jobs.append(
            {
                "name": f"ddpm_taiwan_seed{seed}",
                "cwd": str(repo),
                "output": str(destination),
                "epochs": 200,
                "pilot_epochs": 2,
                "max_pilot_seconds_per_epoch": 120,
                "command": [
                    str(args.python.absolute()),
                    "-m",
                    "corrflow.cli.train_ddpm",
                    "--data",
                    str(data),
                    "--config",
                    str(config),
                    "--output",
                    str(destination),
                    "--device",
                    "cuda",
                    "--threads",
                    "2",
                    "--seed",
                    str(seed),
                    "--epochs",
                    "200",
                    "--budget-epochs",
                    "20",
                    "50",
                    "100",
                    "200",
                ],
                "artifacts": list(map(str, artifacts)),
            }
        )
    plan = {
        "protocol": "taiwan_image_ddpm_budget_v1",
        "scope": "Training only; original continuous condition and frozen splits",
        "test_generation_authorized_by_this_runner": False,
        "environment": {
            "PYTHONPATH": str(repo / "src") + ":" + str(bench / "src"),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        },
        "hashes": {str(path): sha(path) for path in sources},
        "jobs": jobs,
    }
    path = output / "training_plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    (output / "training_plan.sha256").write_text(sha(path) + "  training_plan.json\n")
    print(json.dumps({"plan": str(path), "sha256": sha(path), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
