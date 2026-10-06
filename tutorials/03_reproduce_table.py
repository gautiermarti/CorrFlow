# %% [markdown]
# # Reproduce the matched financial table from audited per-seed evidence
# This inexpensive path recomputes reported means and sample standard deviations.
# It is artifact-level reproduction. Full training and matrix-level rescoring are
# separate procedures with their own data, checkpoint and hardware requirements.
# %%
import hashlib
import json
from pathlib import Path

import numpy as np

base = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
manifest = json.loads((base / "assets/manifest.json").read_text())
source = base / "assets/matched_taiwan_results.json"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"][source.name]
)
result = json.loads(source.read_text())
assert all(state["complete"] for state in result["states"].values())
checks = result["validation_selection_checks"]
assert len(checks) == 15
assert all(
    row["training_validation_swd"] == row["reproduced_native_validation_swd"]
    for row in checks
)
print("Method | weights | structural SWD mean ± sample SD | recall | portfolio SWD")
for summary in result["summary"]:
    rows = [
        row
        for row in result["rows"]
        if (row["method"], row["weights"]) == (summary["method"], summary["weights"])
    ]
    assert len(rows) == 5
    for metric, expected in summary["metrics"].items():
        values = np.array([row[metric] for row in rows])
        np.testing.assert_allclose(
            [values.mean(), values.std(ddof=1)],
            [expected["mean"], expected["sd"]],
            atol=1e-12,
            rtol=0,
        )
    metrics = summary["metrics"]
    score = metrics["feature_swd"]
    print(
        f"{summary['method']} | {summary['weights']} | "
        f"{score['mean']:.4f} ± {score['sd']:.4f} | "
        f"{metrics['recall']['mean']:.4f} | "
        f"{metrics['portfolio_swd']['mean']:.4f}"
    )
# %% [markdown]
# Keep final weights and unfavorable results. The selected flow ordering reverses
# at final weights; replay covers more structure; repaired cCorrGAN has poor
# fidelity. The models had method-specific 20/200/200-epoch budgets. These periods
# were inspected during development. Seed SD is not uncertainty over future markets.
