# CorrFlow runnable lessons

Three short lessons teach correlation geometry, pretrained generation, and
reproduction of the matched financial table. They work on CPU. The sampling
checkpoint is trained on a synthetic six-variable mixture; it is not a model
of financial data. No financial or participant matrices are bundled.

Install compatible CorrBench and CorrFlow release wheels together in a Python
3.12+ environment. For a source checkout, install the sibling CorrBench first,
then CorrFlow. The published release manifest identifies the exact pair.

```bash
python -m pip install ../corrbench .
python -m pip install nbformat nbclient ipykernel
cd tutorials
python 01_geometry.py
python 02_generate.py
python 03_reproduce_table.py
```

Equivalent notebooks sit alongside each script. Open them with a Python kernel
from the same environment and start in this directory. The asset manifest checks
the checkpoint, synthetic splits and report hashes. Only load trusted checkpoints;
the lesson uses PyTorch's restricted weights-only loader.

## What each lesson establishes

1. Geometry: an impossible variance, a converged GFT inverse, bivariate Fisher
   agreement, and consistency when both matrix labels and weights are permuted.
2. Generation: actual pretrained sampling, validity checks, condition-specific
   distributional evaluation, and conditional replay/identity controls. One toy
   fit does not establish empirical financial fidelity or a universal ranking.
3. Results: every matched financial mean and sample SD recomputed from all five
   saved per-seed rows, including final weights and poor cCorrGAN outcomes.
   This is artifact-level reproduction, not retraining or matrix-level rescoring.

The execution audit records measured notebook wall time, including kernel startup,
on the validation machine. Those measurements exclude package installation and
are not guarantees for other computers. The teaching model trains on CPU in a
small fixed budget; the financial paper's training requirements are separate.

## Rebuild the teaching checkpoint

Run from the CorrFlow repository root, choosing new output directories:

```bash
python tutorials/build_toy.py --output artifacts/tutorial_data
corrflow-train --data artifacts/tutorial_data --config tutorials/toy_config.yaml \
  --output artifacts/tutorial_training --device cpu
```

All three split seeds and the synthetic construction appear in the data manifest.
The model runs 40 epochs; raw weights are selected by fixed-noise validation
structural SWD. Keep the complete history. Test data do not select the checkpoint.
Exact trained weights can vary with the platform and dependency versions.

The synthetic matrices are CC0-1.0; tutorial code and the synthetic checkpoint are
MIT licensed. Financial report figures retain the paper's attribution to Gautier
Marti and the explicitly named baseline implementations.
