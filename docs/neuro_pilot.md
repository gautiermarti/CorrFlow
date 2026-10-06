# Experimental fixed-atlas CorrFlow pilot

This is new cross-domain research, separate from the financial manuscript model.
The source/data/split/utility protocol is in the sibling CorrBench repository,
`docs/fmri_pilot_protocol.md`. Use the same 111-ROI OAS-regularized ABIDE matrices
for all methods; do not Ward-sort named brain regions.

Source qualification: the frozen mask has 110 named regions and unnamed
two-voxel label 3455, which alone caused 161 of the original 179 exclusions.
The original learned fits remain attached to that 111-label, 601-subject data.
CorrBench's separately declared 110-region controls reconstruct both the same
cohort and 762 eligible scans; neither replay nor Gaussian augmentation gives
an established gain. Learned 110-region models have not been fitted. See
[the follow-up](https://github.com/gautiermarti/corrbench/blob/main/docs/abide_named110_results_20261006.md).

The original global-conditioned equivariant field and exchangeable source
produce an exchangeable matrix law. That is appropriate for anonymous asset
subsets but cannot generally represent a distribution tied to named ROIs.
`NodeConditionedMatrixVectorField` initializes each node with a learned projection
of its attributes. The pilot attributes are fixed one-hot atlas identities.
The field satisfies

\[
F(PXP^\top, PA,t,y)=P F(X,A,t,y)P^\top.
\]

Holding atlas identities fixed therefore permits region-specific dependence.
Tests check simultaneous matrix/attribute relabeling, symmetry, hollow output,
nonzero learning gradients on node attributes, and explicit handling of node
subsets. A shape-compatible subset does not establish learned size transfer or
projective consistency. One-hot identity does not provide unseen-atlas transfer.

Run the declared node-identity and anonymous-node ablation:

```bash
python -m corrflow.cli.train_neuro_flow --data /path/to/abide_ho \
  --config configs/corrflow_abide_ho_pilot.yaml --output artifacts/nodeflow
python -m corrflow.cli.train_neuro_flow --data /path/to/abide_ho \
  --config configs/corrflow_abide_ho_pilot.yaml --output artifacts/anonymous \
  --anonymous
```

Both use 32 hidden channels, two blocks, 20 epochs, batch 16 and seed 1701.
They select the minimum fixed-validation structural SWD, save selected and last
weights, and never open test data. Sampling uses 25 Heun steps, a float32 field
and converged float64 GFT inverse. These CPU pilots are not isolated timing
benchmarks or equal-compute comparisons with DiffeoCFM.

```bash
python -m corrflow.cli.sample_neuro_augmentation --run artifacts/nodeflow \
  --family gft --per-class 175 --seed 2027 --output artifacts/nodeflow_2027.npz
```

The same augmentation interface samples `--family diffeocfm` or `--family
gaussian` from a run trained with `configs/diffeocfm_abide_ho_pilot.yaml`.
Use common counts, generation seeds, classifier tuning and data partitions.
Source checkpoint hashes and timing accompany each generated file. Five latent
seeds from one training run remain one training seed. The downstream comparison
must include real-only and replay, and retain failed or harmful augmentation.

The initial pilot is complete. Real-only test AUC is .7654; the five-draw means
are .7441 for selected DiffeoCFM, .7573 for node-aware CorrFlow and .7628 for
the anonymous ablation. None establishes an augmentation gain. See
[the preserved initial report](https://github.com/gautiermarti/corrbench/blob/main/docs/fmri_pilot_results.md).

The fixed expansion to five training seeds is complete. Selected DiffeoCFM AUC
is .7452 ± .0056, node-aware CorrFlow .7365 ± .0137 and anonymous CorrFlow
.7314 ± .0184; SD is across fits after averaging five draws within each fit.
None establishes improvement over .7654 real-only. See
[the complete replication](https://github.com/gautiermarti/corrbench/blob/main/docs/fmri_replication_results.md).
The same previously inspected NYU site and unchanged configurations are used;
these are not fresh-site results. Dataset redistribution details remain under review.
