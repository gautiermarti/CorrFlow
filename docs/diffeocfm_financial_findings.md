# DiffeoCFM Taiwan diagnostic — 2026-10-05

M2b completes the explicit-validation adapter and one full financial training /
sampling / scoring workflow. The matched multi-model, multi-seed comparison is
still pending. [Protocol](diffeocfm_protocol.md) and
[machine-readable evidence](diffeocfm_financial_seed1701.json) accompany this report.

The adapter matches pinned upstream components to numerical precision. The
200-epoch run uses 13,600 Ward-ordered training matrices, categorical conditions
only, seed 1701 and the full fitted class-wise OAS Gaussian. Selection and scoring
use the same fixed 300 validation rows, with no newly opened test data. This
selection score is optimistic as a generalization estimate. These historical
periods were already inspected and overlapping windows are not independent.

## What the run shows

| Candidate | Structural SWD ↓ | Entry SWD ↓ | Spectral SWD ↓ | Precision ↑ | Recall ↑ | Portfolio SWD ↓ | Classifier AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fitted DiffeoGauss prior | .9612 | 1.0347 | 1.0304 | .4000 | .2767 | .7770 | .9980 |
| DiffeoCFM, selected epoch 170 | .8598 | .9186 | .8945 | .3500 | .4867 | .9570 | .9994 |
| DiffeoCFM, final epoch 200 | .9061 | .9551 | .9531 | .3500 | .5400 | 1.0147 | .9989 |
| Class-conditional training replay | .9205 | .9173 | 1.0869 | .3900 | .6767 | .8255 | .9532 |

All 300 outputs per candidate are positive definite with unit diagonal; no PD
repair is applied. The shared float32 scoring reproduces the training-time
float64 structural scores to the reported precision. Native validity is saved.

The learned flow improves the full score and spectral summaries relative to its
Gaussian starting distribution. That does not establish uniformly better samples:
replay covers more validation structure, and the Gaussian prior has better
portfolio-function distances. Near-perfect two-sample classifier separation
shows substantial remaining mismatch. Even replay is distinguishable, consistent
with the chronological distribution shift; classifier results do not identify a
single cause of that mismatch.

Portfolio diagnostics use unit marginal volatility and unconstrained
minimum-variance functionals; they are not trading performance. Classifier CV
and coverage summarize this fixed matrix subset, not independent date-level
inference. No confidence intervals or statistical superiority claims are made.
Existing CorrFlow/cCorrGAN paper scores use additional continuous information
and must not be inserted into this table as a matched comparison.

## Compute and implementation evidence

- 200-epoch training/validation loop: 442.5 seconds on the measured GB10; fitted
  preprocessing: 2.6 seconds; peak allocated CUDA tensors: .78 GiB. These do not
  include all loading/feature construction and are not end-to-end timings.
- Selected sampling uses 32 actual vector-field evaluations per trajectory in
  each batch; the final checkpoint averages 34.56. Solver: adaptive dopri5,
  atol=rtol=1e-5, batch 64. Output timestamps are not counted as evaluations.
- Per-call timing is recorded, but first-call initialization affects comparisons.
  A repeated warmed timing study is needed before making speed claims.
- Upstream transform/prior differences are at most 2e-15; network, objective
  and gradient checks agree exactly for identical weights and inputs.
- 54 installed-wheel tests across the three packages pass. Lint, wheel/sdist
  builds, installed training/sampling commands and bundled MIT notice pass.
  The environment reuses scientific dependencies; fresh-machine reproduction
  and remote CI validation remain open.

## Resume work

1. Add categorical-only CorrFlow training and fixed structural validation
   selection, preserving the existing paper configuration and final checkpoints.
2. Freeze categorical cCorrGAN configuration and verify its sampler on this same
   validation selection. Preserve the unresolved historical training fidelity.
3. Run the declared five seeds with matched information and common scoring;
   keep Gaussian/replay, both selected/final endpoints and unfavorable diagnostics.
   Report method-specific training and sampling budgets rather than treating
   equal epochs as equal compute.
4. Reconcile the resulting evidence into reviewer responses and manuscript claims.
   A fresh chronological holdout and non-financial studies remain separate work.

Local run: `artifacts/diffeocfm_taiwan_200_seed1701_20261005/`.
The three-epoch pilot is retained separately. Trained weights and matrix artifacts
remain local; source, protocol and compact evidence are committed privately.

To reproduce the four-candidate diagnostic after training:

```bash
python research/evaluate_diffeocfm_run.py \
  --run artifacts/diffeocfm_seed1701 --data /path/to/taiwan_final_ward \
  --output artifacts/seed1701_diagnostics --device cuda
```

This script refuses existing output and verifies train/validation hashes against
the training run. It records sample/checkpoint hashes, metric-source hashes and
package versions, and never accepts a test split.


## Completed five-seed categorical comparison (later October 5 milestone)

The complete matched CorrFlow/DiffeoCFM test study is now audited in
[CorrBench](https://github.com/gautiermarti/corrbench/blob/main/docs/matched_taiwan_results.md).
All ten native validation scores reproduce exactly and all methods use identical
1,500 test cases. Selected structural SWD is .4709 ± .0615 for CorrFlow and
.7512 ± .0188 for DiffeoCFM; portfolio SWD is .6243 and .8620, respectively.
DiffeoCFM has higher recall (.3549 vs .3204), and both are readily separated from
real data. Final-weight structural scores reverse the ordering (.8363 vs .7980).

The original single-seed validation report above remains intact and is not
compared numerically with the original continuous-conditioned paper test score.
This later study uses categorical conditions only, method-specific 20/200-epoch
budgets and previously inspected periods. The full cCorrGAN suite remains
running. No universal method, equal-compute or speed ranking is claimed.
