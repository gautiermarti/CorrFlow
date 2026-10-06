# Categorical DiffeoCFM financial comparison

Protocol `taiwan_categorical_validation_v1`, 2026-10-05. This is an explicitly
adapted financial experiment, not a reproduction of the upstream fMRI results.
The implementation follows the MIT correlation path at official revision
`902282acd13a67784e2096d09ee93477b268c079`; see the third-party notice.

The first 200-epoch run and common baseline diagnostics are complete; see
[financial findings](diffeocfm_financial_findings.md). The full matched suite is pending.

## Data and information

Use frozen Taiwan return-derived Ward-ordered 80-by-80 matrices: 13,600 training
and 3,920 validation rows. The source hashes and implementation hashes are saved
in each run. No test file is opened in training or the validation pilot. These
periods have been inspected in earlier research and contain dependent overlapping
windows; this study cannot establish performance on a fresh holdout.

Only the three categorical regimes are supplied. The existing Taiwan paper runs
also condition on continuous performance information; their numbers are not a
matched comparison with this categorical-only model. A common comparison still
requires categorical-only cCorrGAN and CorrFlow runs, plus class-wise replay.

## Model and selection

- Normalized-Cholesky coordinates, coordinate-wise training StandardScaler and
  a separate full OAS covariance Gaussian in each class. No diagonal approximation.
  All preprocessing/prior parameters are fitted on the explicit training split.
- Float64 SELU MLP with one 512-wide hidden layer; classic flow matching, sigma=0;
  AdamW at .001, zero weight decay, batch 64, drop incomplete training batches.
  The constant learning rate matches the upstream fMRI factor_lr=1 configuration.
- 200 epochs. Planned training seeds: 1701, 1702, 1703, 1704, 1705. First complete
  seed 1701 establishes the end-to-end workflow before a multi-model seed suite.
- Fixed validation subset: 100 rows per class, NumPy default_rng seed 44301.
  Full structural-feature SWD uses training standard deviations and projection
  seed 44301. Evaluate epoch 1, each multiple of 10, and the final epoch.
  Choose minimum SWD, retaining the first checkpoint on a tie.
- Save `last.pt`, `best_loss.pt` and `best_structural.pt` separately. The upstream
  implementation retains the last epoch; report that endpoint as well. Selecting
  structural fidelity is a benchmark adaptation, not upstream behavior.
- Validation noise/time have separate fixed generators. Repeating validation
  cannot consume training randomness. Torch prior draws have the same Gaussian
  law as upstream, but do not reproduce NumPy RandomState sample sequences.
  CPU and CUDA draws also differ for the same seed; record the sampling device.
- Sample the fitted Gaussian as a distinct baseline using the same initial
  draws as the learned flow. This measures what the flow adds to its own prior.

Adaptive dopri5 uses atol=rtol=1e-5 and min_step=1e-5. Count actual vector-field
calls per batch and report a sample-weighted mean, sampling wall time and inverse
time. Output times are not function-evaluation counts. Sampling batch size is 64;
adaptive error control can make finite-tolerance trajectories batch-dependent.

## Validation and interpretation

Four functional tests cover train-only fitting, inverse geometry, independent
randomness, actual ODE calls, explicit train/validation training and checkpoint
reload. The pinned-source [component audit](diffeocfm_adapter_parity.json) compares
the transform, full fitted prior, inverse, network, loss and gradients. It does
not certify an entire original training trajectory or scientific utility.

The three-epoch full-dimension pilot uses the same hyperparameters and seed 1701,
but additionally evaluates its final epoch 3. Its checkpoint scores are diagnostic;
they must not be added to the full run's declared selection schedule. Training
takes about 2.2 seconds per epoch on the measured GB10; the pilot uses about
0.78 GiB peak allocated GPU memory. Retain the pilot and full run separately.

Use CorrBench's common evaluator on saved samples for structural, entry/spectral,
coverage, classifier and portfolio diagnostics. The existing evaluator scores
float32 matrices after symmetry/unit-diagonal preparation; preserve the native
float64 validity report too. No positive-definite repair is applied to DiffeoCFM
or DiffeoGauss samples. Portfolio functionals are correlation-only diagnostics
with unit marginal volatility, not an investment backtest or a utility claim.

## Commands

```bash
python -m pip install -e ../corrbench -e '.[dev,diffeocfm]'
corrflow-train-diffeocfm --data /path/to/taiwan_final_ward \
  --config configs/diffeocfm_taiwan_categorical.yaml \
  --output artifacts/diffeocfm_seed1701 --device cuda
corrflow-sample-diffeocfm --run artifacts/diffeocfm_seed1701 \
  --data /path/to/taiwan_final_ward --split validation \
  --weights best_structural --output artifacts/seed1701_validation.npz
corrbench-evaluate --data /path/to/taiwan_final_ward --split validation \
  --generated artifacts/seed1701_validation.npz --seed 44301 \
  --output artifacts/seed1701_validation_metrics.json
```

Repeat sampling with `--weights gaussian` and `--weights last` to distinct output
paths. Trainer/sampler reject existing output paths. The parity research script
additionally requires `pyriemann` and `torchcfm` and a clone of the pinned source;
these are not needed for adapter training/sampling.
