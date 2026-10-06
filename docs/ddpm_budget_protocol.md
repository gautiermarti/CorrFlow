# Image DDPM training-budget study — declared 2026-10-06

This addresses the 20-epoch baseline's unresolved optimization budget. It is a
separate retrospective study, not a replacement of the original paper results
or an untouched market test. Longer training need not improve distributional
fidelity. The earlier paper's minimum denoising-validation-loss rule is retained
so that training duration is the intended change within this new paired study.

## Frozen training comparison

- Use the existing Taiwan final Ward-ordered train/validation/test splits,
  without changing matrices, conditions, ordering or regime thresholds.
- Keep the original conditional image U-Net: base width 32, 16 time frequencies,
  three regime indicators plus the training-standardized continuous score.
- Cosine schedule, 1,000 diffusion times, epsilon objective, batch 32, AdamW
  learning rate .0002, decay .0001, gradient clipping 1, EMA decay .995, bfloat16
  forward passes as in configs/ddpm_image.yaml. No hyperparameter search.
- Fit seeds 1701–1705 from initialization for 200 epochs each. The original
  checkpoints do not include optimizer/RNG state and cannot faithfully resume.
- Preserve paired 20/50/100/200-epoch budgets from each trajectory. For each
  budget, select raw weights at the first minimum fixed-noise denoising loss
  among its epochs. The original selection uses up to 256 validation matrices
  and fixed sampled noise/times; preserve its exact RNG order and tensors.
- Store selected and final raw/EMA weights for every budget, complete curves,
  optimizer/RNG state, validation buffers, data/code/config/runtime hashes.
  Resume requires an identical signature; do not switch CPU/CUDA, precision,
  Torch version, data, source or configuration midway through a run.
- Reproduce every native selected validation loss before scoring comparisons.
  Reaching 200 epochs is a declared budget, not proof of convergence or exhaustive
  tuning; the published much longer diffusion recipe is not reproduced here.

The new 20-epoch snapshots form the paired short-budget control. Check their
training histories against the original artifacts, but do not promise bitwise
identity across historical software/device versions or silently substitute them
for the archived table. The old checkpoints and results remain immutable.

## Generation and diagnostics

Use deterministic 50-step DDIM with the existing predicted-clean clipping at 5.
Main comparisons use each budget's selected raw weights. Fixed case-selection
seed 2027 chooses 500 test matrices per regime, identical across all fits and
budgets. Pair generation seeds 2027–2031 with training seeds 1701–1705. The sampler
now separates case selection from latent noise, preserving the old coupled
behavior when no explicit selection seed is passed.

Before test scoring, inspect fixed validation generations (100 per regime,
selection/generation seed 44301) for every selected budget and the final epoch.
These diagnose training; they do not introduce another model-selection rule.
After all five full budgets complete and native selections pass audit, evaluate
all declared test arms together. Do not stop or discard a fit because of an
unfavorable validation or test distributional result.

Report the common structural/energy/orbit distances, precision/recall, C2ST,
nearest-training distances and unit-volatility portfolio functionals. Report raw
validity, upper-triangle symmetry/diagonal normalization, and converged float64
nearest-correlation repair (cap 1,000, existing tolerance and strict-PD floor).
Retain correction magnitudes and total generation/repair cost. No missing or
nonconverged draw is silently removed.

At 200 epochs, retain final raw and validation-selected EMA weights as secondary
sensitivity arms; do not choose between them using test outcomes. Compare paired
within-fit changes from 20 to 200 epochs, all five individual outcomes and sample
SD. Seed variability is not uncertainty over independent financial episodes.
The categorical CorrFlow/DiffeoCFM experiment is a different information setting.

## Resource and execution policy

Measure the full-width training step before allocating the run. The currently
occupied GPU belongs to another job and is not interrupted. The experiment runner
waits for no other CUDA compute PID for 90 seconds. If another job appears, only
this trainer is asked to yield at its next completed epoch; it resumes the same
saved state after the GPU becomes free again. Runtime includes such interruptions
only in the operational ledger; it is not an isolated speed ranking.

The eight-step full-width CPU bfloat16 probe failed to complete within ten
minutes and was stopped; no CPU throughput number is inferred from partial work.
Historical 20-epoch GPU runs took 528–674 seconds. Each new fit first executes
two epochs under the unchanged 200-epoch configuration and saves resumable state.
If that pilot exceeds 120 seconds per epoch on average, the coordinator stops for
resource diagnosis instead of launching the full allocation. Otherwise it resumes
the same trajectory to 200 epochs after checking GPU availability again.

Before launch, freeze a machine-readable command plan containing source/data/
configuration hashes and the five runs. Test exact uninterrupted-versus-resumed
CPU trajectories and interrupted derived-file recovery, then perform the measured
full-dimensional GPU run when the resource becomes available. The runner uses
an exclusive lock and refuses changed plans/inputs or unexplained failed attempts.
No test generation is launched by the training-only runner.
