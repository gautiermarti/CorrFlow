# CorrFlow and comparison models: development model cards

Updated 2026-10-06. Package `0.1.0rc1`, coordinated release candidate. Record the
full model Git revision, pinned CorrBench revision, configuration, data hashes,
training seed, selected checkpoint and sampling seed. Code is MIT licensed;
upstream DiffeoCFM attribution is in `THIRD_PARTY_NOTICES.md`. Data/checkpoint
release rights are separate. A synthetic 6D teaching checkpoint is bundled under tutorials/assets. It is not
a financial or participant-trained checkpoint; its data/provenance manifest and
executed lessons are in tutorials/. Research checkpoints remain separate.

## CorrFlow for anonymous correlation matrices

**Task and representation.** Learn a distribution of positive-definite
correlation matrices in generalized Fisher coordinates. An equivariant graph
vector field transports Gaussian source coordinates under flow matching.
Numerical inverse GFT reconstructs unit-diagonal positive-definite matrices;
its convergence and floating-point validity are checked. The model does not
generate returns, marginal distributions or temporal trajectories.

**Inputs and variants.** The financial paper model receives three regime labels
and a continuous score. The additional matched study supplies only regime labels,
using `configs/flow_taiwan_categorical.yaml`: 48 hidden channels, three blocks,
20 epochs, 25 Heun sampling steps and five training seeds. Do not pool scores
across these information settings. CorrBench owns dated-return construction,
Ward ordering, GFT caches, immutable split hashes and the common evaluator.

**Measured result.** In the completed categorical Taiwan protocol, identical
1,500 test cases give selected structural SWD .4709 ± .0615, versus .7512 ± .0188
for the adapted DiffeoCFM. Portfolio SWD is .6243 versus .8620; recall is .3204
versus .3549. Both classifiers distinguish generated and real data with AUC above
.996. Final-checkpoint structural scores reverse the ordering (.8363 versus
.7980). These standard deviations combine training and paired generation-seed
variability. All 15 native selections reproduce exactly, including the completed categorical
cCorrGAN reconstruction. Its structural SWD is 4.0390 ± .1537, with zero
precision and portfolio SWD 818.7222 ± 824.7008 after converged repair.
This negative reconstruction does not establish original GAN training fidelity.
See [the audited comparison](https://github.com/gautiermarti/corrbench/blob/main/docs/matched_taiwan_results.md).

**Scope.** These are method-specific 20/200-epoch budgets, not equal compute.
Dates were inspected previously; overlapping windows and stock subsets do not
create independent market episodes. The result does not establish universal
superiority, future trading performance or a validated extreme-risk simulator.
Financial empirical evidence is at 80D. Configurable dimensions, computational
scaling and shape-compatible sampling do not establish trained size transfer or
consistent distributions across nested submatrices.

**Symmetry.** Simultaneous row/column relabeling consistently relabels the
anonymous model's output. With an exchangeable source and global conditions,
the resulting law is exchangeable. Fixed anatomical or spatial identities require
appropriate attributes or another source of named-variable information.

## Fixed-atlas node-aware CorrFlow

**Task and inputs.** Experimental extension for ABIDE PCP correlations at 111
source labels: 110 officially named regions and one unnamed label. Each node
receives its atlas identity; matrices and
attributes must be permuted together. `configs/corrflow_abide_ho_pilot.yaml`
uses 32 channels, two blocks, 20 epochs and 25 sampling steps. The anonymous
ablation removes node identities with the same training budget.

**Data and evidence.** The OAS-regularized pilot has 349 training subjects, 92
Michigan validation subjects and 160 NYU test subjects. Constant-ROI scans are
excluded by the declared construction protocol. The initial one-training-seed
augmentation experiment establishes no gain over real-only classification.
The unchanged expansion to five training seeds is complete. Mean AUC is .7365
for node-aware CorrFlow and .7314 for its anonymous ablation, versus .7654
real-only and .7452 selected DiffeoCFM. No augmentation gain is established.
Five draws are averaged within each fit; training SD is .0137 and .0184 for the
two CorrFlow variants. The same NYU site had already been inspected, and the
initial one-fit report remains separately preserved.

**Source sensitivity.** Unnamed label 3455 occupies two voxels and alone caused
161 of the 179 original constant-series exclusions. A separately declared
110-region reconstruction compares the same 601 subjects with all 762 scans
without another constant named region. Neither replay nor Gaussian augmentation
establishes a gain; real-only AUC is .7725 and .8013 on their respective test
cohorts. Learned 110-region models have not been fitted. See the
[atlas audit](https://github.com/gautiermarti/corrbench/blob/main/docs/abide_atlas_audit_20261006.md)
and [separate controls](https://github.com/gautiermarti/corrbench/blob/main/docs/abide_named110_results_20261006.md).
The original results describe the frozen 601-subject cohort, not a validated
anatomical choice or fresh-site result.

**Scope.** Tests establish simultaneous matrix/attribute equivariance and learning
gradients, not improved clinical prediction. One-hot atlas identity cannot encode
unseen-atlas anatomy. Functional correlation does not identify causal or anatomical
connections, and synthetic samples do not create independent subjects. This
research model is not a clinical tool. See the [implementation note](neuro_pilot.md)
and CorrBench's pilot/replication protocols for exact exclusions and task design.

## Adapted DiffeoCFM and its Gaussian prior

**Provenance and method.** Correlation path adapted from official upstream
revision `902282acd13a67784e2096d09ee93477b268c079`. It preserves normalized-
Cholesky coordinates, a training-fitted scaler, the full class-wise OAS Gaussian,
float64 MLP and flow objective. Component parity covers transform, fitted prior,
inverse, network, loss and gradients. It is not a claim to reproduce the complete
upstream experiments or original training trajectory.

**Benchmark adaptation.** Explicit train/validation splits and minimum structural
validation SWD select checkpoints. Upstream final weights remain a separate
sensitivity result. The Gaussian prior is evaluated separately with matching
draws. Adaptive ODE sampling reports actual vector-field evaluations and numerical
tolerances; output times are not evaluation counts. See the
[protocol](diffeocfm_protocol.md) and [parity audit](diffeocfm_adapter_parity.json).

**Evidence and scope.** Five 80D financial fits are complete, with the mixed
comparison above. The initial 111D brain augmentation pilot establishes no
real-only improvement; its completed five-fit selected mean is .7452 versus
.7654 real-only. The joint exploratory paired difference range is negative,
conditional on this inspected site. One 64D
weather fit improves aggregate-variance distance over its Gaussian prior (.0894
versus .1076) but does not beat conditional means (.0824) or matched-draw replay
(.0483). Five generation seeds do not replicate that weather fit. These models
are independently fitted by domain; no financial-to-brain/weather weight transfer
or forecast benefit has been demonstrated.

The package also contains GFT diffusion and image-DDPM research controls. They
remain separately named architectures/configurations. The categorical flow
results above must not be used as their performance evidence. The image-DDPM
trainer supports paired 20/50/100/200-epoch snapshots, exact CPU resume, fixed
generation cases and native validation-score auditing. The longer five-fit study
is queued separately; these implementation checks are not completed training
or improved-fidelity evidence. Public release
requires the checks in [the release contract](release_contract.md).
