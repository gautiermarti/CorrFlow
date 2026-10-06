# %% [markdown]
# # Correlation geometry in a few minutes
# Three plausible pairwise correlations need not form a possible joint system.
# This CPU lesson checks validity, the generalized Fisher transform (GFT), and
# relabeling. It uses fixed mathematical examples, not a trained generator.
#
# Install the compatible CorrBench and CorrFlow packages before running.
# %%
import json
import time

import numpy as np
import torch
from corrbench.geometry import corr_to_gft, gft_to_corr

started = time.perf_counter()
torch.set_num_threads(2)
bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
weights = np.array([-1.0, 1.0, 1.0])
print("Eigenvalues:", np.linalg.eigvalsh(bad))
print("Impossible variance:", weights @ bad @ weights)
assert weights @ bad @ weights < 0
# %% [markdown]
# A covariance cannot give a negative variance. Keeping every entry between
# minus one and one does not enforce the joint constraint.
#
# GFT coordinates are the strict lower triangle of the matrix logarithm.
# Their inverse solves for a diagonal whose matrix exponential has unit diagonal.
# %%
gamma = torch.tensor([[0.7, -0.4, 0.2]], dtype=torch.float64)
good, info = gft_to_corr(gamma, tolerance=1e-10, max_iterations=1000, return_info=True)
assert info["converged"].all()
assert torch.linalg.eigvalsh(good).min() > 0
torch.testing.assert_close(corr_to_gft(good), gamma, atol=1e-9, rtol=0)
print(good[0].numpy())
# %% [markdown]
# In two dimensions, this reduces to Fisher's familiar atanh transform.
# %%
bivariate = torch.tensor([[[1.0, 0.6], [0.6, 1.0]]], dtype=torch.float64)
np.testing.assert_allclose(corr_to_gft(bivariate).item(), np.arctanh(0.6), atol=1e-12)
# %% [markdown]
# Relabel both the matrix and the aggregation weights. The variance is unchanged.
# Holding the weights fixed while changing the matrix's labels is a different task.
# %%
C = good[0].numpy()
p = np.array([2, 0, 1])
w = np.array([0.7, 0.2, 0.1])
permuted = C[p][:, p]
np.testing.assert_allclose(w @ C @ w, w[p] @ permuted @ w[p], atol=1e-12)
report = {
    "negative_variance": float(weights @ bad @ weights),
    "gft_roundtrip_error": float(torch.max(torch.abs(corr_to_gft(good) - gamma))),
    "same_named_variance": float(w @ C @ w),
    "incorrect_fixed_labels_variance": float(w @ permuted @ w),
    "elapsed_seconds": time.perf_counter() - started,
}
print(json.dumps(report, indent=2))
# %% [markdown]
# Validity establishes that a matrix is possible. It does not establish that it
# resembles a population or helps a downstream decision. Continue with generation
# and distributional evaluation in lesson 2.
