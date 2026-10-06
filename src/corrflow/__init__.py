"""CorrFlow: geometry-aware generation of financial correlation matrices."""

from corrbench.geometry import (
    corr_to_gft,
    gft_to_corr,
    is_correlation_matrix,
    lower_to_symmetric,
    symmetric_to_lower,
)

__all__ = [
    "corr_to_gft",
    "gft_to_corr",
    "is_correlation_matrix",
    "lower_to_symmetric",
    "symmetric_to_lower",
]
