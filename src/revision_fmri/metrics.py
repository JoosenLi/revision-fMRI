"""Verified cRSA core; no SRP, fitting, pooling or feature standardization."""
import numpy as np
from numpy.typing import ArrayLike, NDArray


def correlation_rdm(patterns: ArrayLike) -> NDArray:
    values = np.asarray(patterns, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError('Require finite image × feature patterns with both axes >= 2')
    if np.any(np.ptp(values, axis=1) == 0):
        raise ValueError('Constant image patterns have undefined Pearson distance')
    return 1.0 - np.corrcoef(values)


def compare_rdms(left: ArrayLike, right: ArrayLike) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1] or a.shape[0] < 3:
        raise ValueError('Require matching square RDMs with at least three images')
    indices = np.triu_indices(a.shape[0], k=1)
    x, y = a[indices], b[indices]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError('Upper triangles must be finite and nonconstant')
    return float(np.corrcoef(x, y)[0, 1])


def crsa(features: ArrayLike, responses: ArrayLike) -> float:
    """Caller must align image order; use evaluation.evaluate_crsa for ID checks."""
    return compare_rdms(correlation_rdm(features), correlation_rdm(responses))
