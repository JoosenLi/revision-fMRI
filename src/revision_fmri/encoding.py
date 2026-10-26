"""Small-array float64 encoding reference, independently implemented from ridge algebra.

No vision models, GPU execution, optimization backend or implicit alpha grid.
The selected training predictions deliberately retain legacy response centering.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RidgeReferenceFit:
    coefficients: NDArray  # voxels × features
    intercept: NDArray
    alpha_per_voxel: NDArray
    loo_centered: NDArray
    alpha_scores: NDArray  # voxels × alphas, in caller-provided order

    def predict(self, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients.shape[1] or not np.isfinite(x).all():
            raise ValueError('Prediction features must be finite and match fitted feature count')
        return x @ self.coefficients.T + self.intercept


def _pearson_columns(y: NDArray, predictions: NDArray) -> NDArray:
    a, b = y - y.mean(axis=0), predictions - predictions.mean(axis=0)
    # Normalize independently to avoid multiplying two potentially large norms.
    norm_a, norm_b = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise ValueError('Undefined Pearson alpha score: constant target or LOO prediction')
    scores = np.sum((a / norm_a) * (b / norm_b), axis=0)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite Pearson alpha scores')
    return np.clip(scores, -1.0, 1.0)


def fit_ridge_reference(x: NDArray, y: NDArray, *, alphas: tuple[float, ...],
                        legacy_l2_normalize: bool) -> RidgeReferenceFit:
    """Per-voxel Pearson LOOCV alpha selection, fixed externally scaled X.

    Includes an unpenalized intercept. Unlike a nested scaling pipeline, the
    outer-train scaler is not refitted in each leave-one-out fold. Tied alpha
    scores retain the first alpha in the supplied order. Input arrays are not
    mutated; sample weights and alternate scoring are intentionally unsupported.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if type(legacy_l2_normalize) is not bool:
        raise ValueError('Explicit legacy L2 normalization policy is required')
    grid = np.asarray(alphas, dtype=np.float64)
    if (grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or len(np.unique(grid)) != grid.size):
        raise ValueError('Alphas must be a nonempty ordered sequence of unique finite positive values')
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
            or x.shape[0] < 3 or x.shape[1] == 0 or y.shape[1] == 0
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Require finite aligned training matrices with >=3 images and nonempty columns')
    if np.any(np.ptp(y, axis=0) == 0):
        raise ValueError('Constant response columns have undefined Pearson scores')
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    scale = np.ones(x.shape[1])
    if legacy_l2_normalize:
        # sklearn 1.3.2 _preprocess_data receives copy_X as positional normalize.
        variance = np.var(x, axis=0)
        eps = np.finfo(np.float64).eps
        constant = variance <= len(x) * eps * variance + (len(x) * x_mean * eps) ** 2
        scale = np.sqrt(np.sum(xc ** 2, axis=0))
        scale[constant] = 1.0
        xc = xc / scale
    u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    uy = u.T @ yc
    squared = singular ** 2
    scores = np.empty((y.shape[1], grid.size))
    selected = np.zeros(y.shape[1], dtype=int)
    best = np.full(y.shape[1], -np.inf)
    loo = np.empty_like(yc)
    for i, alpha in enumerate(grid):
        shrink = squared / (squared + alpha)
        fitted_centered = u @ (shrink[:, None] * uy)
        # The 1/n term is essential: the intercept is re-estimated in each LOO fit.
        leverage = 1.0 / len(x) + (u ** 2) @ shrink
        denominator = 1.0 - leverage
        if np.any(denominator <= 0):
            raise ValueError('Numerically singular LOO denominator; reference requires rescaling/review')
        predictions = yc - (yc - fitted_centered) / denominator[:, None]
        scores[:, i] = _pearson_columns(yc, predictions)
        update = scores[:, i] > best
        selected[update] = i
        best[update] = scores[update, i]
        loo[:, update] = predictions[:, update]
    chosen = grid[selected]
    coefficients = (vt.T @ ((singular[:, None] / (squared[:, None] + chosen)) * uy)).T / scale
    intercept = y_mean - x_mean @ coefficients.T
    if not all(np.isfinite(v).all() for v in (coefficients, intercept, loo)):
        raise ValueError('Nonfinite fitted ridge outputs')
    return RidgeReferenceFit(coefficients, intercept, chosen, loo, scores)
