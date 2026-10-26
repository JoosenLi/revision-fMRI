"""Validated boundary around attributed, unmodified DeepNSD numerical code.

The compatibility environment is pinned; it is not asserted to be the authors'
original environment. No weights, data, models or network access occur here.
"""
from dataclasses import dataclass
from importlib import metadata, resources
import hashlib
import json
import numpy as np
from numpy.typing import NDArray

EXPECTED_ENVIRONMENT = {'numpy': '1.26.4', 'scipy': '1.11.4', 'scikit-learn': '1.3.2', 'tqdm': '4.67.1'}


def verify_sources() -> dict:
    root = resources.files('revision_fmri._vendor.deepnsd')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != entry['vendored_sha256']:
            raise RuntimeError(f'Vendored DeepNSD source changed: {name}')
    return manifest


def require_compatible_environment() -> dict[str, str]:
    for name, expected in EXPECTED_ENVIRONMENT.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install the pinned 'encoding' extra before using DeepNSD") from exc
        if actual != expected:
            raise RuntimeError(f'DeepNSD compatibility environment requires {name}=={expected}; found {actual}')
    manifest = verify_sources()
    return {'backend': 'deepnsd-vendored-sklearn132-v1',
            'deepnsd_commit': manifest['commit'],
            'deepnsd_ridge_sha256': manifest['files']['ridge_gcv_mod.py']['vendored_sha256'],
            'deepnsd_rsa_sha256': manifest['files']['rsa.py']['vendored_sha256'],
            'deepnsd_srp_sha256': manifest['files']['reduction.py']['vendored_sha256']}


@dataclass(frozen=True)
class DeepNSDFit:
    coefficients: NDArray
    intercept: NDArray
    alpha_per_voxel: NDArray
    loo_centered: NDArray
    alpha_scores: NDArray

    def predict(self, x: NDArray) -> NDArray:
        # C1 uses this exact coefficient + intercept expression.
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.coefficients.shape[1] or not np.isfinite(x).all():
            raise ValueError('Prediction features must be finite and match fitted feature count')
        return x.dot(self.coefficients.transpose()) + self.intercept


def fit_ridge_deepnsd(x: NDArray, y: NDArray, *, alphas: tuple[float, ...]) -> DeepNSDFit:
    """Execute original RidgeCVMod; retain centered cv_values_ and original alphas.

    Only input validation/ID-independent result adaptation is added. No patched
    sklearn private calls, normalization switches or prediction offset fixes.
    """
    require_compatible_environment()
    from ._vendor.deepnsd.ridge_gcv_mod import RidgeCVMod, pearson_r_score
    x, y = np.asarray(x), np.asarray(y)
    grid = np.asarray(alphas, dtype=float)
    if (grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or len(np.unique(grid)) != grid.size):
        raise ValueError('Alphas must be unique finite positive values in explicit order')
    if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3
            or min(x.shape[1], y.shape[1]) < 1
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Require finite aligned training matrices')
    if np.any(np.ptp(y, axis=0) == 0):
        raise ValueError('Constant responses have undefined Pearson scores')
    reg = RidgeCVMod(alphas=grid.tolist(), store_cv_values=True,
                     alpha_per_target=True, scoring='pearson_r')
    reg.fit(x, y)
    alpha = np.atleast_1d(reg.alpha_)
    idx = np.array([grid.tolist().index(a) for a in alpha])
    predictions = np.take_along_axis(reg.cv_values_, idx[None, :, None], axis=2)[:, :, 0]
    scores = np.column_stack([pearson_r_score(y, reg.cv_values_[:, :, i]) for i in range(len(grid))])
    if not all(np.isfinite(v).all() for v in (reg.coef_, reg.intercept_, predictions, scores)):
        raise ValueError('Original DeepNSD returned undefined predictions or Pearson scores')
    return DeepNSDFit(np.atleast_2d(reg.coef_), np.atleast_1d(reg.intercept_),
                      alpha, predictions, scores)
