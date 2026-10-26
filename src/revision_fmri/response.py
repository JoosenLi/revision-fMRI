"""Explicit per-session response normalization, before repeat aggregation."""
from dataclasses import dataclass
import numpy as np
from scipy import stats
from .provenance import array_digest


@dataclass(frozen=True)
class NormalizedSession:
    values: np.ndarray
    provenance: dict[str, str]


def normalize_session(values: np.ndarray, *, policy: str, fit_rows=None) -> NormalizedSession:
    """Input trials × voxels; no inference of sessions or train/test membership.

    legacy_deepnsd_session_v1 uses all session trials, ddof=0 and float32 output
    as in nsd_parser.load_NSD_brain_data. It is NOT independent preprocessing.
    train_only_session_v1 is a separately named adaptation, with explicit rows.
    Neither policy is silently enabled in the LAION reader.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Require a finite trial-by-voxel session matrix')
    if policy == 'legacy_deepnsd_session_v1':
        if fit_rows is not None:
            raise ValueError('Legacy session normalization fits all session trials')
        rows = np.arange(len(x))
        if np.any(x.std(axis=0, ddof=0) == 0):
            raise ValueError('Constant session voxel has undefined z-score')
        result = stats.zscore(x, axis=0, ddof=0)
    elif policy == 'train_only_session_v1':
        rows = np.asarray(fit_rows)
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(rows) < 2
                or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(x))):
            raise ValueError('Explicit unique training row indices required for each session')
        mean, std = x[rows].mean(axis=0), x[rows].std(axis=0, ddof=0)
        if np.any(std == 0):
            raise ValueError('Constant training voxel has undefined session z-score')
        result = (x - mean) / std
    else:
        raise ValueError('Unknown normalization policy; no automatic default')
    result = result.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Session z-score produced nonfinite values; no silent voxel removal')
    return NormalizedSession(result, {'normalization': policy, 'response_precision': 'float32',
        'normalization_fit_rows_sha256': array_digest(rows),
        'normalization_fit_values_sha256': array_digest(x[rows]),
        'normalization_output_sha256': array_digest(result),
        'normalization_reference': 'DeepNSD 36f6a34727dccb5eec8fa49364b210a049a388d7 nsd_parser.py:load_NSD_brain_data',
        'normalization_reference_sha256': 'deb1bc0dc70d686236d83819840df653ac476373389ceaf6e695543a22ad5522',
        'normalization_scope': 'all session trials' if policy.startswith('legacy') else 'explicit training trials'})
