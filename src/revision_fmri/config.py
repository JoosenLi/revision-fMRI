"""Unified local YAML settings; reading settings never creates resources/jobs."""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping
import yaml

ROOT_KEYS = frozenset({'data_root', 'stimuli_root', 'checkpoint_root', 'feature_root', 'results_root'})


@dataclass(frozen=True)
class ComputeConfig:
    device: str
    allowed_gpu_ids: tuple[int, ...]
    precision: str
    ridge_precision: str


@dataclass(frozen=True)
class LocalConfig:
    storage: Mapping[str, Path]
    compute: ComputeConfig
    checkpoints: Mapping[str, Path]


def _path(value: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Paths must be nonempty strings')
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path | None = None) -> LocalConfig:
    """Read explicit YAML or REVISION_FMRI_CONFIG; require CUDA policy, not availability.

    No resource existence checks, device initialization or automatic dtype switch
    occurs here. The execution backend must check availability before any run.
    """
    supplied = path if path is not None else os.environ.get('REVISION_FMRI_CONFIG')
    if not supplied:
        raise ValueError('Supply a local YAML config path or REVISION_FMRI_CONFIG')
    config_path = Path(supplied).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema')
    if set(payload) != {'schema_version', 'storage', 'compute', 'checkpoints'}:
        raise ValueError('Expected schema_version, storage, compute and checkpoints')
    raw_paths = payload['storage']
    if not isinstance(raw_paths, dict) or set(raw_paths) != ROOT_KEYS:
        raise ValueError(f'Expected exactly these storage keys: {sorted(ROOT_KEYS)}')
    storage = {k: _path(v, config_path.parent) for k, v in raw_paths.items()}
    raw_compute = payload['compute']
    if not isinstance(raw_compute, dict) or set(raw_compute) != {'device', 'allowed_gpu_ids', 'precision', 'ridge_precision'}:
        raise ValueError('Explicit device, GPU allowance and both precisions are required')
    if raw_compute['device'] != 'cuda':
        raise ValueError('Vision execution requires cuda; CPU fallback is forbidden')
    ids = raw_compute['allowed_gpu_ids']
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('GPU IDs must be unique nonnegative integers')
    if raw_compute['precision'] not in ('float32', 'float64') or raw_compute['ridge_precision'] not in ('float32', 'float64'):
        raise ValueError('Supported explicit precisions are float32 and float64')
    compute = ComputeConfig(raw_compute['device'], tuple(ids), raw_compute['precision'], raw_compute['ridge_precision'])
    raw_checkpoints = payload['checkpoints']
    if not isinstance(raw_checkpoints, dict) or any(not isinstance(k, str) or not k for k in raw_checkpoints):
        raise ValueError('Checkpoint bindings require nonempty logical model IDs')
    checkpoints = {k: _path(v, storage['checkpoint_root']) for k, v in raw_checkpoints.items()}
    return LocalConfig(storage, compute, checkpoints)
