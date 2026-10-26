"""Immutable, content-addressed raw feature artifacts on configured storage."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping
import numpy as np

from .config import LocalConfig, load_config
from .features import FeatureSet
from .provenance import array_digest, metadata_digest
from .data import unique_ids


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Require a lowercase SHA-256 digest')
    return value


@dataclass(frozen=True)
class FeatureIdentity:
    checkpoint_sha256: str
    transform_sha256: str
    candidate_manifest_sha256: str
    stimuli_sha256: str
    loader_revision: str
    software_versions: Mapping[str, str]
    seed: int

    def __post_init__(self):
        for name in ('checkpoint_sha256', 'transform_sha256', 'candidate_manifest_sha256', 'stimuli_sha256'):
            _sha(getattr(self, name))
        if not isinstance(self.loader_revision, str) or not self.loader_revision:
            raise ValueError('Explicit loader revision required')
        if (not self.software_versions or any(not isinstance(k, str) or not k
                or not isinstance(v, str) or not v for k, v in self.software_versions.items())):
            raise ValueError('Explicit software versions required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Explicit unsigned 32-bit seed required')


@dataclass(frozen=True)
class CachedModel:
    model_id: str
    layer_ids: tuple[str, ...]
    feature_keys: tuple[str, ...]
    identity: FeatureIdentity

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError('Explicit model ID required')
        unique_ids(self.layer_ids, 'Candidate layer IDs')
        unique_ids(self.feature_keys, 'Feature artifact keys')
        if len(self.layer_ids) != len(self.feature_keys):
            raise ValueError('Each candidate layer requires exactly one cache key')
        for key in self.feature_keys:
            _sha(key)


class FeatureStore:
    """Raw flattened features only; no fitted encoding transform in this cache.

    Writers exclusively create a key directory and publish COMPLETE last. Failed
    writes remain incomplete and cannot be read or overwritten automatically.
    This is an integrity protocol for cooperating clients, not an access-control
    boundary against concurrent malicious modification of the filesystem.
    """
    def __init__(self, config: LocalConfig):
        self.root = config.storage['feature_root']

    @classmethod
    def from_config(cls, path=None):
        return cls(load_config(path))

    def _directory(self, key):
        _sha(key)
        directory = self.root / key
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Cache artifact escapes configured feature_root')
        return directory

    def put(self, features: FeatureSet, *, identity: FeatureIdentity) -> str:
        if features.representation != 'raw_flattened':
            raise ValueError('Only raw features are supported; fitted transforms need separate cache identities')
        if features.values.dtype not in (np.dtype('float32'), np.dtype('float64')):
            raise ValueError('Feature storage must explicitly be float32 or float64')
        metadata = {'schema_version': 1, 'identity': asdict(identity),
                    'image_ids': list(features.image_ids), 'model_id': features.model_id,
                    'layer_id': features.layer_id, 'representation': features.representation,
                    'provenance': dict(features.provenance), 'dtype': features.values.dtype.str,
                    'shape': list(features.values.shape), 'values_sha256': array_digest(features.values)}
        key = metadata_digest(metadata)
        payload = json.dumps(metadata, indent=2, allow_nan=False)+'\n'
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self._directory(key)
        directory.mkdir()  # Exclusive; never replace an existing cache, complete or partial.
        with (directory/'values.npy').open('xb') as stream:
            np.save(stream, features.values, allow_pickle=False)
        with (directory/'metadata.json').open('x', encoding='utf-8') as stream:
            stream.write(payload)
        # Detect mutation during writing before publishing the commit marker.
        persisted = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)
        if array_digest(persisted) != metadata['values_sha256']:
            raise ValueError('Feature values changed during cache write; artifact left incomplete')
        with (directory/'COMPLETE').open('x') as stream:
            stream.write(key+'\n')
        return key

    def get(self, key: str, *, expected_identity: FeatureIdentity) -> FeatureSet:
        directory = self._directory(key)
        for name in ('COMPLETE', 'metadata.json', 'values.npy'):
            path = directory/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Cache artifact is incomplete or contains a symlink')
        if (directory/'COMPLETE').read_text().strip() != key:
            raise ValueError('Cache completion marker mismatch')
        metadata = json.loads((directory/'metadata.json').read_text())
        if metadata_digest(metadata) != key or metadata.get('schema_version') != 1:
            raise ValueError('Cache metadata digest/schema mismatch')
        if metadata['identity'] != asdict(expected_identity):
            raise ValueError('Cache scientific identity differs from requested identity')
        values = np.load(directory/'values.npy', mmap_mode='r', allow_pickle=False)
        if (list(values.shape) != metadata['shape'] or values.dtype.str != metadata['dtype']
                or values.dtype not in (np.dtype('float32'), np.dtype('float64'))
                or array_digest(values) != metadata['values_sha256']):
            raise ValueError('Cache values/shape/precision digest mismatch')
        if metadata['representation'] != 'raw_flattened':
            raise ValueError('Only raw feature artifacts can enter this evaluation path')
        return FeatureSet(tuple(metadata['image_ids']), metadata['model_id'], metadata['layer_id'],
                          values, metadata['representation'], metadata['provenance'])
