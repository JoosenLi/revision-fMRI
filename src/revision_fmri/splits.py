"""Versioned explicit split loading; no random splitting or trial-level guessing."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .config import LocalConfig
from .evaluation import EvaluationSplit
from .cache import _sha


@dataclass(frozen=True)
class RecordedSplit:
    split: EvaluationSplit
    dataset_release: str
    source_revision: str
    pool: str

    def __post_init__(self):
        _sha(self.split.source_digest)
        if any(not isinstance(value, str) or not value for value in
               (self.dataset_release, self.source_revision, self.pool)):
            raise ValueError('Recorded split release, revision and pool are required')


def load_split(config: LocalConfig, *, filename: str, expected_sha256: str) -> RecordedSplit:
    """Read a pinned JSON under data_root; exact label strings/order are preserved.

    This is the project's interchange format, not an inferred official LAION
    export schema. Importers must record their source revision and dataset pool.
    """
    _sha(expected_sha256)
    if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
        raise ValueError('Split filename must be relative to configured data_root')
    root = config.storage['data_root'].resolve()
    path = (root/filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Split path escapes data_root')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Split file SHA-256 mismatch')
    payload = json.loads(raw)
    keys = {'schema_version','protocol_id','dataset_release','source_revision','pool','train_ids','test_ids'}
    if not isinstance(payload, dict) or set(payload) != keys or payload['schema_version'] != 1:
        raise ValueError('Unsupported split manifest schema')
    if any(not isinstance(payload[k],str) or not payload[k] for k in
           ('protocol_id','dataset_release','source_revision','pool')):
        raise ValueError('Split protocol, release, source revision and pool are required')
    if any(not isinstance(payload[k],list) for k in ('train_ids','test_ids')):
        raise ValueError('Split partitions must be explicit ID lists')
    split = EvaluationSplit(tuple(payload['train_ids']),tuple(payload['test_ids']),
                            payload['protocol_id'],expected_sha256)
    return RecordedSplit(split,payload['dataset_release'],payload['source_revision'],payload['pool'])
