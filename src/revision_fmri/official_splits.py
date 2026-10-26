"""Offline import of pinned, unmodified official LAION shared-pool assignments."""
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import numpy as np
from .config import LocalConfig
from .evaluation import EvaluationSplit
from .splits import RecordedSplit


def official_split_inventory() -> dict:
    """Return the packaged source lock, not filesystem discovery or generated splits."""
    return json.loads(resources.files('revision_fmri').joinpath('resources/laion_shared_splits.json').read_text())


def load_official_split(config: LocalConfig, *, name: str, dataset_release: str,
                        variant_id: int = 0) -> RecordedSplit:
    """Read existing local official JSON with pinned byte identity and original order.

    Supports the locked shared pool only, including all OOD categories. Never
    downloads, sorts labels, regenerates assignments, or intersects a partial
    participant/session with the requested official split.
    """
    lock = official_split_inventory()
    relative = f'laion_fmri/splits/data/shared/{name}.json'
    if relative not in lock['files'] or not isinstance(name, str):
        raise ValueError('Unknown split name in pinned shared-pool inventory')
    if type(variant_id) is not int:
        raise ValueError('variant_id must be an integer')
    root = config.storage['data_root'].resolve()
    path = root/'official_splits'/lock['commit']/relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Official split path escapes configured data_root')
    raw = path.read_bytes()
    expected = lock['files'][relative]
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected['size'] or digest != expected['sha256']:
        raise ValueError('Official split bytes differ from pinned source')
    payload = json.loads(raw)
    if (payload['name'] != name or payload['pool'] != expected['raw_pool']
            or payload['n_train'] != expected['n_train'] or payload['n_test'] != expected['n_test']):
        raise ValueError('Official split metadata mismatch')
    variants = [v for v in payload['variants'] if v['variant_id'] == variant_id]
    if len(variants) != 1:
        raise ValueError('Requested variant must occur exactly once in official split')
    v = variants[0]
    if len(v['train_ids']) != payload['n_train'] or len(v['test_ids']) != payload['n_test']:
        raise ValueError('Official partition counts do not match metadata')
    split = EvaluationSplit(tuple(v['train_ids']), tuple(v['test_ids']),
                            f'laion-shared-{name}-variant{variant_id}@{lock["commit"]}', digest)
    return RecordedSplit(split, dataset_release, lock['commit'], 'shared')


@dataclass(frozen=True)
class TrialPartition:
    train: np.ndarray
    test: np.ndarray
    excluded: np.ndarray
    missing_train_ids: tuple[str, ...]
    missing_test_ids: tuple[str, ...]


def partition_trials(image_ids, *, split: RecordedSplit, require_complete: bool = True) -> TrialPartition:
    """Exact label matching; all repeats stay together. No partition truncation.

    require_complete=False is for coverage diagnostics only; it reports absent
    images rather than creating a smaller split. Non-pool trials are explicitly
    returned as excluded, as in the upstream mask helper's two false masks.
    """
    if isinstance(image_ids, (str, bytes)):
        raise ValueError('Supply a sequence of trial labels, not one string')
    labels = tuple(image_ids)
    if not labels or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError('Trial labels must be nonempty strings')
    if type(require_complete) is not bool:
        raise ValueError('require_complete must be boolean')
    train_ids, test_ids = set(split.split.train_ids), set(split.split.test_ids)
    observed = set(labels)
    missing_train = tuple(x for x in split.split.train_ids if x not in observed)
    missing_test = tuple(x for x in split.split.test_ids if x not in observed)
    if require_complete and (missing_train or missing_test):
        raise ValueError(f'Incomplete split coverage: {len(missing_train)} train and {len(missing_test)} test images missing')
    train = np.array([x in train_ids for x in labels], dtype=bool)
    test = np.array([x in test_ids for x in labels], dtype=bool)
    return TrialPartition(train, test, ~(train | test), missing_train, missing_test)


def validate_holdout_family(splits) -> None:
    """Validate a declared five-fold family: same pool, disjoint exhaustive tests.

    Does not generate folds, average scores, or mix cross-validation with layer
    selection. Call separately for cluster and random families.
    """
    splits = tuple(splits)
    if len(splits) != 5 or len({s.split.protocol_id for s in splits}) != 5:
        raise ValueError('Require five distinct declared folds')
    if len({(s.dataset_release,s.source_revision,s.pool) for s in splits}) != 1:
        raise ValueError('Fold source/release/pool must agree')
    pool = set(splits[0].split.train_ids) | set(splits[0].split.test_ids)
    heldout = set()
    for s in splits:
        train, test = set(s.split.train_ids), set(s.split.test_ids)
        if train | test != pool or heldout & test:
            raise ValueError('Fold pools differ or held-out images overlap')
        heldout |= test
    if heldout != pool:
        raise ValueError('Held-out folds must cover the pool exactly once')
