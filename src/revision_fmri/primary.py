"""Primary LAION adaptation of C1 even/odd split-half; no default repeat policy."""
from .config import LocalConfig
from .evaluation import even_odd_split
from .official_splits import load_official_split
from .provenance import metadata_digest
from .splits import RecordedSplit
from .repeats import MATCHED3_POLICY

PRIMARY_PROTOCOL = 'laion_splithalf_even_odd_v1'
CANONICAL_ORDER_SHA256 = '8299304097406a108d94db054763edea759ccfe9d1d4446a7ba6e3dd50006db9'
SOURCE_SHA256 = 'd592f700f2ce32c9733800ba524da3b4168af3e53365b25e70aaa993769df842'
SOURCE_REVISION = 'cf82765ced029de81c3bec7254b8cbd6a65e3686'


def load_primary_laion_split(config: LocalConfig, *, dataset_release: str,
                             repeat_policy: str) -> RecordedSplit:
    """Use the published shared ood.json variant-0 train_ids as the non-OOD pool.

    Only the image universe/order is borrowed; the official OOD assignment is
    not used as the primary partition. Require an explicit aggregation policy
    label matching the owner-decided matched-three policy; aggregation is handled
    by the shared data API, not the split loader.
    """
    if repeat_policy != MATCHED3_POLICY:
        raise ValueError('Primary repeat_policy must explicitly be matched3_distinct_sessions_v1')
    original = load_official_split(config, name='ood', dataset_release=dataset_release)
    ids = original.split.train_ids
    if (original.source_revision != SOURCE_REVISION or original.split.source_digest != SOURCE_SHA256
            or len(ids) != 1121 or metadata_digest(ids) != CANONICAL_ORDER_SHA256):
        raise ValueError('Primary canonical image order/source does not match the pinned 1121 pool')
    digest = _primary_split_digest()
    split = even_odd_split(ids, protocol_id=PRIMARY_PROTOCOL, source_digest=digest)
    return RecordedSplit(split, dataset_release, SOURCE_REVISION, 'shared_non_ood')


PRIMARY_PROJECTION_POLICY = 'jl_recomputed_eps0.1'


def primary_projection_spec(split):
    """C3 projects before the split: derive JL size from the complete image universe."""
    from .features import jl_dimensions
    n_samples = len(split.train_ids) + len(split.test_ids)
    return {'policy': PRIMARY_PROJECTION_POLICY, 'n_samples': n_samples, 'eps': 0.1,
            'n_components': jl_dimensions(n_samples, eps=0.1), 'seed': 0,
            'density': 'auto', 'upsampling': True}


PRIMARY_NORMALIZATION = 'legacy_deepnsd_session_v1'
NORMALIZATION_EXCEPTION = 'c6_legacy_session_zscore'


def _primary_split_digest():
    return metadata_digest({'protocol_id': PRIMARY_PROTOCOL,
                            'canonical_order_sha256': CANONICAL_ORDER_SHA256,
                            'source_file_sha256': SOURCE_SHA256,
                            'source_revision': SOURCE_REVISION,
                            'repeat_policy': MATCHED3_POLICY})


def validate_primary_normalization_exception(split, responses, exception):
    """Narrow C6 exception: exact primary split/manifest, never a general bypass."""
    from .repeats import validate_matched3_responses
    if (exception != NORMALIZATION_EXCEPTION or split.protocol_id != PRIMARY_PROTOCOL
            or len(split.train_ids) != 561 or len(split.test_ids) != 560
            or split.source_digest != _primary_split_digest()):
        raise ValueError('Normalization exception is restricted to the canonical primary protocol')
    ids = tuple(x for pair in zip(split.train_ids, split.test_ids) for x in pair) + split.train_ids[-1:]
    if metadata_digest(ids) != CANONICAL_ORDER_SHA256:
        raise ValueError('Normalization exception requires canonical primary image order')
    if (responses.provenance.get('normalization') != PRIMARY_NORMALIZATION
            or responses.provenance.get('aggregation') != MATCHED3_POLICY):
        raise ValueError('Primary requires legacy_deepnsd_session_v1 and distinct-session matched3')
    manifest = validate_matched3_responses(responses)
    if manifest['normalization_training_ids_sha256'] != metadata_digest(None):
        raise ValueError('Primary normalization must include all valid session trials')
    return manifest
