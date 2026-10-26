"""Primary LAION adaptation of C1 even/odd split-half; no default repeat policy."""
from .config import LocalConfig
from .evaluation import even_odd_split
from .official_splits import load_official_split
from .provenance import metadata_digest
from .splits import RecordedSplit

PRIMARY_PROTOCOL = 'laion_splithalf_even_odd_v1'
CANONICAL_ORDER_SHA256 = '8299304097406a108d94db054763edea759ccfe9d1d4446a7ba6e3dd50006db9'
SOURCE_SHA256 = 'd592f700f2ce32c9733800ba524da3b4168af3e53365b25e70aaa993769df842'
SOURCE_REVISION = 'cf82765ced029de81c3bec7254b8cbd6a65e3686'


def load_primary_laion_split(config: LocalConfig, *, dataset_release: str,
                             repeat_policy: str) -> RecordedSplit:
    """Use the published shared ood.json variant-0 train_ids as the non-OOD pool.

    Only the image universe/order is borrowed; the official OOD assignment is
    not used as the primary partition. Require an explicit aggregation policy
    label without choosing or performing trial selection/averaging here.
    """
    if not isinstance(repeat_policy, str) or not repeat_policy.strip():
        raise ValueError('Explicit repeat_policy is required; there is no default')
    original = load_official_split(config, name='ood', dataset_release=dataset_release)
    ids = original.split.train_ids
    if (original.source_revision != SOURCE_REVISION or original.split.source_digest != SOURCE_SHA256
            or len(ids) != 1121 or metadata_digest(ids) != CANONICAL_ORDER_SHA256):
        raise ValueError('Primary canonical image order/source does not match the pinned 1121 pool')
    digest = metadata_digest({'protocol_id': PRIMARY_PROTOCOL,
                              'canonical_order_sha256': CANONICAL_ORDER_SHA256,
                              'source_file_sha256': SOURCE_SHA256,
                              'source_revision': SOURCE_REVISION,
                              'repeat_policy': repeat_policy})
    split = even_odd_split(ids, protocol_id=PRIMARY_PROTOCOL, source_digest=digest)
    return RecordedSplit(split, dataset_release, SOURCE_REVISION, 'shared_non_ood')
