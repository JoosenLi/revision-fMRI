"""Primary split order/policy boundaries and train-only layer selection."""
from dataclasses import replace
import numpy as np
import pytest
from revision_fmri import (DeepNSDProtocol, ReproductionPipeline, FeatureSet, ResponseSet,
                          RecordedSplit, EvaluationSplit, load_primary_laion_split)
from revision_fmri.config import load_config
from revision_fmri.evaluation import even_odd_split, deepnsd_shared1000_split
from revision_fmri.provenance import metadata_digest
from revision_fmri import primary


@pytest.fixture
def pool(monkeypatch):
    # Deliberately non-lexicographic; this fixture catches ad-hoc runtime sorting.
    ids = tuple(f'image-{i}' for i in range(1121))
    config = load_config('configs/paths.example.yaml')
    original = RecordedSplit(EvaluationSplit(ids, ('ood-a', 'ood-b'), 'source', primary.SOURCE_SHA256),
                             'synthetic', primary.SOURCE_REVISION, 'shared')
    monkeypatch.setattr(primary, 'CANONICAL_ORDER_SHA256', metadata_digest(ids))
    monkeypatch.setattr(primary, 'load_official_split', lambda *a, **k: original)
    return config, ids, original


def test_primary_counts_order_no_ood_and_required_repeat_policy(pool):
    config, ids, _ = pool
    split = load_primary_laion_split(config, dataset_release='synthetic', repeat_policy='explicit-policy')
    assert split.split.train_ids == ids[::2] and len(split.split.train_ids) == 561
    assert split.split.test_ids == ids[1::2] and len(split.split.test_ids) == 560
    assert split.pool == 'shared_non_ood'
    assert split.split.protocol_id == 'laion_splithalf_even_odd_v1'
    assert not set(split.split.train_ids + split.split.test_ids) & {'ood-a', 'ood-b'}
    with pytest.raises(TypeError):
        load_primary_laion_split(config, dataset_release='synthetic')
    for policy in (None, '', ' '):
        with pytest.raises(ValueError, match='repeat_policy'):
            load_primary_laion_split(config, dataset_release='synthetic', repeat_policy=policy)
    other = load_primary_laion_split(config, dataset_release='synthetic', repeat_policy='different-policy')
    assert split.split.source_digest != other.split.source_digest
    assert split.split.train_ids == other.split.train_ids


@pytest.mark.parametrize('change', ['order', 'count', 'hash', 'revision'])
def test_canonical_changes_fail_closed(pool, monkeypatch, change):
    config, ids, original = pool
    changed = original
    if change == 'order':
        changed = replace(original, split=replace(original.split, train_ids=ids[::-1]))
    elif change == 'count':
        changed = replace(original, split=replace(original.split, train_ids=ids[:-1]))
    elif change == 'hash':
        changed = replace(original, split=replace(original.split, source_digest='b'*64))
    else:
        changed = replace(original, source_revision='other')
    monkeypatch.setattr(primary, 'load_official_split', lambda *a, **k: changed)
    with pytest.raises(ValueError, match='canonical'):
        load_primary_laion_split(config, dataset_release='synthetic', repeat_policy='explicit')


def test_nsd1000_behavior_unchanged_and_shared_algorithm():
    ids = tuple(str(i) for i in range(1000))
    old = deepnsd_shared1000_split(ids, source_digest='source')
    assert old.train_ids == ids[::2] and old.test_ids == ids[1::2]
    assert old.protocol_id == 'deepnsd-shared1000-even-odd-v1'
    assert old.source_digest == 'source'
    assert old == even_odd_split(ids, protocol_id=old.protocol_id, source_digest='source')
    with pytest.raises(ValueError, match='exactly 1000'):
        deepnsd_shared1000_split(ids[:-1], source_digest='source')


def test_primary_uses_existing_train_only_selector_and_reports_test_half(pool):
    config, ids, _ = pool
    rng = np.random.default_rng(74)
    x = rng.normal(size=(1121, 4)).astype('float32')
    y = x @ rng.normal(size=(4, 5)) + rng.normal(size=(1121, 5)) * .1
    features = [FeatureSet(ids, 'synthetic', name, values, 'raw_flattened', {'fixture': 'true'})
                for name, values in [('signal', x), ('other', rng.normal(size=x.shape).astype('float32'))]]
    response = ResponseSet(ids, tuple(f'v{i}' for i in range(5)), y, 's', 'r', 'space',
                           (3,)*1121, {'dataset_release': 'synthetic', 'aggregation': 'caller-matched-three'})
    pipeline = ReproductionPipeline(config, DeepNSDProtocol('synthetic-only', 3, 'test-only explicit SRP'))
    args = dict(features=features, responses=response, dataset_release='synthetic',
                repeat_policy='caller-matched-three')
    report = pipeline.evaluate_primary_laion_model(**args)
    perturbed = y.copy()
    perturbed[1::2] = rng.normal(size=perturbed[1::2].shape)
    changed = pipeline.evaluate_primary_laion_model(**{**args, 'responses': replace(response, values=perturbed)})
    assert report.train_image_ids == ids[::2] and report.test_image_ids == ids[1::2]
    assert report.layer_training_scores == changed.layer_training_scores
    assert {k:v.selected_layer for k,v in report.metrics.items()} == {k:v.selected_layer for k,v in changed.metrics.items()}
    assert report.metrics['crsa'].test_score != changed.metrics['crsa'].test_score
    assert report.evaluation_provenance['repeat_policy'] == 'caller-matched-three'
    assert report.evaluation_provenance['analysis_role'] == 'primary_replication'
    assert 'no test-dependent filter' in report.evaluation_provenance['selection']
    with pytest.raises(ValueError, match='aggregation'):
        pipeline.evaluate_primary_laion_model(**{**args, 'repeat_policy': 'all-repeats'})
    with pytest.raises(TypeError):
        pipeline.evaluate_primary_laion_model(**{k:v for k,v in args.items() if k != 'repeat_policy'})
    bad = replace(response, provenance={**response.provenance, 'dataset_release': 'other'})
    with pytest.raises(ValueError, match='dataset_release'):
        pipeline.evaluate_primary_laion_model(**{**args, 'responses': bad})
    with pytest.raises(ValueError, match='exactly partition'):
        split = load_primary_laion_split(config, dataset_release='synthetic', repeat_policy='explicit')
        split.split.validate_coverage(ids[:-1])
