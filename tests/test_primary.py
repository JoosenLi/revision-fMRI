"""Primary split order/policy boundaries and train-only layer selection."""
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
from revision_fmri import (DeepNSDProtocol, ReproductionPipeline, FeatureSet, ResponseSet,
                          RecordedSplit, EvaluationSplit, load_primary_laion_split)
from revision_fmri.config import load_config
from revision_fmri.evaluation import even_odd_split, deepnsd_shared1000_split
from revision_fmri.provenance import metadata_digest
from revision_fmri import primary, MATCHED3_POLICY, TrialSession, TrialPresentation, aggregate_matched3
from revision_fmri.evaluation import EncodingPredictions
from revision_fmri.provenance import array_digest


@pytest.fixture
def pool(monkeypatch, tmp_path):
    # Deliberately non-lexicographic; this fixture catches ad-hoc runtime sorting.
    ids = tuple(f'image-{i}' for i in range(1121))
    config = load_config('configs/paths.example.yaml')
    config = replace(config,storage={**config.storage,'results_root':tmp_path})
    original = RecordedSplit(EvaluationSplit(ids, ('ood-a', 'ood-b'), 'source', primary.SOURCE_SHA256),
                             'synthetic', primary.SOURCE_REVISION, 'shared')
    monkeypatch.setattr(primary, 'CANONICAL_ORDER_SHA256', metadata_digest(ids))
    monkeypatch.setattr(primary, 'load_official_split', lambda *a, **k: original)
    return config, ids, original


def test_primary_counts_order_no_ood_and_required_repeat_policy(pool):
    config, ids, _ = pool
    split = load_primary_laion_split(config, dataset_release='synthetic', repeat_policy=MATCHED3_POLICY)
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
    with pytest.raises(ValueError, match='repeat_policy'):
        load_primary_laion_split(config, dataset_release='synthetic', repeat_policy='all-repeat')



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
        load_primary_laion_split(config, dataset_release='synthetic', repeat_policy=MATCHED3_POLICY)


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
    sessions = [TrialSession(f's{rep}',rep,
                tuple(TrialPresentation(f's{rep}-t{i}',image,1,float(i),True) for i,image in enumerate(ids)),
                y.copy()) for rep in range(3)]
    response = aggregate_matched3(sessions,config=config,image_ids=ids,voxel_ids=tuple(f'v{i}' for i in range(5)),
        subject_id='s',roi_id='r',space='space',provenance={'dataset_release':'synthetic'},
        repeat_policy=MATCHED3_POLICY,normalization_policy='train_only_session_v1',training_image_ids=ids[::2])
    split=load_primary_laion_split(config,dataset_release='synthetic',repeat_policy=MATCHED3_POLICY)
    pipeline = ReproductionPipeline(config, DeepNSDProtocol.jl_recomputed(split.split))
    # Test selector orchestration with a cheap deterministic encoder; do not run
    # a 6018-column ridge fit as a unit test. Real source SRP/ridge has separate fixtures.
    class ToyEncoder:
        def fit_predict(self, feature, response, split, *, alphas):
            values=np.column_stack([feature.values, feature.values[:,0]])
            return EncodingPredictions(values[::2],values[1::2],np.ones(5),{},split.train_ids,split.test_ids)
    pipeline._encoder=ToyEncoder()
    args = dict(features=features, responses=response, dataset_release='synthetic',
                repeat_policy=MATCHED3_POLICY)
    report = pipeline.evaluate_primary_laion_model(**args)
    perturbed = response.values.copy()
    perturbed[1::2] = rng.normal(size=perturbed[1::2].shape)
    manifest=json.loads(response.provenance['trial_selection_manifest'])
    manifest['response_values_sha256']=array_digest(perturbed)
    changed_response=replace(response,values=perturbed,provenance={**response.provenance,
        'trial_selection_manifest':json.dumps(manifest),'trial_selection_sha256':metadata_digest(manifest),
        'trial_selection_file':f'trial_selections/{metadata_digest(manifest)}.json'})
    changed = pipeline.evaluate_primary_laion_model(**{**args, 'responses': changed_response})
    assert report.train_image_ids == ids[::2] and report.test_image_ids == ids[1::2]
    assert report.layer_training_scores == changed.layer_training_scores
    assert {k:v.selected_layer for k,v in report.metrics.items()} == {k:v.selected_layer for k,v in changed.metrics.items()}
    assert report.metrics['crsa'].test_score != changed.metrics['crsa'].test_score
    assert report.evaluation_provenance['repeat_policy'] == MATCHED3_POLICY
    assert json.loads(report.evaluation_provenance['projection_spec']) == primary.primary_projection_spec(split.split)
    assert report.evaluation_provenance['trial_selection_sha256'] == response.provenance['trial_selection_sha256']
    bad_pipeline=ReproductionPipeline(config,DeepNSDProtocol('wrong',5920,'fixed-original'))
    with pytest.raises(ValueError,match='projection'):
        bad_pipeline.evaluate_primary_laion_model(**args)
    assert report.evaluation_provenance['analysis_role'] == 'primary_replication'
    assert 'no test-dependent filter' in report.evaluation_provenance['selection']
    with pytest.raises(ValueError, match='repeat_policy'):
        pipeline.evaluate_primary_laion_model(**{**args, 'repeat_policy': 'all-repeats'})
    with pytest.raises(TypeError):
        pipeline.evaluate_primary_laion_model(**{k:v for k,v in args.items() if k != 'repeat_policy'})
    bad = replace(response, provenance={**response.provenance, 'dataset_release': 'other'})
    with pytest.raises(ValueError, match='dataset_release'):
        pipeline.evaluate_primary_laion_model(**{**args, 'responses': bad})
    with pytest.raises(ValueError, match='exactly partition'):
        split = load_primary_laion_split(config, dataset_release='synthetic', repeat_policy=MATCHED3_POLICY)
        split.split.validate_coverage(ids[:-1])


@pytest.mark.parametrize('n,expected', [(1000,5920),(1121,6018)])
def test_primary_jl_full_split_formula(n,expected):
    ids=tuple(f'i{i}' for i in range(n))
    split=even_odd_split(ids,protocol_id='test',source_digest='a'*64)
    spec=primary.primary_projection_spec(split)
    assert spec=={'policy':'jl_recomputed_eps0.1','n_samples':n,'eps':0.1,
                  'n_components':expected,'seed':0,'density':'auto','upsampling':True}
    protocol=DeepNSDProtocol.jl_recomputed(split)
    assert protocol.n_components==expected
    assert protocol.projection_policy=='jl_recomputed_eps0.1'
