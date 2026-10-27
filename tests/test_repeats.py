"""Chronological matched-three selection and durable normalized-response provenance."""
from dataclasses import replace
import json
import numpy as np
import pytest
from revision_fmri import (TrialSession, TrialPresentation, aggregate_matched3, MATCHED3_POLICY,
                          validate_matched3_responses, write_trial_selection_manifest)
from revision_fmri.config import load_config
from revision_fmri.provenance import metadata_digest
from revision_fmri.response import normalize_session


@pytest.fixture
def fixture(tmp_path):
    config = load_config('configs/paths.example.yaml')
    config = replace(config, storage={**config.storage, 'results_root': tmp_path})
    sessions = []
    for rank in (2, 1):
        # Metadata/input storage order is intentionally not acquisition order.
        trials = tuple(TrialPresentation(f's{rank}-t{i}', image, run, onset, valid)
                       for i,(image,run,onset,valid) in enumerate([
                           ('a',2,0.,True), ('b',1,4.,True), ('a',1,2.,True),
                           ('a',1,0.,False), ('b',2,2.,True), ('a',1,3.,True)]))
        values = np.array([[9,7],[2,3],[4,8],[np.nan,np.nan],[6,1],[12,4]], dtype=float) * rank
        sessions.append(TrialSession(f's{rank}', rank, trials, values))
    kwargs = dict(config=config,image_ids=('b','a'),voxel_ids=('v1','v2'),subject_id='s',roi_id='r',space='T1w',
                  provenance={'dataset_release':'synthetic'},repeat_policy=MATCHED3_POLICY,
                  normalization_policy='train_only_session_v1',training_image_ids=('a',))
    return sessions, kwargs


def test_chronology_normalize_before_average_manifest_and_span(fixture):
    sessions, kwargs = fixture
    response = aggregate_matched3(sessions, **kwargs)
    manifest = validate_matched3_responses(response)
    assert response.repeat_counts == (3,3)
    a, b = manifest['images'][1], manifest['images'][0]
    assert [t['trial_id'] for t in a['selected_trials']] == ['s1-t2','s1-t5','s1-t0']
    assert [t['trial_id'] for t in b['selected_trials']] == ['s1-t1','s1-t4','s2-t1']
    assert a['session_span']['n_sessions'] == 1 and b['session_span']['n_sessions'] == 2
    # All valid rows are normalized first; a's training rows are 0,1,3 in chronological order.
    order = [2,5,1,0,4]
    norm = normalize_session(sessions[1].values[order], policy='train_only_session_v1',fit_rows=[0,1,3]).values
    np.testing.assert_array_equal(response.values[1], norm[[0,1,3]].astype('float64').mean(axis=0))
    np.testing.assert_array_equal(response.values[0], norm[[2,4,2]].astype('float64').mean(axis=0))
    artifact = kwargs['config'].storage['results_root']/response.provenance['trial_selection_file']
    document = json.loads(artifact.read_text())
    assert document['content_sha256'] == metadata_digest(document['manifest'])
    assert response.provenance['trial_selection_sha256'] == document['content_sha256']
    again = aggregate_matched3(sessions[::-1], **kwargs)
    assert again.provenance == response.provenance
    assert manifest['seed'] is None


def test_missing_valid_repeats_fails_without_artifact(fixture):
    sessions, kwargs = fixture
    with pytest.raises(ValueError, match='Fewer than 3'):
        aggregate_matched3(sessions[1:], **kwargs)  # b has only two valid presentations.
    assert not list(kwargs['config'].storage['results_root'].rglob('*.json'))


@pytest.mark.parametrize('kind', ['tie', 'duplicate_id', 'nonfinite_valid', 'rank', 'policy', 'normalization'])
def test_invalid_metadata_or_response_fails(fixture, kind):
    sessions, kwargs = fixture
    first = sessions[0]
    if kind == 'tie':
        trials = list(first.trials);trials[0] = replace(trials[0],run_order=1,onset=2.)
        sessions[0] = replace(first,trials=tuple(trials))
    elif kind == 'duplicate_id':
        trials = list(first.trials);trials[0] = replace(trials[0],trial_id=trials[1].trial_id)
        sessions[0] = replace(first,trials=tuple(trials))
    elif kind == 'nonfinite_valid':
        values=first.values.copy();values[0,0]=np.nan;sessions[0]=replace(first,values=values)
    elif kind == 'rank':
        sessions[0]=replace(first,session_order=1)
    elif kind == 'policy': kwargs['repeat_policy']='automatic'
    else: kwargs['normalization_policy']='none'
    with pytest.raises(ValueError): aggregate_matched3(sessions, **kwargs)


def test_manifest_tamper_repeat_count_and_response_binding(fixture):
    sessions, kwargs = fixture
    response = aggregate_matched3(sessions, **kwargs)
    with pytest.raises(ValueError, match='exactly three'):
        validate_matched3_responses(replace(response, repeat_counts=(2,3)))
    with pytest.raises(ValueError, match='does not match'):
        validate_matched3_responses(replace(response, values=response.values+1))
    manifest=json.loads(response.provenance['trial_selection_manifest'])
    manifest['images'][0]['selected_trials']=manifest['images'][0]['valid_presentations'][1:4]
    provenance={**response.provenance,'trial_selection_manifest':json.dumps(manifest),
                'trial_selection_sha256':metadata_digest(manifest),
                'trial_selection_file':f'trial_selections/{metadata_digest(manifest)}.json'}
    with pytest.raises(ValueError, match='earliest'):
        validate_matched3_responses(replace(response, provenance=provenance))
    path=kwargs['config'].storage['results_root']/response.provenance['trial_selection_file']
    path.write_text('do not overwrite')
    with pytest.raises(ValueError, match='no overwrite'):
        write_trial_selection_manifest(kwargs['config'],json.loads(response.provenance['trial_selection_manifest']))
    assert path.read_text()=='do not overwrite'


def test_all_invalid_session_is_audited_without_contributing(fixture):
    sessions, kwargs = fixture
    invalid = TrialSession('s0',0,(TrialPresentation('invalid','a',1,0.,False),),np.array([[np.nan,np.nan]]))
    base=aggregate_matched3(sessions,**kwargs)
    out=aggregate_matched3([invalid,*sessions],**kwargs)
    np.testing.assert_array_equal(out.values,base.values)
    manifest=validate_matched3_responses(out)
    assert manifest['sessions'][0]['n_valid']==0
    assert manifest['session_span']['first_session']=='s1'


def test_mixed_order_units_fail(fixture):
    sessions,kwargs=fixture
    trials=list(sessions[0].trials)
    trials[0]=replace(trials[0],onset=None,beta_index=0,
                      ordering_source='glmsingle_beta_index_order_v1')
    sessions[0]=replace(sessions[0],trials=tuple(trials))
    with pytest.raises(ValueError,match='Mixed ordering'):
        aggregate_matched3(sessions,**kwargs)
