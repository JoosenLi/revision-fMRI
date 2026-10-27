"""F10: source-group counts, cross-session preference and deterministic fill."""
from dataclasses import replace
import json
import numpy as np
import pytest
from revision_fmri import TrialSession, TrialPresentation, aggregate_matched3, validate_matched3_responses
from revision_fmri.repeats import MATCHED3_POLICY, CHRONOLOGICAL_POLICY, VOLUME_ORDER
from revision_fmri.config import load_config
from revision_fmri.provenance import metadata_digest
from revision_fmri.response import normalize_session


@pytest.fixture
def example(tmp_path):
    ids=('shared_12rep_LAION_a.jpg','shared_12rep_THINGS+_b.jpg','shared_4rep_NSD_c.jpg')
    a,b,c=ids
    labels={1:[a,a,b,b,c,c,c,'unique','ood'],2:[a,a,b,b,'unique'],
            3:[a,a,'unique'],4:[a,'unique']}
    sessions=[]
    for number,images in labels.items():
        trials=tuple(TrialPresentation(f's{number}-v{i}',name,10-i,None,True,
                     ordering_source=VOLUME_ORDER,volume_index=i) for i,name in enumerate(images))
        values=np.column_stack([np.arange(len(images))**2+number,np.arange(len(images))*3-number]).astype(float)
        sessions.append(TrialSession(f's{number}',number,trials,values))
    cfg=load_config('configs/paths.example.yaml')
    cfg=replace(cfg,storage={**cfg.storage,'results_root':tmp_path})
    kw=dict(config=cfg,image_ids=ids,voxel_ids=('v1','v2'),subject_id='s',roi_id='r',space='T1w',
            provenance={'dataset_release':'synthetic'},repeat_policy=MATCHED3_POLICY,
            normalization_policy='legacy_deepnsd_session_v1',training_image_ids=None)
    return sessions,kw


def test_prefer_distinct_then_fill_chronologically_and_normalize_all(example):
    sessions,kw=example
    out=aggregate_matched3(sessions[::-1],**kw)
    manifest=validate_matched3_responses(out)
    rows=manifest['images']
    assert [[t['trial_id'] for t in r['selected_trials']] for r in rows]==[
        ['s1-v0','s2-v0','s3-v0'], ['s1-v2','s1-v3','s2-v2'], ['s1-v4','s1-v5','s1-v6']]
    assert [r['filled_within_session'] for r in rows]==[False,True,True]
    assert manifest['selection_summary']=={
        'three_distinct_sessions':1,'filled_within_session':2,'images_with_repeated_session':2,
        'by_source_group':{'LAION':{'three_distinct_sessions':1,'filled_within_session':0},
                           'THINGS+':{'three_distinct_sessions':0,'filled_within_session':1},
                           'NSD':{'three_distinct_sessions':0,'filled_within_session':1},
                           'other':{'three_distinct_sessions':0,'filled_within_session':0}}}
    # Entire valid sessions include unique + OOD + unselected repeats before selection.
    norm=[normalize_session(s.values,policy='legacy_deepnsd_session_v1').values for s in sessions]
    expected=np.array([(norm[0][0].astype(float)+norm[1][0]+norm[2][0])/3,
                       (norm[0][2].astype(float)+norm[0][3]+norm[1][2])/3,
                       norm[0][[4,5,6]].astype(float).mean(axis=0)])
    np.testing.assert_array_equal(out.values,expected)
    again=aggregate_matched3(sessions,**kw)
    assert out.provenance==again.provenance
    old=aggregate_matched3(sessions,**{**kw,'repeat_policy':CHRONOLOGICAL_POLICY})
    assert [t['trial_id'] for t in validate_matched3_responses(old)['images'][0]['selected_trials']]==[
        's1-v0','s1-v1','s2-v0']
    assert out.provenance['trial_selection_sha256'] != old.provenance['trial_selection_sha256']


def test_invalid_earliest_is_excluded_and_less_than_three_still_fails(example):
    sessions,kw=example
    trials=list(sessions[0].trials);trials[0]=replace(trials[0],valid=False)
    sessions[0]=replace(sessions[0],trials=tuple(trials))
    out=aggregate_matched3(sessions,**kw)
    row=validate_matched3_responses(out)['images'][0]
    assert row['selected_trials'][0]['trial_id']=='s1-v1'
    trials[4]=replace(trials[4],valid=False) # NSD now has only two valid trials total.
    sessions[0]=replace(sessions[0],trials=tuple(trials))
    before=list(kw['config'].storage['results_root'].rglob('*.json'))
    with pytest.raises(ValueError,match='Fewer than 3'):
        aggregate_matched3(sessions,**kw)
    assert list(kw['config'].storage['results_root'].rglob('*.json'))==before


@pytest.mark.parametrize('kind',['fill_flag','summary','source','selection'])
def test_manifest_verifies_distinct_session_rule_and_summary(example,kind):
    sessions,kw=example
    out=aggregate_matched3(sessions,**kw)
    manifest=json.loads(out.provenance['trial_selection_manifest'])
    if kind=='fill_flag': manifest['images'][0]['filled_within_session']=True
    elif kind=='summary': manifest['selection_summary']['three_distinct_sessions']=99
    elif kind=='source': manifest['images'][0]['source_group']='NSD'
    else: manifest['images'][0]['selected_trials']=manifest['images'][0]['valid_presentations'][:3]
    digest=metadata_digest(manifest)
    changed=replace(out,provenance={**out.provenance,'trial_selection_manifest':json.dumps(manifest),
         'trial_selection_sha256':digest,'trial_selection_file':f'trial_selections/{digest}.json'})
    with pytest.raises(ValueError): validate_matched3_responses(changed)


def test_three_plus_one_prefers_both_sessions_before_fill(example):
    sessions,kw=example
    a=kw['image_ids'][0]
    trials=tuple(TrialPresentation(f's1-v{i}',a,1,None,True,ordering_source=VOLUME_ORDER,volume_index=i)
                 for i in range(3))
    first=TrialSession('s1',1,trials,np.array([[1.,4.],[3.,2.],[6.,1.]]))
    last=TrialSession('s2',2,(TrialPresentation('s2-v0',a,1,None,True,ordering_source=VOLUME_ORDER,volume_index=0),
                             TrialPresentation('s2-v1','unique',1,None,True,ordering_source=VOLUME_ORDER,volume_index=1)),
                      np.array([[5.,4.],[2.,8.]]))
    out=aggregate_matched3([last,first],**{**kw,'image_ids':(a,)})
    row=validate_matched3_responses(out)['images'][0]
    assert [t['trial_id'] for t in row['selected_trials']]==['s1-v0','s1-v1','s2-v0']
    assert row['filled_within_session'] is True
