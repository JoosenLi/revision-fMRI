"""Scientific boundary checks: union order, units, thresholds and fitting scope."""
from dataclasses import replace
import json
import numpy as np
import pytest
from revision_fmri import (VoxelMap, OTCProtocol, select_otc, normalize_session,
                           ResponseSet)


def objects():
    def m(values, kind='binary_mask'):
        return VoxelMap(tuple(f'v{i}' for i in range(5)), np.array(values),
                        'sub-01', 'T1w', 'grid-hash', kind, {'source': 'synthetic'})
    p = OTCProtocol('synthetic-otc-v1', ('category-a', 'category-b'), .2, 'gt', {
        'category_threshold': 'Synthetic binary masks; no t-map claim',
        'missing_regions': 'Synthetic test only', 'stream_mapping': 'Synthetic test only',
        'reliability_scope': 'Synthetic independent data'})
    return dict(general=m([1,1,0,0,0]), lateral=m([1,0,0,0,0]), ventral=m([0,1,0,0,0]),
                categories={'category-a':m([1,0,1,0,0]),'category-b':m([0,0,1,1,0])},
                reliability=m([.3,.2,.3,.1,.9],'ncsnr'), protocol=p)


def test_union_outside_general_overlap_threshold_and_apply():
    args = objects()
    selected = select_otc(**args)
    np.testing.assert_array_equal(selected.selected, [1,0,1,0,0])
    assert selected.counts == dict(core=2, category_union=3, overlap=1,
                                  candidate_union=4, excluded_reliability=2, selected=2)
    assert json.loads(selected.provenance['roi_counts'])['selected'] == 2
    r = ResponseSet(('a','b'),selected.voxel_ids,np.arange(10).reshape(2,5),
                    'sub-01','input','T1w',(1,1),{'grid_sha256':'grid-hash'})
    out = selected.apply(r)
    np.testing.assert_array_equal(out.values, [[0,2],[5,7]])
    assert out.voxel_ids == ('v0','v2') and out.roi_id == 'synthetic-otc-v1'
    ge = select_otc(**{**args,'protocol':replace(args['protocol'], threshold_operator='ge')})
    np.testing.assert_array_equal(ge.selected, [1,1,1,0,0])
    with pytest.raises(ValueError,match='grid'):
        selected.apply(replace(r,provenance={'grid_sha256':'other-grid'}))


@pytest.mark.parametrize('field,value', [('subject_id','sub-02'),('space','surface'),
                                       ('grid_sha256','other'),('voxel_ids',('v4','v3','v2','v1','v0'))])
def test_map_alignment(field,value):
    args=objects();args['lateral']=replace(args['lateral'],**{field:value})
    with pytest.raises(ValueError,match='share'):
        select_otc(**args)


def test_no_silent_missing_regions_or_wrong_units():
    args=objects();args['categories'].pop('category-b')
    with pytest.raises(ValueError,match='exactly'):
        select_otc(**args)
    with pytest.raises(ValueError,match='NC percent'):
        replace(objects()['reliability'],kind='nc_percent')
    args=objects();args['reliability']=replace(args['reliability'],values=np.zeros(5))
    with pytest.raises(ValueError,match='No voxels'):
        select_otc(**args)


def test_legacy_normalization_axis_ddof_and_precision():
    x=np.array([[1.,10.],[2.,20.],[3.,30.]])
    out=normalize_session(x,policy='legacy_deepnsd_session_v1')
    expected=np.repeat(np.array([-np.sqrt(1.5),0,np.sqrt(1.5)])[:,None],2,axis=1)
    np.testing.assert_allclose(out.values,expected,rtol=1e-7)
    assert out.values.dtype == np.float32
    assert out.provenance['normalization_scope']=='all session trials'
    with pytest.raises(ValueError,match='all session'):
        normalize_session(x,policy='legacy_deepnsd_session_v1',fit_rows=[0,1])


def test_training_normalization_does_not_fit_heldout_values():
    x=np.array([[1.,10.],[3.,30.],[100.,200.]])
    first=normalize_session(x,policy='train_only_session_v1',fit_rows=[0,1])
    x[2]=[-300,800]
    second=normalize_session(x,policy='train_only_session_v1',fit_rows=[0,1])
    np.testing.assert_array_equal(first.values[:2],second.values[:2])
    np.testing.assert_array_equal(first.values[:2],[[-1,-1],[1,1]])
    assert first.provenance['normalization_fit_values_sha256']==second.provenance['normalization_fit_values_sha256']
    assert not np.array_equal(first.values[2],second.values[2])


@pytest.mark.parametrize('rows',[None,[],[0,0],[-1,0],[0,3],[0.,1.]])
def test_invalid_normalization_fit_rows(rows):
    with pytest.raises(ValueError,match='training row'):
        normalize_session(np.ones((3,2)),policy='train_only_session_v1',fit_rows=rows)


def test_independent_evaluation_rejects_legacy_response_scope():
    from revision_fmri.evaluation import evaluate_layers
    r=ResponseSet(('a','b'),('v0',),np.ones((2,1)),'s','r','T1w',(1,1),
                  {'normalization':'legacy_deepnsd_session_v1'})
    with pytest.raises(ValueError,match='Legacy'):
        evaluate_layers((),r,None,None,alphas=())
