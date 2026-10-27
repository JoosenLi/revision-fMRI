"""G1: independent transcription of pinned C1/C6 orchestration, synthetic only.

DeepNSD 36f6a34727dccb5eec8fa49364b210a049a388d7:
main_analysis.py get_splithalf_xy/run_benchmarking; benchmark.py get_rdms/
get_splithalf_rdms. Uses original vendored numeric primitives, never package
projection/metric/selection/encoding wrappers on the expected-result path.
"""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection
from revision_fmri import FeatureSet, ResponseSet
from revision_fmri.evaluation import (evaluate_layers, even_odd_split,
                                     DeepNSDEncodingEvaluator, RELEASED_CODE_ALPHAS)
from revision_fmri._vendor.deepnsd.ridge_gcv_mod import RidgeCVMod
from revision_fmri._vendor.deepnsd.rsa import compare_rdms


def original_flow(raw_layers, responses, n_components):
    # Original responses are voxels x images; whole-brain RDM then half-slicing.
    response_data=responses.T
    brain_rdm=1-np.corrcoef(response_data.T)
    targets={'train':brain_rdm[::2,::2],'test':brain_rdm[1::2,1::2]}
    alphas=np.logspace(-1,5,7).tolist()
    scores={metric:{part:{} for part in ('train','test')} for metric in ('crsa','versa')}
    predictions={}
    for name,raw in raw_layers.items():
        # C3 fits before split; explicit tiny projection budget is a test-only fixture.
        projected=SparseRandomProjection(n_components=n_components,random_state=0).fit_transform(raw)
        scaler=StandardScaler()
        xtrain=scaler.fit_transform(projected[::2,:]);xtest=scaler.transform(projected[1::2,:])
        ridge=RidgeCVMod(alphas=alphas,store_cv_values=True,alpha_per_target=True,scoring='pearson_r')
        ridge.fit(xtrain,response_data[:,::2].T)
        indices=np.array([alphas.index(a) for a in ridge.alpha_])
        train=np.take_along_axis(ridge.cv_values_,indices[None,:,None],axis=2)[:,:,0]
        test=xtest.dot(ridge.coef_.T)+ridge.intercept_
        predictions[name]=(train,test,ridge.alpha_)
        for part,actual,pred in [('train',raw[::2,:],train),('test',raw[1::2,:],test)]:
            scores['crsa'][part][name]=compare_rdms(1-np.corrcoef(actual),targets[part])
            scores['versa'][part][name]=compare_rdms(1-np.corrcoef(pred),targets[part])
    # C1 train-selected maximum, first retained tie; C7 filter intentionally absent.
    selected={metric:max(scores[metric]['train'],key=scores[metric]['train'].get) for metric in scores}
    return scores,selected,predictions


def test_original_end_to_end_three_layers_nonzero_voxel_means():
    rng=np.random.default_rng(1201)
    latent=rng.normal(size=(80,8))
    raw={f'layer-{i}':(latent@rng.normal(size=(8,width))+rng.normal(size=(80,width))*noise).astype('float32')
         for i,(width,noise) in enumerate([(16,.1),(21,.5),(25,2.)])}
    y=latent@rng.normal(size=(8,30))+rng.normal(size=(80,30))*.2+np.arange(30)+10
    assert np.all(y.mean(axis=0)>5) # Centered LOO versus intercept-restored test matters.
    ids=tuple(f'image-{i}' for i in range(80))
    layers=tuple(FeatureSet(ids,'synthetic',name,x,'raw_flattened',{'seed':'1201'}) for name,x in raw.items())
    responses=ResponseSet(ids,tuple(f'v{i}' for i in range(30)),y,'s','roi','space',(3,)*80,{'seed':'1201'})
    split=even_odd_split(ids,protocol_id='synthetic',source_digest='synthetic')
    expected,selected,predictions=original_flow(raw,y,9)
    out=evaluate_layers(layers,responses,split,DeepNSDEncodingEvaluator(9,0,True,'float64'),alphas=RELEASED_CODE_ALPHAS)
    assert out.selected_layers==selected
    for metric in ('crsa','versa'):
        for name in raw:
            np.testing.assert_allclose(out.training_scores[metric][name],expected[metric]['train'][name],rtol=0,atol=1e-10)
        np.testing.assert_allclose(out.test_scores[metric],expected[metric]['test'][selected[metric]],rtol=0,atol=1e-10)
    for name,(train,test,alpha) in predictions.items():
        np.testing.assert_allclose(out.predictions[name].train_loo_centered,train,rtol=0,atol=1e-10)
        np.testing.assert_allclose(out.predictions[name].test_with_intercept,test,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(out.predictions[name].alpha_per_voxel,alpha)


@pytest.mark.parametrize('n_voxels',[1,10,11])
def test_roi_boundary_before_encoding(n_voxels):
    rng=np.random.default_rng(1202)
    ids=tuple(f'i{i}' for i in range(12))
    f=FeatureSet(ids,'m','layer',rng.normal(size=(12,5)).astype('float32'),'raw_flattened',{'fixture':'true'})
    r=ResponseSet(ids,tuple(f'v{i}' for i in range(n_voxels)),rng.normal(size=(12,n_voxels)),
                  's','roi','space',(3,)*12,{'fixture':'true'})
    split=even_odd_split(ids,protocol_id='synthetic',source_digest='synthetic')
    if n_voxels < 11:
        class ForbiddenEncoder:
            def fit_predict(self,*a,**k): pytest.fail('Tiny ROI must fail before fitting')
        with pytest.raises(ValueError,match='at least 11 voxels'):
            evaluate_layers((f,),r,split,ForbiddenEncoder(),alphas=RELEASED_CODE_ALPHAS)
    else:
        out=evaluate_layers((f,),r,split,DeepNSDEncodingEvaluator(4,0,True,'float64'),alphas=RELEASED_CODE_ALPHAS)
        assert out.provenance['roi_voxel_count']=='11'
