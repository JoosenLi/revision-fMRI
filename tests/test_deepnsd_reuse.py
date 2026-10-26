"""Verify actual source reuse, compatibility failures, and numerical equivalence."""
from importlib import resources
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import DataDimensionalityWarning
from revision_fmri import deepnsd
from revision_fmri.config import load_config
from revision_fmri.data import ResponseSet
from revision_fmri.features import FeatureSet
from revision_fmri.evaluation import (DeepNSDEncodingEvaluator, RELEASED_CODE_ALPHAS,
                                     EvaluationSplit, evaluate_layers, deepnsd_shared1000_split)

CASES = json.loads((Path(__file__).parent/'fixtures/encoding_reference.json').read_text())['cases']


def objects(case):
    raw = np.asarray(case['raw'], dtype=np.float32)
    y = np.asarray(case['responses'])
    ids = tuple(f'image-{i}' for i in range(len(raw)))
    f = FeatureSet(ids, 'fixture-model', 'layer-1', raw, 'raw_flattened', {'source': 'synthetic'})
    r = ResponseSet(ids, tuple(f'voxel-{i}' for i in range(y.shape[1])), y,
                    'subject', 'roi', 'space', (3,)*len(ids), {'source': 'synthetic'})
    s = EvaluationSplit(tuple(ids[i] for i in case['train']), tuple(ids[i] for i in case['test']),
                        'synthetic-v1', 'synthetic-digest')
    encoder = DeepNSDEncodingEvaluator(case['n_components'], 0, True, 'float64')
    return f,r,s,encoder


def test_source_integrity_and_license_resources():
    manifest = deepnsd.verify_sources()
    assert manifest['files']['ridge_gcv_mod.py']['source_sha256'] == manifest['files']['ridge_gcv_mod.py']['vendored_sha256']
    root = resources.files('revision_fmri._vendor.deepnsd')
    assert 'GNU GENERAL PUBLIC LICENSE' in (root/'LICENSE').read_text()
    assert 'scikit-learn developers' in (root/'SKLEARN_COPYING').read_text()
    assert (root/'NOTICE.md').is_file()
    assert hashlib.sha256((root/'SKLEARN_COPYING').read_bytes()).hexdigest() == manifest['sklearn_license_sha256']


@pytest.mark.parametrize('case', CASES, ids=lambda x:x['name'])
def test_original_backend_matches_fixture(case):
    f,r,s,encoder = objects(case)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DataDimensionalityWarning)
        out = encoder.fit_predict(f,r,s,alphas=RELEASED_CODE_ALPHAS)
    assert_array_equal(out.alpha_per_voxel, case['alpha'])
    assert_allclose(out.train_loo_centered, case['loo_centered'], rtol=0, atol=1e-12)
    assert_allclose(out.test_with_intercept, case['test_predictions'], rtol=0, atol=1e-12)
    assert out.provenance['backend'] == 'deepnsd-vendored-sklearn132-v1'
    assert out.provenance['deepnsd_commit'] == '36f6a34727dccb5eec8fa49364b210a049a388d7'


def test_calls_actual_upstream_fit_not_reference(monkeypatch):
    from revision_fmri._vendor.deepnsd.ridge_gcv_mod import RidgeCVMod
    from revision_fmri import encoding
    original = RidgeCVMod.fit
    calls = []
    def spy(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Source backend must not use the independent SVD oracle')
    monkeypatch.setattr(RidgeCVMod, 'fit', spy)
    monkeypatch.setattr(encoding, 'fit_ridge_reference', forbidden)
    f,r,s,encoder = objects(CASES[1])
    encoder.fit_predict(f,r,s,alphas=RELEASED_CODE_ALPHAS)
    assert calls == [True]


def test_environment_mismatch_fails_before_private_import(monkeypatch):
    actual = deepnsd.metadata.version
    monkeypatch.setattr(deepnsd.metadata, 'version', lambda name: '99.0' if name=='scikit-learn' else actual(name))
    with pytest.raises(RuntimeError, match='scikit-learn==1.3.2'):
        deepnsd.require_compatible_environment()


def test_original_profile_cannot_silently_change_settings():
    f,r,s,encoder = objects(CASES[1])
    with pytest.raises(ValueError, match='normalization'):
        DeepNSDEncodingEvaluator(5,0,False,'float64')
    with pytest.raises(ValueError, match='seed'):
        DeepNSDEncodingEvaluator(5,1,True,'float64')
    with pytest.raises(ValueError, match='RELEASED_CODE_ALPHAS'):
        encoder.fit_predict(f,r,s,alphas=(1.,10.))
    cfg = load_config(Path(__file__).parents[1]/'configs/paths.example.yaml')
    assert DeepNSDEncodingEvaluator.from_config(cfg,n_components=5) == encoder


def test_original_split_is_order_preserving_not_random():
    ids = tuple(f'image-{i}' for i in range(1000))
    split = deepnsd_shared1000_split(ids, source_digest='ordered-csv-hash')
    assert split.train_ids == ids[::2] and split.test_ids == ids[1::2]
    with pytest.raises(ValueError, match='1000'):
        deepnsd_shared1000_split(ids[:-1], source_digest='hash')


def test_source_backend_test_response_independence():
    f,r,s,encoder = objects(CASES[1])
    rng = np.random.default_rng(72)
    second = FeatureSet(f.image_ids,f.model_id,'layer-2',rng.normal(size=f.values.shape).astype('float32'),
                         'raw_flattened',{'source':'synthetic'})
    result = evaluate_layers((f,second),r,s,encoder,alphas=RELEASED_CODE_ALPHAS)
    y = r.values.copy()
    y[16:] = rng.normal(size=y[16:].shape)
    changed = ResponseSet(r.image_ids,r.voxel_ids,y,r.subject_id,r.roi_id,r.space,r.repeat_counts,r.provenance)
    result2 = evaluate_layers((f,second),changed,s,encoder,alphas=RELEASED_CODE_ALPHAS)
    assert result.selected_layers == result2.selected_layers
    assert result.training_scores == result2.training_scores
    for layer in result.predictions:
        assert_array_equal(result.predictions[layer].alpha_per_voxel,result2.predictions[layer].alpha_per_voxel)
    assert result.test_scores != result2.test_scores
