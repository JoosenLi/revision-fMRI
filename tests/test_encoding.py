"""Numerical parity, leakage guards and edge cases on small synthetic arrays only."""
import json
from pathlib import Path
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.preprocessing import StandardScaler

from revision_fmri.encoding import fit_ridge_reference
from revision_fmri.features import FeatureSet, project_features, jl_dimensions
from revision_fmri.data import ResponseSet
from revision_fmri.evaluation import EvaluationSplit, ReferenceEncodingEvaluator, evaluate_layers
from revision_fmri.metrics import versa

FIXTURE = json.loads((Path(__file__).parent/'fixtures/encoding_reference.json').read_text())


def objects(case):
    raw = np.asarray(case['raw'], dtype='float32')
    ids = tuple(f'image-{i}' for i in range(len(raw)))
    y = np.asarray(case['responses'])
    f = FeatureSet(ids, 'fixture-model', 'fixture-layer', raw, 'raw_flattened', {'fixture': case['name']})
    r = ResponseSet(ids, tuple(f'voxel-{i}' for i in range(y.shape[1])), y,
                    'fixture-subject', 'fixture-roi', 'fixture-space', (3,)*len(ids), {'source': 'synthetic'})
    s = EvaluationSplit(tuple(ids[i] for i in case['train']), tuple(ids[i] for i in case['test']),
                        'synthetic-explicit-v1', 'fixture-digest')
    return f, r, s


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda x:x['name'])
def test_against_unmodified_released_ridge(case):
    fit = fit_ridge_reference(np.asarray(case['xtrain']), np.asarray(case['responses'])[case['train']],
                              alphas=tuple(case['alphas']), legacy_l2_normalize=True)
    assert_array_equal(fit.alpha_per_voxel, case['alpha'])
    for actual, expected in [(fit.coefficients, case['coefficients']), (fit.intercept, case['intercept']),
                             (fit.loo_centered, case['loo_centered']), (fit.alpha_scores, case['alpha_scores']),
                             (fit.predict(np.asarray(case['xtest'])), case['test_predictions'])]:
        assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda x:x['name'])
def test_full_projection_scaling_ridge_path(case):
    f, r, s = objects(case)
    encoder = ReferenceEncodingEvaluator(case['n_components'], case['seed'], True, 'float64')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DataDimensionalityWarning)  # Expected legacy upsampling.
        p = encoder.fit_predict(f, r, s, alphas=tuple(case['alphas']))
    assert_array_equal(p.alpha_per_voxel, case['alpha'])
    assert_allclose(p.train_loo_centered, case['loo_centered'], atol=1e-9, rtol=1e-8)
    assert_allclose(p.test_with_intercept, case['test_predictions'], atol=1e-9, rtol=1e-8)
    assert p.train_ids == s.train_ids and p.test_ids == s.test_ids
    assert p.provenance['ridge_precision'] == 'float64'
    if r.values.shape[1] > 1:
        assert versa(p.train_loo_centered, r.values[case['train']]) == pytest.approx(case['train_versa'], abs=1e-9)
        assert versa(p.test_with_intercept, r.values[case['test']]) == pytest.approx(case['test_versa'], abs=1e-9)


@pytest.mark.parametrize('normalize', [False, True])
def test_brute_force_leave_one_out(normalize):
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(9, 3)) + [2, -4, 7]
    alpha = 1.7
    actual = fit_ridge_reference(x, y, alphas=(alpha,), legacy_l2_normalize=normalize)
    z = x.copy()
    if normalize:
        z = (x-x.mean(0)) / np.linalg.norm(x-x.mean(0), axis=0)
    expected = np.empty_like(y)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        train, target = z[keep], y[keep]
        xm, ym = train.mean(0), target.mean(0)
        coefficients = np.linalg.solve((train-xm).T@(train-xm)+alpha*np.eye(x.shape[1]),
                                       (train-xm).T@(target-ym))
        expected[i] = (z[i]-xm)@coefficients + ym - y.mean(0)
    assert_allclose(actual.loo_centered, expected, atol=1e-12)
    assert not np.allclose(actual.predict(x)-y.mean(0), actual.loo_centered)


def test_individual_alphas_and_tie_order():
    all_alphas = [a for case in FIXTURE['cases'] for a in case['alpha']]
    assert len(set(all_alphas)) > 1
    y = np.arange(24.).reshape(8,3)
    fit = fit_ridge_reference(np.zeros((8,2)), y, alphas=(100.,.1,1.), legacy_l2_normalize=True)
    assert_array_equal(fit.alpha_per_voxel, [100.,100.,100.])


def test_projection_geometry_policy():
    assert jl_dimensions(1000, eps=.1) == 5920
    f, _, _ = objects(FIXTURE['cases'][1])
    p = project_features(f, n_components=5, seed=0)
    assert_allclose(p.values, FIXTURE['cases'][1]['projected'], rtol=0, atol=0)
    assert p.values.dtype == np.float32
    modified = FeatureSet(f.image_ids, f.model_id, 'other', f.values*2, f.representation, f.provenance)
    p2 = project_features(modified, n_components=5, seed=0)
    assert p.provenance['projection_sha256'] == p2.provenance['projection_sha256']
    assert_allclose(p2.values, 2*p.values)
    assert p.provenance['raw_features_sha256'] != p2.provenance['raw_features_sha256']
    with pytest.raises(ValueError):
        project_features(p, n_components=5, seed=0)


def test_outer_scaler_and_selection_do_not_use_test_responses():
    case = FIXTURE['cases'][1]
    f, r, s = objects(case)
    rng = np.random.default_rng(13)
    other = FeatureSet(f.image_ids, f.model_id, 'second', rng.normal(size=f.values.shape).astype('float32'),
                       'raw_flattened', {'fixture': 'second'})
    encoder = ReferenceEncodingEvaluator(5, 0, True, 'float64')
    original = evaluate_layers((f, other), r, s, encoder, alphas=tuple(case['alphas']))
    changed_y = r.values.copy()
    changed_y[case['test']] = rng.normal(size=changed_y[case['test']].shape)*20
    changed = ResponseSet(r.image_ids, r.voxel_ids, changed_y, r.subject_id, r.roi_id,
                          r.space, r.repeat_counts, r.provenance)
    result = evaluate_layers((f, other), changed, s, encoder, alphas=tuple(case['alphas']))
    assert original.selected_layers == result.selected_layers
    assert original.training_scores == result.training_scores
    assert original.test_scores != result.test_scores
    for layer in original.predictions:
        a,b = original.predictions[layer], result.predictions[layer]
        assert a.provenance == b.provenance
        assert_array_equal(a.alpha_per_voxel, b.alpha_per_voxel)
        assert_array_equal(a.test_with_intercept, b.test_with_intercept)
    changed_x = f.values.copy()
    changed_x[case['test']] *= 100
    test_changed_features = FeatureSet(f.image_ids, f.model_id, f.layer_id, changed_x,
                                       f.representation, f.provenance)
    pred = encoder.fit_predict(test_changed_features, r, s, alphas=tuple(case['alphas']))
    initial = original.predictions[f.layer_id]
    for key in ['scaler_sha256','coefficients_sha256','intercept_sha256']:
        assert initial.provenance[key] == pred.provenance[key]
    assert_array_equal(initial.train_loo_centered, pred.train_loo_centered)


def test_order_errors_and_precision_guard():
    f,r,s = objects(FIXTURE['cases'][1])
    encoder = ReferenceEncodingEvaluator(5, 0, True, 'float64')
    wrong = FeatureSet(tuple(reversed(f.image_ids)), f.model_id, f.layer_id, f.values,
                        f.representation, f.provenance)
    with pytest.raises(ValueError, match='order'):
        encoder.fit_predict(wrong, r, s, alphas=(.1,1.))
    with pytest.raises(ValueError, match='float64'):
        ReferenceEncodingEvaluator(5, 0, True, 'float32')
    with pytest.raises(ValueError, match='unique'):
        evaluate_layers((f,f), r, s, encoder, alphas=(.1,1.))


@pytest.mark.parametrize('alphas', [(), (0.,), (-1.,), (float('nan'),), (1.,1.)])
def test_bad_grids_fail(alphas):
    with pytest.raises(ValueError, match='Alphas'):
        fit_ridge_reference(np.ones((4,2)), np.arange(8.).reshape(4,2),
                            alphas=alphas, legacy_l2_normalize=True)


def test_constant_responses_fail():
    with pytest.raises(ValueError, match='Constant'):
        fit_ridge_reference(np.arange(12.).reshape(4,3), np.ones((4,2)),
                            alphas=(.1,1.), legacy_l2_normalize=True)
