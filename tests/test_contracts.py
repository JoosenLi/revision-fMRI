import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import pearsonr

from revision_fmri.config import load_config
from revision_fmri.data import ResponseSet
from revision_fmri.models import verify_checkpoint
from revision_fmri.features import FeatureSet
from revision_fmri.metrics import crsa, compare_rdms, correlation_rdm
from revision_fmri.evaluation import EvaluationSplit, evaluate_crsa, select_layer, RELEASED_CODE_ALPHAS

ROOT = Path(__file__).resolve().parents[1]


def settings(tmp_path):
    payload = yaml.safe_load((ROOT / 'configs/paths.example.yaml').read_text())
    payload['checkpoints'] = {'test-model': 'weights.bin'}
    path = tmp_path / 'configs' / 'paths.local.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(payload))
    return path


def test_config_is_relative_to_file_not_cwd(tmp_path, monkeypatch):
    path = settings(tmp_path)
    monkeypatch.chdir('/')
    config = load_config(path)
    assert config.storage['data_root'] == tmp_path / 'data'
    assert config.checkpoints['test-model'] == tmp_path / 'checkpoints/weights.bin'
    assert not config.storage['data_root'].exists()
    assert config.compute.precision == 'float32'
    assert config.compute.ridge_precision == 'float64'
    assert config.compute.allowed_gpu_ids == ()


def test_config_environment_and_missing(tmp_path, monkeypatch):
    path = settings(tmp_path)
    monkeypatch.setenv('REVISION_FMRI_CONFIG', str(path))
    assert load_config() == load_config(path)
    monkeypatch.delenv('REVISION_FMRI_CONFIG')
    with pytest.raises(ValueError):
        load_config()


def test_checkpoint_local_only(tmp_path):
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'local fixture, not a model')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_checkpoint(path, digest) == path
    with pytest.raises(ValueError, match='mismatch'):
        verify_checkpoint(path, '0' * 64)
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(tmp_path / 'missing', digest)
    with pytest.raises(ValueError, match='SHA-256'):
        verify_checkpoint(path, 'unresolved')


@pytest.mark.parametrize('key,value', [('device', 'cpu'), ('precision', 'float16'),
                                      ('ridge_precision', 'bfloat16'), ('allowed_gpu_ids', [0, 0]),
                                      ('allowed_gpu_ids', [-1]), ('allowed_gpu_ids', [True])])
def test_invalid_compute_policy(tmp_path, key, value):
    path = settings(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload['compute'][key] = value
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError):
        load_config(path)


def test_explicit_float32_ridge_policy(tmp_path):
    path = settings(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload['compute']['ridge_precision'] = 'float32'
    path.write_text(yaml.safe_dump(payload))
    assert load_config(path).compute.ridge_precision == 'float32'


def test_crsa_independent_reference():
    rng = np.random.default_rng(71)
    f, b = rng.normal(size=(7, 11)), rng.normal(size=(7, 9))
    # Explicit pair-by-pair scipy Pearson, independent of matrix implementation.
    left = [pearsonr(f[i], f[j]).statistic for i in range(7) for j in range(i+1, 7)]
    right = [pearsonr(b[i], b[j]).statistic for i in range(7) for j in range(i+1, 7)]
    assert crsa(f, b) == pytest.approx(pearsonr(left, right).statistic, abs=1e-14)
    assert crsa(f, f) == pytest.approx(1.0)


def test_upper_triangle_only():
    a = correlation_rdm(np.random.default_rng(5).normal(size=(5, 8)))
    b = a.copy()
    b[np.tril_indices(5)] = np.nan
    assert compare_rdms(a, b) == pytest.approx(1.0)


@pytest.mark.parametrize('values', [np.ones((4, 5)), np.array([[1., np.nan], [2., 3.]]), np.ones(4)])
def test_degenerate_patterns_fail(values):
    with pytest.raises(ValueError):
        correlation_rdm(values)


def response(ids=('a', 'b', 'c'), values=None, counts=(3, 3, 3)):
    return ResponseSet(ids, ('v1', 'v2', 'v3', 'v4'),
                       np.arange(12.).reshape(3, 4) if values is None else values,
                       'sub-01', 'explicit-roi', 'T1w', counts, {'policy': 'fixture'})


def test_response_axes_and_ids():
    with pytest.raises(ValueError, match='axes'):
        response(values=np.ones((4, 3)))
    with pytest.raises(ValueError, match='unique'):
        response(ids=('a', 'a', 'c'))
    with pytest.raises(ValueError, match='repeat'):
        response(counts=(3, 0, 3))


def test_crsa_alignment_and_representation():
    rng = np.random.default_rng(19)
    r = response(values=rng.normal(size=(3, 4)))
    f = FeatureSet(r.image_ids, 'm', 'l', rng.normal(size=(3, 6)), 'raw_flattened', {'source': 'fixture'})
    assert np.isfinite(evaluate_crsa(f, r))
    for ids, representation in [(('b', 'a', 'c'), 'raw_flattened'), (r.image_ids, 'srp')]:
        wrong = FeatureSet(ids, 'm', 'l', f.values, representation, {'source': 'fixture'})
        with pytest.raises(ValueError):
            evaluate_crsa(wrong, r)


def test_partition_and_coverage():
    split = EvaluationSplit(('a', 'b'), ('c', 'd'), 'fixture-v1', 'fixture-digest')
    split.validate_coverage(('d', 'b', 'c', 'a'))
    with pytest.raises(ValueError, match='partition'):
        split.validate_coverage(('a', 'b', 'c', 'e'))
    with pytest.raises(ValueError, match='both'):
        EvaluationSplit(('a', 'b'), ('b', 'c'), 'fixture-v1', 'fixture-digest')


def test_train_only_selection_and_tie_order():
    assert select_layer({'late': .7, 'earlier': .7, 'last': .2}) == 'late'
    with pytest.raises(ValueError):
        select_layer({'layer': np.nan})
    with pytest.raises(TypeError):
        select_layer({'layer': .4}, test_scores={'layer': .99})
    np.testing.assert_array_equal(RELEASED_CODE_ALPHAS, np.logspace(-1, 5, 7))


def test_publication_inventory():
    models = json.loads((ROOT / 'manifests/models.json').read_text())['models']
    assert len(models) == len({m['model_id'] for m in models}) == 224
    assert sum(m['comparisons'].get('compare_architecture') == 'Convolutional' for m in models) == 34
    assert sum(m['comparisons'].get('compare_architecture') == 'Transformer' for m in models) == 21
    assert sum('compare_goal_slip' in m['comparisons'] for m in models) == 9
    assert sum('compare_diet_imagenetsize' in m['comparisons'] for m in models) == 24
    assert sum('compare_diet_ipcl' in m['comparisons'] for m in models) == 4
    assert all(m['extraction_approved'] is False and m['local_sha256'] is None for m in models)
    for m in models:
        if 'compare_goal_slip' in m['comparisons']:
            assert m['checkpoint_locator'].endswith('_25ep.pt')
        if m['family'] == 'open_ipcl' and m['diet'] == 'imagenet':
            assert 'pct40' in m['checkpoint_locator']


def test_base_import_does_not_load_optional_vision():
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    subprocess.run([sys.executable, '-c',
                    'import sys; import revision_fmri; '
                    'assert not ({"torch", "torchvision", "PIL"} & sys.modules.keys())'],
                   env=env, check=True)
