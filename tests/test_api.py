"""Collaboration facade: portable summaries and safe configured output."""
from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from revision_fmri import (DeepNSDProtocol, EvaluationSplit, FeatureSet,
                           ReproductionPipeline, ResponseSet)
from revision_fmri.provenance import metadata_digest


@pytest.fixture
def inputs():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(24, 8)).astype('float32')
    y = x @ rng.normal(size=(8, 6)) + rng.normal(size=(24, 6)) * .1 + 2
    ids = tuple(f'image-{i}' for i in range(24))
    features = [FeatureSet(ids, 'synthetic', 'layer-1', x, 'raw_flattened', {'source': 'synthetic'})]
    responses = ResponseSet(ids, tuple(f'voxel-{i}' for i in range(6)), y,
                            'subject', 'roi', 'space', (3,)*24, {'source': 'synthetic'})
    split = EvaluationSplit(ids[:16], ids[16:], 'synthetic-v1', 'synthetic')
    return dict(features=features, responses=responses, split=split)


def pipeline(root):
    root.mkdir(parents=True)
    cfg = {'schema_version': 1,
           'storage': {key: str(root/key) for key in ('data_root', 'stimuli_root',
                      'checkpoint_root', 'feature_root', 'results_root')},
           'compute': {'device': 'cuda', 'allowed_gpu_ids': [], 'precision': 'float32',
                       'ridge_precision': 'float64'}, 'checkpoints': {}}
    path = root/'paths.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return ReproductionPipeline.from_config(path, protocol=DeepNSDProtocol(
        'synthetic-api-v1', 5, 'Tiny synthetic fixture only; not a LAION policy'))


def test_report_is_portable_and_source_backed(tmp_path, inputs):
    a, b = pipeline(tmp_path/'a'), pipeline(tmp_path/'b')
    report = a.evaluate_model(**inputs)
    assert report.to_dict() == b.evaluate_model(**inputs).to_dict()
    assert report.content_sha256 == metadata_digest(report.to_dict())
    assert str(tmp_path) not in json.dumps(report.to_dict())
    assert report.to_dict()['api_version'] == '1'
    assert report.encoding_provenance['layer-1']['backend'] == 'deepnsd-vendored-sklearn132-v1'
    assert report.train_image_ids == inputs['split'].train_ids
    assert set(report.metrics) == {'crsa', 'versa'}
    assert len(report.alpha_per_layer['layer-1']) == 6
    path = a.save_report(report, run_id='synthetic-1')
    assert path.parent == a.config.storage['results_root']
    saved = json.loads(path.read_text())
    assert saved == {'content_sha256': report.content_sha256, 'report': report.to_dict()}
    with pytest.raises(FileExistsError):
        a.save_report(report, run_id='synthetic-1')
    with pytest.raises(ValueError, match='protocol'):
        a.save_report(replace(report, protocol={}), run_id='wrong-protocol')


@pytest.mark.parametrize('run_id', ['../escape', '/absolute', 'a/b', '', '.hidden', 'x'*129])
def test_reject_output_paths(tmp_path, inputs, run_id):
    p = pipeline(tmp_path/'config')
    report = p.evaluate_model(**inputs)
    with pytest.raises(ValueError, match='run_id'):
        p.save_report(report, run_id=run_id)
    assert not p.config.storage['results_root'].exists()


def test_api_preserves_alignment_and_precision_checks(tmp_path, inputs):
    p = pipeline(tmp_path/'config')
    wrong = replace(inputs['responses'], image_ids=inputs['responses'].image_ids[::-1])
    with pytest.raises(ValueError, match='image-ID aligned'):
        p.evaluate_model(**{**inputs, 'responses': wrong})
    bad = replace(p.config, compute=replace(p.config.compute, ridge_precision='float32'))
    with pytest.raises(ValueError, match='float64'):
        ReproductionPipeline(bad, p.protocol)
