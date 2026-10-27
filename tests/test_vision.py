"""CPU transform/lifecycle tests and opt-in read-only CUDA preflight; no forward."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
Image = pytest.importorskip('PIL.Image')

from revision_fmri.config import ComputeConfig, LocalConfig, load_config
from revision_fmri.models import ModelSpec
from revision_fmri.vision import (
    ClassificationTransform, LOADER_REVISION, TRANSFORM_ID,
    TorchvisionAlexNetLoader, _gpu_preflight,
)
import revision_fmri.vision as vision
from revision_fmri._vendor.deepnsd import vision as source


def test_rectangular_rgb_transform_matches_original_and_explicit_pixels():
    pixels = np.arange(19 * 31 * 3, dtype=np.uint8).reshape(19, 31, 3)
    image = Image.fromarray(pixels)
    adapter = ClassificationTransform()
    actual = adapter(image)
    original = source.get_torchvision_transforms('classification')(image)
    assert torch.equal(actual, original)
    resized = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
    expected = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255
    expected = (expected - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32 and actual.shape == (3, 224, 224)
    assert adapter.sha256 == ClassificationTransform().sha256
    assert len(adapter.sha256) == 64


@pytest.mark.parametrize('mode', ['L', 'RGBA'])
def test_transform_rejects_implicit_mode_conversion(mode):
    with pytest.raises(ValueError, match='RGB PIL'):
        ClassificationTransform()(Image.new(mode, (3, 4)))


def test_transform_rejects_default_float64():
    adapter = ClassificationTransform()
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(ValueError, match='float32'):
            adapter(Image.new('RGB', (3, 4)))
    finally:
        torch.set_default_dtype(before)


@pytest.fixture
def setup_loader(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'tiny.pth'
    torch.save({'weight': torch.ones((2, 3)), 'bias': torch.zeros(2)}, checkpoint)
    config = LocalConfig({}, ComputeConfig('cuda', (1,), 'float32', 'float64'),
                         {'alexnet_classification': checkpoint})
    spec = ModelSpec('alexnet_classification', hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     LOADER_REVISION, TRANSFORM_ID, 'a' * 64)
    calls = []

    class Tiny(torch.nn.Linear):
        def to(self, *args, **kwargs):
            calls.append(('to', kwargs))
            return self  # No CUDA allocation and no forward in these tests.

    def constructor(name, pretrained):
        calls.append(('constructor', name, pretrained))
        return Tiny(3, 2)

    # Tiny lifecycle seam only; separate tests exercise the real count contract.
    monkeypatch.setattr(vision, 'ALEXNET_PARAMETER_COUNT', 8)
    monkeypatch.setattr(source, 'get_torchvision_model', constructor)
    monkeypatch.setattr(vision, '_gpu_preflight', lambda *a: {'mock': True})
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    return TorchvisionAlexNetLoader(config), spec, checkpoint, calls


def test_offline_load_freezes_evals_records_provenance_and_restores_rng(setup_loader, monkeypatch):
    loader, spec, checkpoint, calls = setup_loader
    load = torch.load
    kwargs = []
    def spy(*args, **kw):
        kwargs.append(kw)
        return load(*args, **kw)
    monkeypatch.setattr(torch, 'load', spy)
    rng = torch.get_rng_state().clone()
    result = loader.load_configured(spec, gpu_id=1)
    assert torch.equal(torch.get_rng_state(), rng)
    assert kwargs == [{'map_location': 'cpu', 'weights_only': True}]
    assert calls == [('constructor', 'models.alexnet', False),
                     ('to', {'device': 'cuda:1', 'dtype': torch.float32})]
    assert not result.model.training
    assert all(not p.requires_grad for p in result.model.parameters())
    assert torch.equal(result.model.weight, torch.ones((2, 3)))
    assert result.provenance['parameter_count'] == 8  # Tiny lifecycle seam.
    assert result.provenance['checkpoint_sha256'] == spec.checkpoint_sha256
    assert result.provenance['transform_sha256'] == result.preprocess.sha256
    assert str(checkpoint) not in json.dumps(result.provenance)


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'cuda:0', 'cuda:-1'])
def test_device_policy_rejected_before_load(setup_loader, device):
    loader, spec, checkpoint, calls = setup_loader
    with pytest.raises(ValueError, match='allowed CUDA'):
        loader.load(spec, checkpoint, device=device)
    assert calls == []


def test_corrupted_checkpoint_never_deserialized(setup_loader, monkeypatch):
    loader, spec, checkpoint, calls = setup_loader
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('deserialized corrupt checkpoint'))
    checkpoint.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        loader.load_configured(spec, gpu_id=1)
    assert calls == []


@pytest.mark.parametrize('changes', [dict(model_id='alexnet_random'), dict(loader_revision='other'),
                                    dict(transform_id='DEFAULT'), dict(candidate_manifest_sha256='')])
def test_identity_rejected(setup_loader, changes):
    loader, spec, checkpoint, calls = setup_loader
    with pytest.raises(ValueError):
        loader.load(replace(spec, **changes), checkpoint, device='cuda:1')
    assert calls == []


@pytest.mark.parametrize('state,match', [
    ({'state_dict': {}}, 'plain tensor'),
    ({'wrong': torch.ones(2)}, 'keys'),
    ({'weight': torch.ones((2, 3), dtype=torch.float64), 'bias': torch.zeros(2)}, 'dtype'),
    ({'weight': torch.ones((2, 4)), 'bias': torch.zeros(2)}, 'size mismatch'),
])
def test_strict_state_dict(setup_loader, state, match):
    loader, spec, checkpoint, calls = setup_loader
    torch.save(state, checkpoint)
    spec = replace(spec, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    with pytest.raises((ValueError, RuntimeError), match=match):
        loader.load_configured(spec, gpu_id=1)
    assert not any(c[0] == 'to' for c in calls)


def test_precision_and_tf32_fail_closed(setup_loader, monkeypatch):
    loader, spec, checkpoint, calls = setup_loader
    bad = replace(loader.config, compute=replace(loader.config.compute, precision='float64'))
    with pytest.raises(ValueError, match='float32 only'):
        TorchvisionAlexNetLoader(bad).load_configured(spec, gpu_id=1)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)
    with pytest.raises(ValueError, match='TF32'):
        loader.load_configured(spec, gpu_id=1)
    assert calls == []


def test_binding_and_missing_file(setup_loader):
    loader, spec, checkpoint, calls = setup_loader
    with pytest.raises(ValueError, match='binding'):
        loader.load(spec, checkpoint.with_name('other'), device='cuda:1')
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):
        loader.load_configured(spec, gpu_id=1)
    assert calls == []


def test_version_guard(monkeypatch):
    monkeypatch.setattr(vision.metadata, 'version', lambda name: '0.0')
    with pytest.raises(RuntimeError, match='compatibility requires'):
        ClassificationTransform()


@pytest.mark.parametrize('available,free,report,error', [
    (False, 1000, '', 'unavailable'),
    (True, 9, '', 'Insufficient'),
    (True, 1000, 'GPU-test, 999999999', 'other compute'),
    (True, 1000, 'bad report', 'Unrecognized'),
    (True, 1000, '', None),
    (True, 1000, f'GPU-test, {os.getpid()}', None),
    (True, 1000, 'GPU-other, 999999999', None),
])
def test_gpu_preflight_without_cuda(monkeypatch, available, free, report, error):
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(uuid='test'),
        mem_get_info=lambda index: (free, 2000)))
    monkeypatch.setattr(vision.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=report))
    if error:
        with pytest.raises(RuntimeError, match=error):
            _gpu_preflight(fake, 1, 10)
    else:
        assert _gpu_preflight(fake, 1, 10)['model_bytes'] == 10


def test_original_constructor_passes_pretrained_false(monkeypatch):
    import torchvision.models
    calls = []
    sentinel = object()
    def constructor(*, pretrained):
        calls.append(pretrained)
        return sentinel
    monkeypatch.setattr(torchvision.models, 'alexnet', constructor)
    assert source.get_torchvision_model('models.alexnet', pretrained=False) is sentinel
    assert calls == [False]


def test_autocast_rejected(setup_loader):
    loader, spec, checkpoint, calls = setup_loader
    with torch.autocast('cpu'):
        with pytest.raises(ValueError, match='Autocast'):
            loader.load_configured(spec, gpu_id=1)
    assert calls == []


def test_failed_cuda_preflight_does_not_transfer(setup_loader, monkeypatch):
    loader, spec, checkpoint, calls = setup_loader
    def fail(*args):
        raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(vision, '_gpu_preflight', fail)
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        loader.load_configured(spec, gpu_id=1)
    assert not any(c[0] == 'to' for c in calls)


@pytest.mark.parametrize('uuid', ['test', 'GPU-test', '', 'MIG-test'])
def test_gpu_uuid_normalization(monkeypatch, uuid):
    fake = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2,
        get_device_properties=lambda index: SimpleNamespace(uuid=uuid),
        mem_get_info=lambda index: (1000, 2000)))
    monkeypatch.setattr(vision.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    if uuid in ('', 'MIG-test'):
        with pytest.raises(RuntimeError, match='Cannot map CUDA device'):
            _gpu_preflight(fake, 1, 10)
    else:
        assert _gpu_preflight(fake, 1, 10)['gpu_uuid'] == 'GPU-test'


@pytest.mark.skipif(os.environ.get('REVISION_FMRI_TEST_CUDA') != '1',
                    reason='Read-only CUDA check requires explicit opt-in')
def test_real_gpu_preflight_read_only():
    # Check inside the test so ordinary collection never probes/initializes CUDA.
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    config = load_config()
    if not config.compute.allowed_gpu_ids:
        pytest.skip('No configured GPU allowance')
    for index in config.compute.allowed_gpu_ids:
        snapshot = _gpu_preflight(torch, index, required_bytes=1)
        bare = str(torch.cuda.get_device_properties(index).uuid).removeprefix('GPU-')
        assert snapshot['gpu_uuid'] == f'GPU-{bare}'
        assert 1 <= snapshot['free_bytes'] <= snapshot['total_bytes']
        assert snapshot['model_bytes'] == 1  # Query only; no tensor/model allocation.


def test_published_parameter_count_guard_precedes_transfer(setup_loader, monkeypatch):
    loader,spec,checkpoint,calls=setup_loader
    monkeypatch.setattr(vision,'ALEXNET_PARAMETER_COUNT',61_100_840)
    with pytest.raises(ValueError,match='parameter count'):
        loader.load_configured(spec,gpu_id=1)
    assert calls == [('constructor','models.alexnet',False)]


def test_real_alexnet_constructor_count_matches_published_metadata():
    # Meta tensors inspect the real constructor without storage, CUDA or forward.
    from torchvision.models import alexnet
    with torch.device('meta'):
        model=alexnet(weights=None)
    published=json.loads((Path(__file__).resolve().parents[1]/'manifests/models.json').read_text())
    row=next(r for r in published['models'] if r['model_id']=='alexnet_classification')
    count=sum(p.numel() for p in model.parameters())
    assert count == vision.ALEXNET_PARAMETER_COUNT == 61_100_840
    assert count == int(row['raw_result_metadata']['total_parameter_count'])
