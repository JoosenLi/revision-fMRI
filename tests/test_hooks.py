"""Synthetic CPU hook semantics; runtime boundary is mocked only in these tests."""
from types import SimpleNamespace
from contextlib import nullcontext, contextmanager
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from revision_fmri import hooks
from revision_fmri.config import ComputeConfig, LocalConfig
from revision_fmri.vision import LoadedVisionModel, LOADER_REVISION, _environment
from revision_fmri._vendor.deepnsd import hooks as upstream


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False),
                                            torch.nn.ReLU(inplace=True))
        self.pipeline[0].weight.data.copy_(torch.eye(2))
        self.shared = torch.nn.Identity()

    def forward(self, x):
        y = self.pipeline(x)
        return self.shared(self.shared(y))


@pytest.fixture
def setup(monkeypatch):
    model = Toy().eval().requires_grad_(False)
    loaded = LoadedVisionModel(model, None, {'loader_revision': LOADER_REVISION,
                               'software_versions': _environment()})
    # Explicit synthetic test seam; no public CPU inference option is provided.
    monkeypatch.setattr(hooks, '_validate_runtime', lambda *a: {'synthetic': True})
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    monkeypatch.setattr(upstream, 'prep_model_for_extraction', lambda model, inputs: (model, inputs))
    inputs = torch.tensor([[-1., 2.], [3., -4.], [-5., 6.]])
    return loaded, inputs


def capture(loaded, inputs):
    return hooks.capture_feature_batch(loaded, inputs, image_ids=('a', 'b', 'c'), config=None)


def test_execution_order_reused_modules_parent_and_pre_relu(setup):
    loaded, inputs = setup
    out = capture(loaded, inputs)
    assert tuple(out.activations) == ('Linear-1', 'ReLU-1', 'Identity-1', 'Identity-2', 'Toy-1')
    np.testing.assert_array_equal(out.activations['Linear-1'], inputs.numpy())
    np.testing.assert_array_equal(out.activations['ReLU-1'], inputs.clamp_min(0).numpy())
    assert not loaded.model.pipeline[1].inplace
    assert all(not m.training and not m._forward_hooks for m in loaded.model.modules())
    assert all(a.dtype == np.float32 for a in out.activations.values())
    again = capture(loaded, inputs)
    assert out.provenance == again.provenance
    assert out.provenance['candidate_selection'] == 'not applied'
    assert out.provenance['activation_sha256']['Identity-1'] == out.provenance['activation_sha256']['Identity-2']


def test_exception_cleanup_and_existing_hook_preservation(setup):
    loaded, inputs = setup
    handle = loaded.model.shared.register_forward_hook(lambda *a: None)
    with pytest.raises(ValueError, match='no existing'):
        capture(loaded, inputs)
    assert loaded.model.shared._forward_hooks
    handle.remove()
    def fail(x):
        loaded.model.pipeline(x)
        raise RuntimeError('synthetic failure')
    loaded.model.forward = fail
    with pytest.raises(RuntimeError, match='synthetic failure'):
        capture(loaded, inputs)
    assert all(not m._forward_hooks for m in loaded.model.modules())


def test_tuple_axis_scalar_and_mapping_semantics(setup):
    loaded, inputs = setup
    class Outputs(torch.nn.Module):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            return (x.T, torch.tensor(1.), torch.zeros(2, 3, 3), {'nested': x})
    loaded.model.outputs = Outputs().eval()
    loaded.model.forward = lambda x: {'ignored_root': loaded.model.outputs(x)}
    out = capture(loaded, inputs)
    assert tuple(out.activations) == ('Outputs-1-1',)
    np.testing.assert_array_equal(out.activations['Outputs-1-1'], inputs.numpy())


def test_original_substring_occurrence_count():
    # Preserve source substring counting, not class-only counts.
    assert upstream.get_module_name(torch.nn.ReLU(), ['OtherReLU-1']) == 'ReLU-2'


@pytest.mark.parametrize('ids', [('a', 'a', 'c'), ('a',), ()])
def test_ids_rejected(setup, ids):
    loaded, inputs = setup
    with pytest.raises(ValueError):
        hooks.capture_feature_batch(loaded, inputs, image_ids=ids, config=None)


def test_public_runtime_rejects_cpu():
    config = LocalConfig({}, ComputeConfig('cuda', (1,), 'float32', 'float64'), {})
    with pytest.raises(ValueError, match='no CPU fallback'):
        hooks._validate_runtime(SimpleNamespace(), torch.zeros(3, 3, 224, 224), config)


def test_nonfinite_rejected_and_cleanup(setup):
    loaded, inputs = setup
    inputs[0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        capture(loaded, inputs)
    assert all(not m._forward_hooks for m in loaded.model.modules())


def test_selected_device_context_is_restored(setup, monkeypatch):
    loaded, inputs = setup
    events = []
    @contextmanager
    def context(device):
        events.append(('enter', device))
        try:
            yield
        finally:
            events.append(('exit', device))
    monkeypatch.setattr(torch.cuda, 'device', context)
    capture(loaded, inputs)
    assert events == [('enter', inputs.device), ('exit', inputs.device)]
