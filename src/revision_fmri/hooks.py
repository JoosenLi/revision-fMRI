"""Single-batch raw hook capture; candidate selection and full extraction deferred."""
from dataclasses import dataclass
from importlib import metadata
from typing import Mapping
import numpy as np

from .config import LocalConfig
from .data import unique_ids
from .deepnsd import verify_sources
from .provenance import array_digest, metadata_digest
from .vision import LoadedVisionModel, LOADER_REVISION, _environment, _gpu_preflight


@dataclass(frozen=True)
class CapturedHookBatch:
    """Unfiltered activations in execution order, NOT an approved candidate manifest."""
    image_ids: tuple[str, ...]
    activations: Mapping[str, np.ndarray]
    provenance: dict


def _validate_runtime(loaded, inputs, config):
    import torch
    if config.compute.precision != 'float32' or config.compute.device != 'cuda':
        raise ValueError('Hook capture requires configured CUDA float32')
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4 or inputs.shape[1:] != (3, 224, 224):
        raise ValueError('Require preprocessed N x 3 x 224 x 224 tensor')
    if inputs.dtype != torch.float32 or inputs.device.type != 'cuda':
        raise ValueError('Inputs must already be CUDA float32; no CPU fallback')
    if inputs.device.index not in config.compute.allowed_gpu_ids:
        raise ValueError('Input GPU is not allowed by configuration')
    if loaded.provenance.get('device') != str(inputs.device):
        raise ValueError('Input device must match the loaded model provenance')
    tensors = list(loaded.model.parameters()) + list(loaded.model.buffers())
    if not tensors or any(t.device != inputs.device for t in tensors):
        raise ValueError('All model tensors must already be on the input device')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors):
        raise ValueError('Model floating tensors must be float32')
    if any(p.requires_grad for p in loaded.model.parameters()):
        raise ValueError('Model must be frozen before hook capture')
    if any(m.training for m in loaded.model.modules()):
        raise ValueError('Model must be in eval mode before hook capture')
    if torch.get_default_dtype() != torch.float32:
        raise ValueError('Default dtype must be float32')
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError('Disable TF32 explicitly before hook capture')
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise ValueError('Autocast is not supported')
    return _gpu_preflight(torch, inputs.device.index, required_bytes=1)


def capture_feature_batch(loaded: LoadedVisionModel, inputs, *, image_ids: tuple[str, ...],
                          config: LocalConfig) -> CapturedHookBatch:
    """Call the original C2 hook routine without per-batch dedup or layer filtering.

    Inputs are already preprocessed/on the configured device. This function never
    moves a model/input, chooses a batch size or writes a cache. Requires exclusive
    use of the model: preexisting forward hooks are rejected. ReLU replacement is
    an intentional persistent upstream mutation. No activation-memory guarantee.
    """
    unique_ids(image_ids, 'Image IDs')
    if not image_ids or getattr(inputs, 'ndim', 0) < 1 or inputs.shape[0] != len(image_ids):
        raise ValueError('Explicit image IDs must match the nonempty batch')
    if loaded.provenance.get('loader_revision') != LOADER_REVISION:
        raise ValueError('Only the reviewed offline AlexNet loader profile is supported')
    versions = _environment()
    if loaded.provenance.get('software_versions') != versions:
        raise ValueError('Runtime versions differ from loaded-model provenance')
    source = verify_sources()
    model = loaded.model
    if any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise ValueError('Exclusive model with no existing forward hooks required')
    gpu = _validate_runtime(loaded, inputs, config)
    import torch
    from ._vendor.deepnsd.hooks import get_feature_maps
    try:
        # Upstream uses bare .cuda(); bind its current device explicitly so a
        # configured non-default GPU cannot silently redirect inputs elsewhere.
        with torch.cuda.device(inputs.device):
            maps = get_feature_maps(model, inputs, layers_to_retain=None,
                                    remove_duplicates=False, enforce_input_shape=True)
    finally:
        # Upstream removes handles only after a successful forward. We reject
        # existing hooks above, so cleanup on failure removes only this call's hooks.
        for module in model.modules():
            module._forward_hooks.clear()
        # Newly replaced ReLU modules default to training=True; restore eval.
        model.eval()
    if not maps:
        raise ValueError('No tensor activations survived original shape handling')
    arrays = {name: value.numpy().reshape(len(image_ids), -1).copy()
              for name, value in maps.items()}
    if any(a.shape[1] == 0 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Empty/nonfinite hook activations are not supported')
    order = tuple(arrays)
    return CapturedHookBatch(image_ids, arrays, {
        'stage': 'unfiltered_single_batch_v1',
        'model': dict(loaded.provenance),
        'deepnsd_commit': source['commit'],
        'hook_source_sha256': source['files']['hooks.py']['vendored_sha256'],
        'software_versions': {**versions, 'numpy': np.__version__,
                              'tqdm': metadata.version('tqdm')},
        'image_order_sha256': metadata_digest(image_ids),
        'input_sha256': array_digest(inputs.detach().cpu().numpy()),
        'observed_layer_order': order,
        'observed_layer_order_sha256': metadata_digest(order),
        'activation_sha256': {name: array_digest(a) for name, a in arrays.items()},
        'precision': 'float32', 'deduplication': 'not applied at batch stage',
        'candidate_selection': 'not applied',
        'seed': None, 'randomness': 'no random probe or hash dedup invoked',
        'gpu_preflight': gpu,
    })
