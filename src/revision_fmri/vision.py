"""Offline AlexNet compatibility adapter; optional vision imports are lazy.

This is a tested compatibility profile, not a recovered historical environment.
CPU staging is allowed; execution requires an explicitly configured CUDA device.
"""
from dataclasses import dataclass
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .config import LocalConfig
from .deepnsd import verify_sources
from .models import ModelSpec, verify_checkpoint

LOADER_REVISION = 'deepnsd-alexnet-offline-v1'
TRANSFORM_ID = 'deepnsd-torchvision-classification-pil-v1'
VISION_VERSIONS = {'torch': '2.6.0', 'torchvision': '0.21.0', 'Pillow': '12.0.0'}


def _environment() -> dict[str, str]:
    versions = {}
    for name, expected in VISION_VERSIONS.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install the pinned 'vision' extra") from exc
        if actual.split('+')[0] != expected:
            raise RuntimeError(f'Vision compatibility requires {name}=={expected}; found {actual}')
        versions[name] = actual
    return versions


class ClassificationTransform:
    """Exact released transform on RGB PIL inputs; reject implicit mode changes."""

    def __init__(self):
        self.versions = _environment()
        source = verify_sources()
        from ._vendor.deepnsd.vision import get_torchvision_transforms
        self._transform = get_torchvision_transforms('classification', input_type='PIL')
        self.identity = {
            'transform_id': TRANSFORM_ID,
            'deepnsd_commit': source['commit'],
            'source_sha256': source['files']['vision.py']['vendored_sha256'],
            'software_versions': self.versions,
            'input_mode': 'RGB PIL',
            'precision': 'float32',
        }
        self.sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()

    def __call__(self, image):
        import torch
        from PIL import Image
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('Require an RGB PIL image; no implicit conversion')
        if torch.get_default_dtype() != torch.float32:
            raise ValueError('Original ToTensor compatibility requires default float32')
        return self._transform(image)


@dataclass(frozen=True)
class LoadedVisionModel:
    model: Any
    preprocess: ClassificationTransform
    provenance: dict


def _gpu_preflight(torch, index: int, required_bytes: int) -> dict:
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError('Configured CUDA device is unavailable; no CPU fallback')
    properties = torch.cuda.get_device_properties(index)
    uuid = str(getattr(properties, 'uuid', ''))
    if not uuid.startswith('GPU-'):
        raise RuntimeError('Cannot map CUDA device to GPU UUID for process inspection')
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=15)
    for row in result.stdout.splitlines():
        fields = [field.strip() for field in row.split(',')]
        if len(fields) != 2 or not fields[1].isdigit():
            raise RuntimeError('Unrecognized GPU process report')
        if fields[0] == uuid and int(fields[1]) != os.getpid():
            raise RuntimeError('Configured GPU has other compute processes')
    free, total = torch.cuda.mem_get_info(index)
    if free < required_bytes:
        raise RuntimeError('Insufficient free GPU memory for model parameters and buffers')
    # This is a snapshot, not a reservation or an inference-memory estimate.
    return {'gpu_uuid': uuid, 'free_bytes': free, 'total_bytes': total,
            'model_bytes': required_bytes}


class TorchvisionAlexNetLoader:
    """ModelLoader implementation with configured artifact/device binding.

    The caller must disable TF32 before loading. No inference is run here.
    Candidate-layer provenance is required but not enumerated/validated here.
    """

    def __init__(self, config: LocalConfig):
        self.config = config

    def load_configured(self, spec: ModelSpec, *, gpu_id: int) -> LoadedVisionModel:
        if spec.model_id not in self.config.checkpoints:
            raise ValueError('No configured local checkpoint for this model')
        return self.load(spec, self.config.checkpoints[spec.model_id], device=f'cuda:{gpu_id}')

    def load(self, spec: ModelSpec, checkpoint: Path, *, device: str) -> LoadedVisionModel:
        if spec.model_id != 'alexnet_classification':
            raise ValueError('Only alexnet_classification is implemented')
        if spec.loader_revision != LOADER_REVISION or spec.transform_id != TRANSFORM_ID:
            raise ValueError('Unrecognized loader/transform compatibility profile')
        if not re.fullmatch('[0-9a-f]{64}', spec.candidate_manifest_sha256):
            raise ValueError('Explicit candidate manifest SHA-256 required')
        compute = self.config.compute
        match = re.fullmatch(r'cuda:(0|[1-9][0-9]*)', device)
        if (compute.device != 'cuda' or match is None
                or int(match[1]) not in compute.allowed_gpu_ids):
            raise ValueError('Device must be an explicitly allowed CUDA index; no CPU fallback')
        if compute.precision != 'float32':
            raise ValueError('This vision profile implements float32 only')
        bound = self.config.checkpoints.get(spec.model_id)
        if bound is None or Path(checkpoint).resolve() != bound.resolve():
            raise ValueError('Checkpoint must match the local configuration binding')
        path = verify_checkpoint(checkpoint, spec.checkpoint_sha256)
        preprocess = ClassificationTransform()
        import torch
        from ._vendor.deepnsd.vision import get_torchvision_model
        if torch.get_default_dtype() != torch.float32:
            raise ValueError('Model staging requires default float32')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError('Disable matmul and cuDNN TF32 explicitly before loading')
        if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
            raise ValueError('Autocast is not supported by this float32 profile')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not state or any(
                not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
            raise ValueError('Require a plain tensor state_dict, without wrapper/key rewriting')
        # All initialized tensors are overwritten. Restore caller CPU RNG state;
        # never initialize/seed all CUDA devices as a constructor side effect.
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):
            torch.random.default_generator.manual_seed(0)
            model = get_torchvision_model('models.alexnet', pretrained=False)
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError('Checkpoint keys do not exactly match the architecture')
        if any(state[k].dtype != expected[k].dtype for k in expected):
            raise ValueError('Checkpoint dtype mismatch; implicit casting is forbidden')
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        model.eval()
        size = sum(v.numel() * v.element_size() for v in model.state_dict().values())
        gpu = _gpu_preflight(torch, int(match[1]), size)
        model.to(device=device, dtype=torch.float32)
        return LoadedVisionModel(model, preprocess, {
            'model_id': spec.model_id,
            'checkpoint_sha256': spec.checkpoint_sha256,
            'loader_revision': LOADER_REVISION,
            'transform_sha256': preprocess.sha256,
            'transform': preprocess.identity,
            'candidate_manifest_sha256': spec.candidate_manifest_sha256,
            'candidate_manifest_status': 'caller-supplied; hooks not validated',
            'software_versions': preprocess.versions,
            'precision': 'float32', 'device': device,
            'tf32': False, 'autocast': False,
            'constructor_seed': 0, 'constructor_state': 'fully overwritten by strict checkpoint load',
            'gpu_preflight': gpu,
        })
