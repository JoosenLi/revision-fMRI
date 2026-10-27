"""Local stimulus bytes -> explicit RGB -> existing model-specific transform."""
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any, Iterator

from .config import LocalConfig, load_config
from .data import unique_ids
from .provenance import array_digest, metadata_digest
from .vision import ClassificationTransform


@dataclass(frozen=True)
class StimulusBatch:
    image_ids: tuple[str, ...]
    inputs: Any  # CPU float32 tensor; explicit device transfer belongs to the caller.
    provenance: dict


def _contained_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Stimulus filenames must be nonempty relative paths')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Stimulus file escapes configured stimuli_root or is not a file')
    return path


class StimulusReader:
    """Explicit manifest under configured stimuli_root, no discovery or downloads.

    This initial profile accepts the reviewed ClassificationTransform only. It
    prepares CPU image tensors; it never constructs/runs/moves a vision model.
    """
    def __init__(self, config: LocalConfig, *, manifest: str):
        self.config = config
        self._root = config.storage['stimuli_root']
        raw = _contained_file(self._root, manifest).read_bytes()
        document = json.loads(raw)
        if (not isinstance(document, dict)
                or set(document) != {'schema_version', 'dataset_release', 'images'}
                or document['schema_version'] != 1
                or not isinstance(document['dataset_release'], str) or not document['dataset_release']):
            raise ValueError('Explicit stimulus manifest schema and dataset release required')
        records = document['images']
        if not isinstance(records, list) or not records:
            raise ValueError('A nonempty stimulus manifest is required')
        for row in records:
            if (not isinstance(row, dict) or set(row) != {'image_id', 'file', 'sha256'}
                    or not isinstance(row['file'], str) or not row['file']
                    or Path(row['file']).is_absolute()
                    or not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
                raise ValueError('Each stimulus requires image_id, relative file and full SHA256')
        unique_ids(tuple(row['image_id'] for row in records), 'Manifest image IDs')
        self._records = {row['image_id']: dict(row) for row in records}
        self._release = document['dataset_release']
        self._manifest_sha256 = hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_config(cls, path=None, *, manifest: str):
        return cls(load_config(path), manifest=manifest)

    def batches(self, *, image_ids: tuple[str, ...], batch_size: int,
                preprocess: ClassificationTransform) -> Iterator[StimulusBatch]:
        """Yield requested order without shuffle, sorting, dropping or augmentation.

        All requested IDs are validated before the first yield. Each file's exact
        bytes are checked immediately before decoding those same bytes. Later
        batches may fail even if earlier ones succeeded; there is no whole-run
        atomicity. The caller owns yielded tensors; retaining all batches consumes
        whole-dataset memory. Image decode size must be budgeted by the caller.
        """
        unique_ids(image_ids, 'Requested stimulus IDs')
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Explicit positive integer batch_size required')
        if self.config.compute.precision != 'float32':
            raise ValueError('Reviewed stimulus profile requires float32')
        if not isinstance(preprocess, ClassificationTransform):
            raise ValueError('Use the reviewed model-specific ClassificationTransform')
        missing = set(image_ids) - self._records.keys()
        if missing:
            raise ValueError('Requested image IDs are absent from the stimulus manifest')
        import torch
        from PIL import Image
        for start in range(0, len(image_ids), batch_size):
            ids = image_ids[start:start+batch_size]
            tensors, sources = [], []
            for image_id in ids:
                row = self._records[image_id]
                raw = _contained_file(self._root, row['file']).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest != row['sha256']:
                    raise ValueError(f'Stimulus SHA256 mismatch for image ID {image_id!r}')
                # Decode the verified bytes, not a second potentially changed file.
                with Image.open(BytesIO(raw)) as image:
                    source = {'image_id': image_id, 'sha256': digest,
                              'mode': image.mode, 'size': list(image.size)}
                    # C2 ImageDataset.__getitem__: every image, including RGB.
                    with image.convert('RGB') as rgb:
                        tensor = preprocess(rgb)
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                        or tensor.dtype != torch.float32 or tuple(tensor.shape) != (3, 224, 224)
                        or not torch.isfinite(tensor).all()):
                    raise ValueError('Transform must return finite CPU float32 3x224x224 input')
                tensors.append(tensor)
                sources.append(source)
            inputs = torch.stack(tensors)
            yield StimulusBatch(ids, inputs, {
                'stage': 'stimulus_rgb_classification_v1', 'dataset_release': self._release,
                'manifest_sha256': self._manifest_sha256,
                'image_order_sha256': metadata_digest(ids),
                'request_order_sha256': metadata_digest(image_ids),
                'source_images': sources, 'source_images_sha256': metadata_digest(sources),
                'decode_policy': "PIL.Image.open(verified_bytes).convert('RGB'); no EXIF transpose",
                'transform_sha256': preprocess.sha256, 'transform': dict(preprocess.identity),
                'software_versions': dict(preprocess.versions), 'precision': 'float32',
                'input_sha256': array_digest(inputs.numpy()), 'seed': None,
                'randomness': 'no shuffle or augmentation',
            })
