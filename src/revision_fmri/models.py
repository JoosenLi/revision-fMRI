"""Offline model identity and local artifact verification; optional backend in revision_fmri.vision."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint_sha256: str
    loader_revision: str
    transform_id: str
    candidate_manifest_sha256: str


def verify_checkpoint(path: Path, expected_sha256: str) -> Path:
    """Verify local bytes without a download fallback. Never deserialize weights."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('A lowercase SHA-256 digest is required')
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Local checkpoint checksum mismatch')
    return path


class ModelLoader(Protocol):
    """Backend contract: verify local artifact, require CUDA, eval, freeze parameters.

    No training, fine-tuning, implicit CPU inference fallback or downloads.
    """
    def load(self, spec: ModelSpec, checkpoint: Path, *, device: str) -> Any: ...
