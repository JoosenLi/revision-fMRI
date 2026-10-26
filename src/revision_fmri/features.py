"""Feature API. Raw flattened features and SRP features are distinct artifacts."""
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Any
import numpy as np
from numpy.typing import NDArray
from .data import unique_ids


@dataclass(frozen=True)
class FeatureSet:
    image_ids: tuple[str, ...]
    model_id: str
    layer_id: str
    values: NDArray
    representation: str
    provenance: Mapping[str, str]

    def __post_init__(self):
        unique_ids(self.image_ids, 'Image IDs')
        if self.values.ndim != 2 or self.values.shape[0] != len(self.image_ids) or self.values.shape[1] == 0:
            raise ValueError('Features must have image × feature axes')
        if not np.isfinite(self.values).all():
            raise ValueError('Features must be finite')
        if self.representation not in {'raw_flattened', 'srp'}:
            raise ValueError('Feature representation must be explicit')
        if not all((self.model_id, self.layer_id, self.provenance)):
            raise ValueError('Model, layer and provenance are required')


class FeatureExtractor(Protocol):
    """Backend must reproduce the audited hook/dedup/candidate semantics."""
    def extract(self, model: Any, images: Iterable[Any], *, image_ids: tuple[str, ...],
                candidate_layers: tuple[str, ...]) -> Iterable[FeatureSet]: ...
