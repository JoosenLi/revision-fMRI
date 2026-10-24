"""Data contracts. Readers must preserve stimulus, voxel, trial and space provenance."""
from dataclasses import dataclass
from typing import Mapping, Protocol
import numpy as np
from numpy.typing import NDArray


def unique_ids(ids: tuple[str, ...], label: str) -> None:
    if not ids or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{label} must be unique nonempty string IDs')


@dataclass(frozen=True)
class ResponseSet:
    """Image-averaged responses, images × voxels, within one subject/ROI/space.

    Aggregation is deliberately not automatic: provenance must describe its policy.
    """
    image_ids: tuple[str, ...]
    voxel_ids: tuple[str, ...]
    values: NDArray
    subject_id: str
    roi_id: str
    space: str
    repeat_counts: tuple[int, ...]
    provenance: Mapping[str, str]

    def __post_init__(self):
        unique_ids(self.image_ids, 'Image IDs')
        unique_ids(self.voxel_ids, 'Voxel IDs')
        if self.values.shape != (len(self.image_ids), len(self.voxel_ids)):
            raise ValueError('Responses must have image × voxel axes')
        if not np.isfinite(self.values).all():
            raise ValueError('Responses must be finite')
        if len(self.repeat_counts) != len(self.image_ids) or any(type(n) is not int or n < 1 for n in self.repeat_counts):
            raise ValueError('Require positive integer repeat counts for every image')
        if not all((self.subject_id, self.roi_id, self.space, self.provenance)):
            raise ValueError('Subject, ROI, space and provenance are required')


class DatasetAdapter(Protocol):
    """Future local-data adapter; must not download or infer ROI definitions."""
    def responses(self, subject_id: str, roi_id: str, image_ids: tuple[str, ...]) -> ResponseSet: ...
