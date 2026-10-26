"""Explicit voxel-aligned OTC union and NCSNR selection; no inferred LAION masks."""
from dataclasses import dataclass, asdict
from typing import Mapping
import json
import numpy as np
from .data import ResponseSet, unique_ids
from .provenance import array_digest, metadata_digest


@dataclass(frozen=True)
class VoxelMap:
    voxel_ids: tuple[str, ...]
    values: np.ndarray
    subject_id: str
    space: str
    grid_sha256: str
    kind: str  # binary_mask or ncsnr; NC percentages are deliberately rejected.
    provenance: Mapping[str, str]

    def __post_init__(self):
        unique_ids(self.voxel_ids, 'Voxel IDs')
        if self.values.shape != (len(self.voxel_ids),) or not np.isfinite(self.values).all():
            raise ValueError('Voxel map must be a finite vector aligned to voxel IDs')
        if not all((self.subject_id, self.space, self.grid_sha256, self.provenance)):
            raise ValueError('Subject, space, grid hash and provenance are required')
        if self.kind == 'binary_mask':
            if not np.isin(self.values, [0, 1]).all():
                raise ValueError('Mask must be binary')
        elif self.kind == 'ncsnr':
            if np.any(self.values < 0):
                raise ValueError('NCSNR must be nonnegative')
        else:
            raise ValueError('Only binary_mask or ncsnr units are supported; not NC percent/R2')


@dataclass(frozen=True)
class OTCProtocol:
    protocol_id: str
    category_names: tuple[str, ...]
    ncsnr_threshold: float
    threshold_operator: str
    mapping_decisions: Mapping[str, str]

    def __post_init__(self):
        unique_ids(self.category_names, 'Explicit category names')
        decisions = {'category_threshold', 'missing_regions', 'stream_mapping', 'reliability_scope'}
        if (not self.protocol_id or not decisions <= set(self.mapping_decisions)
                or any(not isinstance(v, str) or not v for v in self.mapping_decisions.values())):
            raise ValueError('Named protocol and explicit mapping decisions required')
        if self.threshold_operator not in ('gt', 'ge'):
            raise ValueError('Choose gt (paper) or ge (released threshold boundary) explicitly')
        if not np.isfinite(self.ncsnr_threshold) or self.ncsnr_threshold < 0:
            raise ValueError('Explicit finite nonnegative NCSNR threshold required')


@dataclass(frozen=True)
class ROISelection:
    voxel_ids: tuple[str, ...]
    selected: np.ndarray
    subject_id: str
    space: str
    grid_sha256: str
    roi_id: str
    provenance: Mapping[str, str]
    counts: Mapping[str, int]

    def __post_init__(self):
        unique_ids(self.voxel_ids, 'Selection voxel IDs')
        if (self.selected.shape != (len(self.voxel_ids),) or self.selected.dtype != np.bool_
                or not self.selected.any()):
            raise ValueError('Selection must be a nonempty aligned boolean mask')
        if not all((self.subject_id, self.space, self.grid_sha256, self.roi_id, self.provenance)):
            raise ValueError('Selection identity and provenance are required')

    def apply(self, responses: ResponseSet) -> ResponseSet:
        if (responses.voxel_ids != self.voxel_ids or responses.subject_id != self.subject_id
                or responses.space != self.space
                or responses.provenance.get('grid_sha256') != self.grid_sha256):
            raise ValueError('Response voxel order, subject, space or grid differs from ROI selection')
        return ResponseSet(responses.image_ids,
                           tuple(v for v, keep in zip(self.voxel_ids, self.selected) if keep),
                           responses.values[:, self.selected], responses.subject_id, self.roi_id,
                           responses.space, responses.repeat_counts,
                           {**responses.provenance, **self.provenance})


def select_otc(*, general: VoxelMap, lateral: VoxelMap, ventral: VoxelMap,
               categories: Mapping[str, VoxelMap], reliability: VoxelMap,
               protocol: OTCProtocol) -> ROISelection:
    """reliable ∩ ((general ∩ (lateral ∪ ventral)) ∪ categories).

    Categories may extend outside general. Inputs already encode any localizer
    t-threshold. No t statistic is reconstructed from a binary mask.
    """
    if set(categories) != set(protocol.category_names):
        raise ValueError('Category maps must exactly match protocol; no silent missing/extra ROI')
    maps = {'general': general, 'lateral': lateral, 'ventral': ventral,
            **{f'category:{k}': v for k, v in categories.items()}, 'reliability': reliability}
    for name, m in maps.items():
        if (m.voxel_ids != general.voxel_ids or m.subject_id != general.subject_id
                or m.space != general.space or m.grid_sha256 != general.grid_sha256):
            raise ValueError('All maps must share voxel order, subject, space and grid')
        if m.kind != ('ncsnr' if name == 'reliability' else 'binary_mask'):
            raise ValueError('Incorrect map units for mask/reliability input')
    core = general.values.astype(bool) & (lateral.values.astype(bool) | ventral.values.astype(bool))
    category_union = np.logical_or.reduce([categories[k].values.astype(bool) for k in protocol.category_names])
    candidate = core | category_union
    reliable = (reliability.values > protocol.ncsnr_threshold if protocol.threshold_operator == 'gt'
                else reliability.values >= protocol.ncsnr_threshold)
    selected = candidate & reliable
    if not selected.any():
        raise ValueError('No voxels survive the explicitly requested ROI/reliability protocol')
    counts = {'core': int(core.sum()), 'category_union': int(category_union.sum()),
              'overlap': int((core & category_union).sum()), 'candidate_union': int(candidate.sum()),
              'excluded_reliability': int((candidate & ~reliable).sum()), 'selected': int(selected.sum())}
    evidence = {k: {'values_sha256': array_digest(m.values), 'provenance': dict(m.provenance)}
                for k, m in maps.items()}
    provenance = {'roi_protocol_sha256': metadata_digest(asdict(protocol)),
                  'roi_protocol': json.dumps(asdict(protocol), sort_keys=True),
                  'roi_inputs': json.dumps(evidence, sort_keys=True),
                  'roi_counts': json.dumps(counts, sort_keys=True),
                  'roi_inputs_sha256': metadata_digest(evidence),
                  'roi_selected_sha256': array_digest(selected),
                  'roi_counts_sha256': metadata_digest(counts),
                  'grid_sha256': general.grid_sha256}
    return ROISelection(general.voxel_ids, selected, general.subject_id, general.space,
                        general.grid_sha256, protocol.protocol_id, provenance, counts)
