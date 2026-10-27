"""Offline LAION volumetric adapter; explicit manifest, no inferred OTC or scaling."""
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .config import LocalConfig, load_config
from .data import ResponseSet, unique_ids
from .provenance import metadata_digest
from .repeats import (REPEAT_POLICIES, VOLUME_ORDER, ORDERING_SOURCES,
                      TrialPresentation, TrialSession, aggregate_matched3)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _local_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Manifest paths must be nonempty relative filenames')
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Manifest file must remain inside configured data_root')
    return path


@dataclass(frozen=True)
class TrialRecord:
    session: str
    run: str
    beta_index: int
    image_id: str
    volume_index: int
    onset: float | None = None
    valid: bool | None = None


def _read_trials(path: Path, session: str, n_volumes: int, *, index_mode: str,
                 run_order: list[str], read_acquisition: bool = False) -> tuple[TrialRecord, ...]:
    """The documented label joins stimuli; beta_index, not TSV order, joins volumes."""
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'session', 'run', 'beta_index', 'label'} <= set(reader.fieldnames or ()):
            raise ValueError('Trials TSV requires session, run, beta_index and label')
        records = []
        for row in reader:
            if (row['session'] != session or not row['run'] or not row['label']
                    or row['label'] in ('n/a', 'NaN')):
                raise ValueError('Trial session/run/label missing or inconsistent')
            try:
                index = int(row['beta_index'])
            except (ValueError, TypeError) as exc:
                raise ValueError('beta_index must be a zero-based integer') from exc
            onset = float(row['onset']) if read_acquisition and row.get('onset', '').strip() else None
            flag = row.get('valid', '').strip().lower() if read_acquisition else ''
            if flag not in ('', '0', '1', 'false', 'true'):
                raise ValueError('Optional valid column must contain explicit boolean values')
            valid = None if not flag else flag in ('1', 'true')
            records.append(TrialRecord(session, row['run'], index, row['label'], index, onset, valid))
    if index_mode == 'run_index':
        if set(run_order) != {r.run for r in records}:
            raise ValueError('Explicit run_order must cover exactly the TSV runs')
        offset, remapped = 0, []
        for run in run_order:
            group = [r for r in records if r.run == run]
            if sorted(r.beta_index for r in group) != list(range(len(group))):
                raise ValueError('Run-local beta_index must cover each run exactly once')
            remapped.extend(replace(r, volume_index=offset+r.beta_index) for r in group)
            offset += len(group)
        records = remapped
    if len(records) != n_volumes or sorted(r.volume_index for r in records) != list(range(n_volumes)):
        raise ValueError('beta_index must map every beta volume exactly once')
    return tuple(sorted(records, key=lambda row: row.volume_index))


def _acquisition_rank(label: str, prefix: str) -> int:
    match = re.fullmatch(r'(?:' + prefix + r'-)?([0-9]+)', label)
    if not match:
        raise ValueError('Acquisition order requires numeric session/run labels')
    return int(match[1])


ACQUISITION_STATEMENT = '4D NIfTI with one volume per trial, ordered by acquisition time across runs.'


def _verify_acquisition_sidecar(sidecar):
    if (not isinstance(sidecar, dict)
            or not isinstance(sidecar.get('Description'), str)
            or ACQUISITION_STATEMENT not in sidecar['Description']):
        raise ValueError('Beta sidecar must explicitly state acquisition-time volume order')


def matched3_trial_presentations(records: tuple[TrialRecord, ...], *, subject_id: str,
                                 index_mode: str, ordering_source: str,
                                 validity_policy: str, beta_sidecar: dict | None = None) -> tuple[TrialPresentation, ...]:
    """Metadata-only conversion; acquisition volume order requires sidecar evidence.

    No beta values or events files are read here. The beta-index policy requires
    the validated run-local mapping, not TSV row order or session-global indices.
    """
    if ordering_source not in ORDERING_SOURCES:
        raise ValueError('Explicit supported ordering source required')
    if ordering_source == 'glmsingle_beta_index_order_v1' and index_mode != 'run_index':
        raise ValueError('Beta-index ordering requires explicit run_index mapping')
    if validity_policy not in ('tsv_valid_column_v1', 'all_presentations_valid_v1'):
        raise ValueError('Explicit supported validity policy required')
    if ordering_source == VOLUME_ORDER:
        _verify_acquisition_sidecar(beta_sidecar)
        if sorted(r.volume_index for r in records) != list(range(len(records))):
            raise ValueError('Acquisition volume indices must cover the session exactly once')
    trials = []
    for r in records:
        _acquisition_rank(r.session, 'ses')
        if ordering_source == 'tsv_onset_v1' and r.onset is None:
            raise ValueError('Onset ordering requires explicit trial onset metadata')
        if validity_policy == 'tsv_valid_column_v1' and r.valid is None:
            raise ValueError('Validity column is required for every trial')
        if validity_policy == 'all_presentations_valid_v1' and r.valid is False:
            raise ValueError('All-valid declaration conflicts with explicit invalid flag')
        trials.append(TrialPresentation(
            f'{subject_id}/{r.session}/{r.run}/volume-{r.volume_index}', r.image_id,
            _acquisition_rank(r.run, 'run'),
            r.onset if ordering_source == 'tsv_onset_v1' else None,
            r.valid if validity_policy == 'tsv_valid_column_v1' else True,
            ordering_source=ordering_source, beta_index=r.beta_index, volume_index=r.volume_index))
    return tuple(trials)


class LaionNiftiAdapter:
    """DatasetAdapter for explicitly listed T1w NIfTI sessions and a supplied mask.

    All-listed means and explicit normalized matched-three selection are supported.
    These are LAION adaptations, not identical NSD trials. The manifest
    must document upstream normalization, mask/reliability and session coverage.
    No automatic ROI construction, reliability threshold or resampling occurs.
    """
    def __init__(self, config: LocalConfig, *, manifest: str):
        self.config = config
        self._root = config.storage['data_root']
        self._manifest_path = _local_file(self._root, manifest)
        self._manifest_bytes = self._manifest_path.read_bytes()
        m = self.manifest
        required = {'schema_version', 'subject_id', 'roi_id', 'space', 'mask',
                    'sessions', 'aggregation', 'normalization', 'provenance'}
        if not isinstance(m, dict) or set(m) != required or m['schema_version'] != 1:
            raise ValueError('Unsupported LAION adapter manifest schema')
        if any(not isinstance(m[k], str) or not m[k] for k in ('subject_id', 'roi_id', 'space')):
            raise ValueError('Subject, ROI and space must be explicit strings')
        if m['space'] != 'T1w':
            raise ValueError('Only explicit native T1w volumetric inputs are supported')
        if not ((m['aggregation'] == 'mean_all_listed_trials_v1' and m['normalization'] == 'none')
                or (m['aggregation'] in REPEAT_POLICIES and m['normalization'] in
                    ('legacy_deepnsd_session_v1', 'train_only_session_v1'))):
            raise ValueError('Explicit supported aggregation/normalization pair required')
        p = m['provenance']
        required_provenance = {'dataset_release', 'roi_definition', 'reliability',
                               'upstream_normalization', 'session_coverage'}
        if (not isinstance(p, dict) or not required_provenance <= set(p)
                or any(not isinstance(v, str) or not v for v in p.values())):
            raise ValueError('Explicit dataset, ROI, reliability, normalization and session provenance required')
        sessions = m['sessions']
        if not isinstance(sessions, list) or not sessions:
            raise ValueError('An explicit nonempty session list is required')
        for s in sessions:
            if (not isinstance(s, dict) or set(s) - {'beta_sidecar'} != {'session_id', 'betas', 'trials', 'index_mode', 'run_order'}
                    or any(not isinstance(s[k], str) or not s[k] for k in ('session_id', 'betas', 'trials'))):
                raise ValueError('Each session requires explicit files and index/run-order policy')
            if s['index_mode'] not in ('session_index', 'run_index') or not isinstance(s['run_order'], list):
                raise ValueError('Explicit session_index or run_index mode required')
            if s['index_mode'] == 'run_index':
                unique_ids(tuple(s['run_order']), 'Run order')
            elif s['run_order']:
                raise ValueError('session_index requires empty run_order')
        unique_ids(tuple(s['session_id'] for s in sessions), 'Session IDs')

    @property
    def manifest(self):
        """Return a copy of the exact manifest snapshot used for provenance."""
        return json.loads(self._manifest_bytes)

    @classmethod
    def from_config(cls, path=None, *, manifest: str):
        return cls(load_config(path), manifest=manifest)

    def trials(self) -> tuple[TrialRecord, ...]:
        """Inspect validated metadata in manifest session and beta-index order."""
        import nibabel as nib
        records = []
        for entry in self.manifest['sessions']:
            beta = nib.load(_local_file(self._root, entry['betas']), mmap=True)
            if len(beta.shape) != 4:
                raise ValueError('Betas must be a 4D trial volume')
            records.extend(_read_trials(_local_file(self._root, entry['trials']),
                                        entry['session_id'], beta.shape[3],
                                        index_mode=entry['index_mode'], run_order=entry['run_order'],
                                        read_acquisition=self.manifest['aggregation'] in REPEAT_POLICIES))
        return tuple(records)

    def _prepare_response_inputs(self, subject_id: str, roi_id: str, image_ids: tuple[str, ...]):
        import nibabel as nib
        m = self.manifest
        unique_ids(image_ids, 'Requested image IDs')
        if (subject_id, roi_id) != (m['subject_id'], m['roi_id']):
            raise ValueError('Requested subject/ROI differs from manifest')
        # Verify all geometry and trial mappings before accumulating any response.
        mask_path = _local_file(self._root, m['mask'])
        mask_img = nib.load(mask_path)
        mask = np.asarray(mask_img.dataobj)
        if mask.ndim != 3 or not np.isin(mask, [0, 1]).all() or not mask.any():
            raise ValueError('ROI mask must be a nonempty binary 3D volume')
        if not np.isfinite(mask_img.affine).all() or np.linalg.det(mask_img.affine[:3, :3]) == 0:
            raise ValueError('ROI affine must be finite and invertible')
        coords = np.nonzero(mask)
        sessions, seen, source_hashes = [], set(), []
        for entry in m['sessions']:
            beta_path = _local_file(self._root, entry['betas'])
            trials_path = _local_file(self._root, entry['trials'])
            if beta_path in seen:
                raise ValueError('The same beta file cannot count as multiple sessions')
            seen.add(beta_path)
            beta = nib.load(beta_path, mmap=True)
            if (len(beta.shape) != 4 or beta.shape[:3] != mask.shape
                    or not np.allclose(beta.affine, mask_img.affine, rtol=0, atol=1e-5)):
                raise ValueError('Beta and mask shape/affine mismatch; no implicit resampling')
            records = _read_trials(trials_path, entry['session_id'], beta.shape[3],
                                   index_mode=entry['index_mode'], run_order=entry['run_order'],
                                   read_acquisition=m['aggregation'] in REPEAT_POLICIES)
            sessions.append((beta, records))
            source_hashes.append({'session_id': entry['session_id'],
                                  'betas_sha256': _digest(beta_path), 'trials_sha256': _digest(trials_path)})
        return m, mask_path, mask_img, mask, coords, sessions, source_hashes

    def responses(self, subject_id: str, roi_id: str, image_ids: tuple[str, ...]) -> ResponseSet:
        if self.manifest['aggregation'] in REPEAT_POLICIES:
            raise ValueError('Use responses_matched3 with explicit normalization/training/validity parameters')
        import nibabel as nib
        m, mask_path, mask_img, mask, coords, sessions, source_hashes = self._prepare_response_inputs(
            subject_id, roi_id, image_ids)
        lookup = {label: i for i, label in enumerate(image_ids)}
        counts = np.zeros(len(image_ids), dtype=np.int64)
        sums = np.zeros((len(image_ids), len(coords[0])), dtype=np.float64)
        for beta, records in sessions:
            for record in records:
                if record.image_id not in lookup:
                    continue
                # One volume at a time: no full 4D array or GPU allocation.
                values = np.asarray(beta.dataobj[..., record.volume_index])[coords]
                if not np.isfinite(values).all():
                    raise ValueError('Selected ROI contains nonfinite beta values')
                i = lookup[record.image_id]
                sums[i] += values
                counts[i] += 1
        if np.any(counts == 0):
            raise ValueError('Requested image IDs are missing from listed sessions')
        values = sums / counts[:, None]
        voxel_ids = tuple(f'T1w:{i}:{j}:{k}' for i, j, k in zip(*coords))
        provenance = dict(m['provenance'])
        provenance.update({
            'adapter': 'laion-nifti-v1', 'nibabel_version': nib.__version__,
            'manifest_sha256': hashlib.sha256(self._manifest_bytes).hexdigest(),
            'source_files_sha256': metadata_digest(source_hashes),
            'source_files': json.dumps(source_hashes, sort_keys=True),
            'mask_sha256': _digest(mask_path),
            'grid_sha256': metadata_digest({'shape': list(mask.shape), 'affine': mask_img.affine.tolist()}),
            'aggregation': m['aggregation'], 'normalization': m['normalization'],
            'response_precision': 'float64', 'trial_selection': 'all occurrences of requested labels in listed sessions',
        })
        return ResponseSet(image_ids, voxel_ids, values, subject_id, roi_id, m['space'],
                           tuple(int(n) for n in counts), provenance)


    def responses_matched3(self, subject_id: str, roi_id: str, image_ids: tuple[str, ...], *,
                           repeat_policy: str, normalization_policy: str,
                           training_image_ids: tuple[str, ...] | None,
                           validity_policy: str, ordering_source: str) -> ResponseSet:
        """Read sessions once, normalize valid rows, select/average first three, persist manifest.

        Numeric BIDS labels define ranks; the within-run ordering source is required.
        Validity must be explicitly declared from the TSV or as all presentations
        valid. No inference from beta magnitudes or implicit bad-trial exclusion.
        """
        m = self.manifest
        if (repeat_policy not in REPEAT_POLICIES or m['aggregation'] != repeat_policy
                or m['normalization'] != normalization_policy):
            raise ValueError('Matched3 call must agree with manifest aggregation/normalization')
        if ordering_source not in ORDERING_SOURCES:
            raise ValueError('Explicit supported ordering source required')
        if validity_policy not in ('tsv_valid_column_v1', 'all_presentations_valid_v1'):
            raise ValueError('Explicit supported validity policy required')
        m, mask_path, mask_img, mask, coords, sessions, source_hashes = self._prepare_response_inputs(
            subject_id, roi_id, image_ids)
        sidecars = {}
        if ordering_source == VOLUME_ORDER:
            for entry, hashes in zip(m['sessions'], source_hashes):
                if not entry.get('beta_sidecar'):
                    raise ValueError('Volume ordering requires an explicit beta_sidecar file')
                path = _local_file(self._root, entry['beta_sidecar'])
                sidecars[entry['session_id']] = json.loads(path.read_text())
                _verify_acquisition_sidecar(sidecars[entry['session_id']])
                hashes['beta_sidecar_sha256'] = _digest(path)
        def session_arrays():
            for (beta, records), entry in zip(sessions, m['sessions']):
                trials = matched3_trial_presentations(records, subject_id=subject_id,
                    index_mode=entry['index_mode'], ordering_source=ordering_source,
                    validity_policy=validity_policy, beta_sidecar=sidecars.get(entry['session_id']))
                # One session's ROI matrix, never the full dataset or GPU memory.
                values = np.stack([np.asarray(beta.dataobj[..., r.volume_index])[coords] for r in records])
                yield TrialSession(entry['session_id'], _acquisition_rank(entry['session_id'], 'ses'), tuple(trials), values)
        provenance = {**m['provenance'], 'adapter': 'laion-nifti-matched3-v1',
                      'manifest_sha256': hashlib.sha256(self._manifest_bytes).hexdigest(),
                      'source_files_sha256': metadata_digest(source_hashes),
                      'source_files': json.dumps(source_hashes, sort_keys=True),
                      'mask_sha256': _digest(mask_path), 'validity_policy': validity_policy,
                      'ordering_source': ordering_source,
                      'grid_sha256': metadata_digest({'shape': list(mask.shape), 'affine': mask_img.affine.tolist()})}
        return aggregate_matched3(session_arrays(), config=self.config, image_ids=image_ids,
            voxel_ids=tuple(f'T1w:{i}:{j}:{k}' for i,j,k in zip(*coords)),
            subject_id=subject_id, roi_id=roi_id, space=m['space'], provenance=provenance,
            repeat_policy=repeat_policy, normalization_policy=normalization_policy,
            training_image_ids=training_image_ids)
