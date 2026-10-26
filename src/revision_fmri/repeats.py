"""Explicit chronological matched-three aggregation after session normalization."""
from dataclasses import dataclass
import json
from typing import Iterable, Mapping
import numpy as np
from .config import LocalConfig
from .data import ResponseSet, unique_ids
from .provenance import array_digest, metadata_digest
from .response import normalize_session

MATCHED3_POLICY = 'matched3_first_chronological_v1'


@dataclass(frozen=True)
class TrialPresentation:
    trial_id: str
    image_id: str
    run_order: int
    onset: float
    valid: bool


@dataclass(frozen=True)
class TrialSession:
    session_id: str
    session_order: int
    trials: tuple[TrialPresentation, ...]
    values: np.ndarray  # trials x voxels, before session normalization


def _span(trials):
    ordered = sorted({(t['session_order'], t['session_id']) for t in trials})
    return {'first_session': ordered[0][1], 'last_session': ordered[-1][1],
            'first_session_order': ordered[0][0], 'last_session_order': ordered[-1][0],
            'n_sessions': len(ordered), 'sessions': [x[1] for x in ordered]}


def write_trial_selection_manifest(config: LocalConfig, manifest: dict) -> str:
    """Content-addressed JSON under results_root; verify existing bytes, never overwrite."""
    digest = metadata_digest(manifest)
    relative = f'trial_selections/{digest}.json'
    root = config.storage['results_root'].resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Trial-selection path escapes configured results_root')
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps({'content_sha256': digest, 'manifest': manifest},
                          sort_keys=True, indent=2, allow_nan=False) + '\n'
    try:
        with path.open('x') as stream:
            stream.write(document)
    except FileExistsError:
        if path.read_text() != document:
            raise ValueError('Existing trial-selection artifact differs; no overwrite')
    return relative


def validate_matched3_responses(responses: ResponseSet) -> dict:
    """Verify embedded manifest, selected trial order/counts, and response binding."""
    if responses.provenance.get('aggregation') != MATCHED3_POLICY or any(n != 3 for n in responses.repeat_counts):
        raise ValueError('Primary responses require matched3 aggregation and exactly three repeats per image')
    try:
        manifest = json.loads(responses.provenance['trial_selection_manifest'])
        expected = responses.provenance['trial_selection_sha256']
        if responses.provenance.get('trial_selection_file') != f'trial_selections/{expected}.json':
            raise ValueError('Trial-selection artifact reference mismatch')
        if metadata_digest(manifest) != expected:
            raise ValueError('Trial-selection manifest hash mismatch')
        if (manifest['schema_version'] != 1 or manifest['seed'] is not None
                or manifest['normalization'] not in ('legacy_deepnsd_session_v1', 'train_only_session_v1')
                or manifest['policy'] != MATCHED3_POLICY or tuple(manifest['image_ids']) != responses.image_ids
                or manifest['response_values_sha256'] != array_digest(responses.values)
                or manifest['voxel_ids_sha256'] != metadata_digest(responses.voxel_ids)
                or manifest['subject_id'] != responses.subject_id or manifest['roi_id'] != responses.roi_id
                or manifest['space'] != responses.space
                or manifest['dataset_release'] != responses.provenance['dataset_release']
                or manifest['normalization'] != responses.provenance['normalization']):
            raise ValueError('Trial-selection manifest does not match responses')
        if tuple(row['image_id'] for row in manifest['images']) != responses.image_ids:
            raise ValueError('Trial-selection image order mismatch')
        seen = set()
        for row in manifest['images']:
            selected = row['selected_trials']
            eligible = row['valid_presentations']
            order = lambda t: (t['session_order'], t['run_order'], t['onset'])
            if (len(eligible) != row['n_valid_available'] or eligible != sorted(eligible, key=order)
                    or selected != eligible[:3]
                    or any(t['valid'] is not True or t['image_id'] != row['image_id'] for t in eligible)
                    or len({t['trial_id'] for t in eligible}) != len(eligible)
                    or row['session_span'] != _span(selected)):
                raise ValueError('Manifest must select the earliest three valid presentations')
            if len(selected) != 3 or row['n_valid_available'] < 3:
                raise ValueError('Require exactly three selected valid trials per image')
            keys = [(t['session_order'], t['run_order'], t['onset']) for t in selected]
            if keys != sorted(keys) or len(set(keys)) != 3:
                raise ValueError('Selected trial chronology is ambiguous or unordered')
            for trial in selected:
                if trial['trial_id'] in seen:
                    raise ValueError('Selected trial ID reused')
                seen.add(trial['trial_id'])
        if manifest['session_span'] != _span([t for row in manifest['images'] for t in row['selected_trials']]):
            raise ValueError('Selected session span mismatch')
    except (KeyError, TypeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError('Complete trial-selection manifest provenance is required') from exc
    return manifest


def aggregate_matched3(sessions: Iterable[TrialSession], *, config: LocalConfig,
                       image_ids: tuple[str, ...], voxel_ids: tuple[str, ...],
                       subject_id: str, roi_id: str, space: str,
                       provenance: Mapping[str, str], repeat_policy: str,
                       normalization_policy: str,
                       training_image_ids: tuple[str, ...] | None) -> ResponseSet:
    """Normalize all valid session rows, then average earliest three valid presentations.

    Acquisition ranks and validity are explicit metadata, never inferred from
    array/manifest order or response magnitudes. Invalid trials are excluded from
    normalization; nonfinite valid betas fail. Only three normalized vectors per
    requested image are retained across sessions. No RNG, missing-image fallback,
    automatic normalization policy, or mean-all fallback exists.
    """
    if repeat_policy != MATCHED3_POLICY:
        raise ValueError('Unsupported primary repeat_policy')
    unique_ids(image_ids, 'Requested image IDs')
    unique_ids(voxel_ids, 'Voxel IDs')
    if not provenance.get('dataset_release'):
        raise ValueError('Explicit dataset_release required')
    if normalization_policy not in ('legacy_deepnsd_session_v1', 'train_only_session_v1'):
        raise ValueError('Explicit supported session normalization is required')
    if normalization_policy == 'train_only_session_v1':
        if training_image_ids is None:
            raise ValueError('Explicit training image IDs required for train-only normalization')
        unique_ids(training_image_ids, 'Training image IDs')
        if not set(training_image_ids) <= set(image_ids):
            raise ValueError('Training IDs must belong to the requested pool')
    elif training_image_ids is not None:
        raise ValueError('Legacy normalization must not declare training-only fit IDs')
    training = set(training_image_ids or ())
    candidates = {image: [] for image in image_ids}
    available = dict.fromkeys(image_ids, 0)
    eligible = {image: [] for image in image_ids}
    session_ids, session_orders, trial_ids, audits = set(), set(), set(), []
    for session in sessions:
        if (not isinstance(session.session_id, str) or not session.session_id or type(session.session_order) is not int or session.session_order < 0
                or session.session_id in session_ids or session.session_order in session_orders):
            raise ValueError('Session IDs/acquisition ranks must be explicit and unique')
        session_ids.add(session.session_id); session_orders.add(session.session_order)
        values = np.asarray(session.values)
        if values.shape != (len(session.trials), len(voxel_ids)) or not session.trials:
            raise ValueError('Trial metadata must match the session matrix and voxel axis')
        metadata, keys = [], set()
        for t in session.trials:
            if (not isinstance(t.trial_id, str) or not t.trial_id
                    or not isinstance(t.image_id, str) or not t.image_id
                    or t.trial_id in trial_ids or type(t.valid) is not bool
                    or type(t.run_order) is not int or t.run_order < 0
                    or isinstance(t.onset, bool) or not isinstance(t.onset, (int,float,np.integer,np.floating))
                    or not np.isfinite(t.onset) or t.onset < 0):
                raise ValueError('Valid unique trial IDs and explicit acquisition metadata required')
            key = (t.run_order, t.onset)
            if key in keys:
                raise ValueError('Ambiguous trial acquisition order: duplicate run/onset')
            keys.add(key); trial_ids.add(t.trial_id)
            metadata.append({'trial_id': t.trial_id, 'image_id': t.image_id,
                             'session_id': session.session_id, 'session_order': session.session_order,
                             'run_order': t.run_order, 'onset': float(t.onset), 'valid': t.valid})
        # Sort before normalization as well, to make floating accumulation stable
        # when TSV rows/volume storage are permuted but acquisition metadata agrees.
        valid_rows = sorted((i for i,t in enumerate(session.trials) if t.valid),
                            key=lambda i: (session.trials[i].run_order, session.trials[i].onset))
        if not valid_rows:
            audits.append({'session_id': session.session_id, 'session_order': session.session_order,
                           'input_values_sha256': array_digest(values),
                           'trial_metadata_sha256': metadata_digest(metadata),
                           'n_valid': 0, 'n_invalid': len(values), 'normalization': None,
                           'exclusion': 'no valid presentations; contributes no responses'})
            continue
        fit_rows = ([j for j,i in enumerate(valid_rows) if session.trials[i].image_id in training]
                    if normalization_policy == 'train_only_session_v1' else None)
        normalized = normalize_session(values[valid_rows], policy=normalization_policy, fit_rows=fit_rows)
        audits.append({'session_id': session.session_id, 'session_order': session.session_order,
                       'input_values_sha256': array_digest(values),
                       'trial_metadata_sha256': metadata_digest(metadata),
                       'n_valid': len(valid_rows), 'n_invalid': len(values)-len(valid_rows),
                       'normalization': normalized.provenance})
        for j,i in enumerate(valid_rows):
            t = session.trials[i]
            if t.image_id not in candidates:
                continue
            available[t.image_id] += 1
            eligible[t.image_id].append(metadata[i])
            entry = (metadata[i], normalized.values[j].copy())
            candidates[t.image_id].append(entry)
            candidates[t.image_id].sort(key=lambda e: (e[0]['session_order'], e[0]['run_order'], e[0]['onset']))
            del candidates[t.image_id][3:]
    missing = [image for image in image_ids if available[image] < 3]
    if missing:
        raise ValueError(f'Fewer than 3 valid presentations for {len(missing)} requested images; no dropping/fallback')
    values = np.stack([np.asarray([entry[1] for entry in candidates[image]], dtype=np.float64).mean(axis=0)
                       for image in image_ids])
    images = [{'image_id': image, 'n_valid_available': available[image],
               'valid_presentations': sorted(eligible[image], key=lambda t: (t['session_order'], t['run_order'], t['onset'])),
               'selected_trials': [entry[0] for entry in candidates[image]],
               'session_span': _span([entry[0] for entry in candidates[image]])} for image in image_ids]
    manifest = {'schema_version': 1, 'policy': repeat_policy, 'seed': None,
                'ordering': 'session_order, run_order, onset; no tie breaking',
                'subject_id': subject_id, 'roi_id': roi_id, 'space': space,
                'dataset_release': provenance['dataset_release'], 'image_ids': image_ids,
                'voxel_ids_sha256': metadata_digest(voxel_ids), 'normalization': normalization_policy,
                'normalization_training_ids_sha256': metadata_digest(training_image_ids),
                'operation_order': 'valid session rows -> session normalization -> first three -> float64 mean',
                'sessions': sorted(audits, key=lambda a: a['session_order']), 'images': images,
                'session_span': _span([t for row in images for t in row['selected_trials']]),
                'response_values_sha256': array_digest(values)}
    digest = metadata_digest(manifest)
    response = ResponseSet(image_ids, voxel_ids, values, subject_id, roi_id, space, (3,)*len(image_ids),
                           {**provenance, 'aggregation': repeat_policy, 'normalization': normalization_policy,
                            'response_precision': 'float64', 'trial_selection_sha256': digest,
                            'trial_selection_manifest': json.dumps(manifest, sort_keys=True, allow_nan=False),
                            'trial_selection_file': f'trial_selections/{digest}.json'})
    validate_matched3_responses(response)
    write_trial_selection_manifest(config, manifest)
    return response
