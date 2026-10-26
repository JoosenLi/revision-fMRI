"""Real tiny NIfTI/TSV fixtures: verify the dataset boundary without real fMRI."""
from dataclasses import replace
import json
import numpy as np
import nibabel as nib
import pytest
from revision_fmri.config import load_config
from revision_fmri.laion import LaionNiftiAdapter
from revision_fmri.evaluation import EvaluationSplit


@pytest.fixture
def local_data(tmp_path):
    cfg = load_config('configs/paths.example.yaml')
    cfg = replace(cfg, storage={**cfg.storage, 'data_root': tmp_path})
    affine = np.eye(4)
    mask = np.zeros((2, 2, 2), dtype='uint8')
    mask[0, 0, 0] = mask[1, 0, 1] = 1
    nib.save(nib.Nifti1Image(mask, affine), tmp_path/'mask.nii')
    sessions = []
    for s in ('01', '02'):
        values = np.zeros((2, 2, 2, 3), dtype='float32')
        values[0, 0, 0] = [1, 5, 3] if s == '01' else [7, 11, 9]
        values[1, 0, 1] = values[0, 0, 0] * 2
        nib.save(nib.Nifti1Image(values, affine), tmp_path/f'{s}.nii')
        # Intentionally permuted TSV rows: index must win over row position.
        (tmp_path/f'{s}.tsv').write_text(
            f'session\trun\tbeta_index\tlabel\n{s}\t1\t2\ta.jpg\n{s}\t1\t0\ta.jpg\n{s}\t1\t1\tb.jpg\n')
        sessions.append({'session_id': s, 'betas': f'{s}.nii', 'trials': f'{s}.tsv',
                         'index_mode': 'session_index', 'run_order': []})
    manifest = {'schema_version': 1, 'subject_id': 'sub-01', 'roi_id': 'synthetic',
                'space': 'T1w', 'mask': 'mask.nii', 'sessions': sessions,
                'aggregation': 'mean_all_listed_trials_v1', 'normalization': 'none',
                'provenance': {k: 'synthetic fixture' for k in ['dataset_release', 'roi_definition',
                              'reliability', 'upstream_normalization', 'session_coverage']}}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    return cfg, path, manifest


def adapter(local_data):
    return LaionNiftiAdapter(local_data[0], manifest='manifest.json')


def test_index_join_requested_order_repeats_and_provenance(local_data):
    records = adapter(local_data).trials()
    assert len(records) == 6
    assert [r.beta_index for r in records] == [0, 1, 2, 0, 1, 2]
    assert records[0].image_id == 'a.jpg'
    out = adapter(local_data).responses('sub-01', 'synthetic', ('b.jpg', 'a.jpg'))
    np.testing.assert_array_equal(out.values, [[8, 16], [5, 10]])
    assert out.repeat_counts == (2, 4)
    assert out.voxel_ids == ('T1w:0:0:0', 'T1w:1:0:1')
    assert out.values.dtype == np.float64
    assert 'mask_sha256' in out.provenance
    assert str(local_data[1].parent) not in json.dumps(out.provenance)
    split = EvaluationSplit(('a.jpg',), ('b.jpg',), 'test', 'hash')
    split.validate_coverage(out.image_ids)
    # All four a trials and both b trials remain in their respective image groups.
    assert out.repeat_counts[out.image_ids.index(split.train_ids[0])] == 4


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'negative', 'session', 'blank', 'fraction'])
def test_bad_trial_mapping(local_data, kind):
    p = local_data[1].parent/'01.tsv'
    text = p.read_text()
    replacements = {'duplicate': ('\t2\t', '\t0\t'), 'missing': ('01\t1\t2\ta.jpg\n', ''),
                    'negative': ('\t2\t', '\t-1\t'), 'session': ('01\t1', '03\t1'),
                    'blank': ('a.jpg', ''), 'fraction': ('\t2\t', '\t2.5\t')}
    text = text.replace(*replacements[kind])
    p.write_text(text)
    with pytest.raises(ValueError):
        adapter(local_data).responses('sub-01', 'synthetic', ('a.jpg', 'b.jpg'))


@pytest.mark.parametrize('kind', ['affine', 'shape', 'nan', 'mask'])
def test_geometry_and_finite_checks(local_data, kind):
    root = local_data[1].parent
    path = root/('mask.nii' if kind == 'mask' else '01.nii')
    img = nib.load(path)
    data, affine = np.array(img.dataobj), img.affine.copy()
    if kind == 'affine': affine[0, 3] = 3
    if kind == 'shape': data = data[:1]
    if kind == 'nan': data[0, 0, 0, 0] = np.nan
    if kind == 'mask': data[0, 0, 0] = 2
    nib.save(nib.Nifti1Image(data, affine), path)
    with pytest.raises(ValueError):
        adapter(local_data).responses('sub-01', 'synthetic', ('a.jpg', 'b.jpg'))


def test_missing_image_and_identity(local_data):
    obj = adapter(local_data)
    with pytest.raises(ValueError, match='missing'):
        obj.responses('sub-01', 'synthetic', ('absent.jpg',))
    with pytest.raises(ValueError, match='subject/ROI'):
        obj.responses('sub-02', 'synthetic', ('a.jpg',))


@pytest.mark.parametrize('kind', ['normalization', 'aggregation', 'reliability', 'duplicate_session', 'same_beta', 'absolute'])
def test_explicit_protocol_and_files(local_data, kind):
    cfg, path, m = local_data
    if kind in ('normalization', 'aggregation'): m[kind] = 'auto'
    if kind == 'reliability': del m['provenance']['reliability']
    if kind == 'duplicate_session': m['sessions'][1]['session_id'] = '01'
    if kind == 'same_beta': m['sessions'][1]['betas'] = '01.nii'
    if kind == 'absolute': m['mask'] = str(path.parent/'mask.nii')
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        adapter(local_data).responses('sub-01', 'synthetic', ('a.jpg',))


def test_run_local_indices_need_explicit_order(local_data):
    cfg, path, m = local_data
    for entry in m['sessions']:
        s = entry['session_id']
        entry.update(index_mode='run_index', run_order=['01', '02'])
        (path.parent/entry['trials']).write_text(
            f'session\trun\tbeta_index\tlabel\n{s}\t02\t0\ta.jpg\n{s}\t01\t1\tb.jpg\n{s}\t01\t0\ta.jpg\n')
    path.write_text(json.dumps(m))
    obj = adapter(local_data)
    assert [r.volume_index for r in obj.trials()] == [0, 1, 2, 0, 1, 2]
    np.testing.assert_array_equal(obj.responses('sub-01', 'synthetic', ('b.jpg', 'a.jpg')).values,
                                  [[8, 16], [5, 10]])
    m['sessions'][0]['run_order'] = ['01']
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='run_order'):
        adapter(local_data).trials()


def test_matched3_nifti_reader_normalizes_sessions_before_selection(local_data):
    from revision_fmri import MATCHED3_POLICY, validate_matched3_responses
    cfg, path, manifest = local_data
    cfg = replace(cfg, storage={**cfg.storage, 'results_root': path.parent/'results'})
    manifest.update(aggregation=MATCHED3_POLICY, normalization='train_only_session_v1')
    manifest['sessions'].reverse()  # Must not determine acquisition order.
    for entry in manifest['sessions']:
        s=entry['session_id']
        (path.parent/entry['trials']).write_text(
            f'session\trun\tbeta_index\tlabel\tonset\tvalid\n{s}\t1\t2\ta.jpg\t8\t1\n'
            f'{s}\t1\t0\ta.jpg\t0\t1\n{s}\t1\t1\tb.jpg\t4\t1\n')
    path.write_text(json.dumps(manifest))
    obj=LaionNiftiAdapter(cfg,manifest='manifest.json')
    with pytest.raises(ValueError,match='responses_matched3'):
        obj.responses('sub-01','synthetic',('a.jpg',))
    kwargs=dict(repeat_policy=MATCHED3_POLICY,normalization_policy='train_only_session_v1',
                training_image_ids=('a.jpg',),validity_policy='tsv_valid_column_v1')
    out=obj.responses_matched3('sub-01','synthetic',('a.jpg',),**kwargs)
    np.testing.assert_allclose(out.values,[[-1/3,-1/3]])
    selection=validate_matched3_responses(out)
    assert [t['trial_id'] for t in selection['images'][0]['selected_trials']] == [
        'sub-01/01/1/volume-0','sub-01/01/1/volume-2','sub-01/02/1/volume-0']
    assert (cfg.storage['results_root']/out.provenance['trial_selection_file']).exists()
    with pytest.raises(ValueError,match='Fewer than 3'):
        obj.responses_matched3('sub-01','synthetic',('a.jpg','b.jpg'),**kwargs)
    p=path.parent/'01.tsv';p.write_text(p.read_text().replace('\tonset\t','\tmissing_onset\t'))
    with pytest.raises(ValueError,match='onset'):
        obj.responses_matched3('sub-01','synthetic',('a.jpg',),**kwargs)
