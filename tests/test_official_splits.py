"""Offline schema/coverage tests; actual source parity is checked by a separate script."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from revision_fmri import (load_official_split,partition_trials,validate_holdout_family,
                           RecordedSplit,EvaluationSplit,official_split_inventory)
from revision_fmri.config import load_config
from revision_fmri import official_splits


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    cfg=load_config('configs/paths.example.yaml')
    cfg=replace(cfg,storage={**cfg.storage,'data_root':tmp_path})
    payload={'name':'tau','pool':'human readable pool','n_train':2,'n_test':1,
             'variants':[{'variant_id':0,'train_ids':['b.jpg','a.jpg'],'test_ids':['c.jpg']}]}
    raw=json.dumps(payload).encode();rel='laion_fmri/splits/data/shared/tau.json'
    path=tmp_path/'official_splits'/'fixture'/rel;path.parent.mkdir(parents=True);path.write_bytes(raw)
    lock={'commit':'fixture','files':{rel:{'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                          'n_train':2,'n_test':1,'raw_pool':'human readable pool'}}}
    monkeypatch.setattr(official_splits,'official_split_inventory',lambda:lock)
    return cfg,path,lock


def test_official_import_preserves_exact_order_and_metadata(fixture):
    cfg,_,_=fixture
    r=load_official_split(cfg,name='tau',dataset_release='reviewed-release')
    assert r.split.train_ids==('b.jpg','a.jpg') and r.split.test_ids==('c.jpg',)
    assert r.pool=='shared' and r.source_revision=='fixture'
    p=partition_trials(['a.jpg','c.jpg','b.jpg','c.jpg','outside'],split=r)
    np.testing.assert_array_equal(p.train,[1,0,1,0,0])
    np.testing.assert_array_equal(p.test,[0,1,0,1,0])
    np.testing.assert_array_equal(p.excluded,[0,0,0,0,1])
    assert not p.missing_test_ids and not p.missing_train_ids


def test_partial_sessions_do_not_silently_change_official_split(fixture):
    cfg,_,_=fixture;r=load_official_split(cfg,name='tau',dataset_release='fixture')
    with pytest.raises(ValueError,match='Incomplete'):partition_trials(['b.jpg','b.jpg'],split=r)
    result=partition_trials(['b.jpg','b.jpg','unknown'],split=r,require_complete=False)
    assert result.missing_train_ids==('a.jpg',) and result.missing_test_ids==('c.jpg',)
    assert r.split.train_ids==('b.jpg','a.jpg')
    with pytest.raises(ValueError,match='sequence'):partition_trials('b.jpg',split=r)


@pytest.mark.parametrize('name,variant',[('../escape',0),('ood',0),('tau',1),('tau',True)])
def test_unknown_names_and_variants(fixture,name,variant):
    with pytest.raises(ValueError):load_official_split(fixture[0],name=name,dataset_release='fixture',variant_id=variant)


def test_changed_official_bytes_rejected(fixture):
    cfg,path,_=fixture;path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='pinned'):load_official_split(cfg,name='tau',dataset_release='fixture')


def folds():
    ids=tuple(f'i{k}' for k in range(5))
    return [RecordedSplit(EvaluationSplit(tuple(x for x in ids if x!=held),(held,),f'fold-{i}','a'*64),
                          'release','revision','shared') for i,held in enumerate(ids)]


def test_holdouts_disjoint_exhaustive_and_compatible():
    family=folds();validate_holdout_family(family)
    with pytest.raises(ValueError,match='five'):validate_holdout_family(family[:4])
    with pytest.raises(ValueError,match='agree'):
        validate_holdout_family([*family[:4],replace(family[4],dataset_release='other')])
    other=replace(family[0],split=replace(family[0].split,protocol_id='other'))
    with pytest.raises(ValueError,match='overlap'):validate_holdout_family([*family[:4],other])


def test_packaged_lock_contains_only_expected_small_files():
    lock=official_split_inventory()
    assert lock['commit']=='cf82765ced029de81c3bec7254b8cbd6a65e3686'
    assert len(lock['files'])==13
    assert sum(f['size'] for f in lock['files'].values())<1_000_000
    for path,entry in lock['files'].items():
        assert path.startswith('laion_fmri/splits/') and '..' not in path
        assert len(entry['sha256'])==64
