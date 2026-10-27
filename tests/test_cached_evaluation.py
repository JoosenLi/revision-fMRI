"""Portable cached evaluation, stale identity rejection and pinned image splits."""
from dataclasses import replace
import hashlib
import json
import shutil
import numpy as np
import pytest
from revision_fmri import (FeatureIdentity, FeatureStore, CachedModel, load_split,
    FeatureSet, ResponseSet, ReproductionPipeline, DeepNSDProtocol)
from revision_fmri.config import load_config


@pytest.fixture
def case(tmp_path):
    config=load_config('configs/paths.example.yaml')
    config=replace(config,storage={k:tmp_path/k for k in config.storage})
    config.storage['data_root'].mkdir()
    identity=FeatureIdentity('a'*64,'b'*64,'c'*64,'d'*64,'fixture-loader-v1',{'numpy':np.__version__},0)
    rng=np.random.default_rng(43)
    ids=tuple(f'image-{i}' for i in range(20))
    x=rng.normal(size=(20,8)).astype('float32')
    f=FeatureSet(ids,'fixture-model','layer-1',x,'raw_flattened',{'source':'synthetic'})
    y=x@rng.normal(size=(8,12)) + rng.normal(size=(20,12))*.2
    r=ResponseSet(ids,tuple(f'v{i}' for i in range(12)),y,'subject','roi','T1w',(3,)*20,
                  {'source':'synthetic','dataset_release':'fixture-v1'})
    payload={'schema_version':1,'protocol_id':'synthetic-split-v1', 'dataset_release':'fixture-v1',
             'source_revision':'fixture-source-v1','pool':'synthetic','train_ids':list(ids[:14]),'test_ids':list(ids[14:])}
    path=config.storage['data_root']/'split.json'
    raw=json.dumps(payload).encode(); path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    return config,identity,f,r,digest


def test_cached_pipeline_matches_direct_and_transfers_between_roots(case,tmp_path):
    cfg,identity,f,r,digest=case
    store=FeatureStore(cfg);key=store.put(f,identity=identity)
    model=CachedModel(f.model_id,(f.layer_id,),(key,),identity)
    split=load_split(cfg,filename='split.json',expected_sha256=digest)
    p=ReproductionPipeline(cfg,DeepNSDProtocol('synthetic-v1',5,'synthetic fixture only'))
    report=p.evaluate_cached_model(model=model,responses=r,split=split)
    direct=p.evaluate_model(features=[f],responses=r,split=split.split)
    assert report.metrics==direct.metrics
    assert report.alpha_per_layer==direct.alpha_per_layer
    assert report.evaluation_provenance['split_source_revision']=='fixture-source-v1'
    second=replace(cfg,storage={k:tmp_path/'other'/k for k in cfg.storage})
    shutil.copytree(cfg.storage['feature_root'],second.storage['feature_root'])
    shutil.copytree(cfg.storage['data_root'],second.storage['data_root'])
    second_split=load_split(second,filename='split.json',expected_sha256=digest)
    report2=ReproductionPipeline(second,p.protocol).evaluate_cached_model(model=model,responses=r,split=second_split)
    assert report.to_dict()==report2.to_dict()
    assert str(tmp_path) not in json.dumps(report.to_dict())
    loaded=store.get(key,expected_identity=identity)
    assert isinstance(loaded.values,np.memmap) and not loaded.values.flags.writeable
    with pytest.raises(FileExistsError): store.put(f,identity=identity)


@pytest.mark.parametrize('field,value',[('checkpoint_sha256','e'*64),('transform_sha256','e'*64),
    ('candidate_manifest_sha256','e'*64),('stimuli_sha256','e'*64),('seed',4),('loader_revision','new'),
    ('software_versions',{'numpy':'new'})])
def test_identity_changes_invalidate_cache(case,field,value):
    cfg,identity,f,_,_=case
    store=FeatureStore(cfg);old=store.put(f,identity=identity)
    changed=replace(identity,**{field:value})
    with pytest.raises(ValueError,match='identity'): store.get(old,expected_identity=changed)
    assert store.put(f,identity=changed)!=old


@pytest.mark.parametrize('change',['values','metadata','incomplete'])
def test_tampering_and_incomplete_artifacts_fail(case,change):
    cfg,identity,f,_,_=case
    store=FeatureStore(cfg);key=store.put(f,identity=identity);folder=store.root/key
    if change=='values':
        with (folder/'values.npy').open('wb') as stream:np.save(stream,f.values+1,allow_pickle=False)
    if change=='metadata':
        m=json.loads((folder/'metadata.json').read_text());m['layer_id']='other'
        (folder/'metadata.json').write_text(json.dumps(m))
    if change=='incomplete':(folder/'COMPLETE').unlink()
    with pytest.raises(ValueError):store.get(key,expected_identity=identity)


def test_candidate_order_and_id_checks(case):
    cfg,identity,f,r,digest=case
    store=FeatureStore(cfg);key=store.put(f,identity=identity)
    split=load_split(cfg,filename='split.json',expected_sha256=digest)
    p=ReproductionPipeline(cfg,DeepNSDProtocol('synthetic-v1',5,'synthetic'))
    model=CachedModel(f.model_id,('wrong-layer',),(key,),identity)
    with pytest.raises(ValueError,match='candidate order'):
        p.evaluate_cached_model(model=model,responses=r,split=split)
    model=replace(model,layer_ids=(f.layer_id,))
    with pytest.raises(ValueError,match='aligned'):
        p.evaluate_cached_model(model=model,responses=replace(r,image_ids=r.image_ids[::-1]),split=split)
    with pytest.raises(ValueError,match='dataset_release'):
        p.evaluate_cached_model(model=model,responses=replace(r,provenance={'dataset_release':'other'}),split=split)


def test_changed_precision_and_image_order_get_new_keys(case):
    cfg,identity,f,_,_=case
    store=FeatureStore(cfg);key=store.put(f,identity=identity)
    assert store.put(replace(f,values=f.values.astype('float64')),identity=identity)!=key
    assert store.put(replace(f,image_ids=f.image_ids[::-1],values=f.values[::-1]),identity=identity)!=key


@pytest.mark.parametrize('change',['digest','overlap','missing_release','nonlist'])
def test_invalid_pinned_split(case,change):
    cfg,_,_,_,digest=case
    path=cfg.storage['data_root']/'split.json';m=json.loads(path.read_text())
    if change=='overlap':m['test_ids'][0]=m['train_ids'][0]
    if change=='missing_release':m['dataset_release']=''
    if change=='nonlist':m['train_ids']='image-1'
    raw=json.dumps(m,indent=2).encode();path.write_bytes(raw)
    if change!='digest':digest=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):load_split(cfg,filename='split.json',expected_sha256=digest)


def test_cache_path_traversal_and_pickle_precision_rejected(case):
    cfg,identity,f,_,digest=case
    with pytest.raises(ValueError):FeatureStore(cfg).get('../escape',expected_identity=identity)
    with pytest.raises(ValueError):load_split(cfg,filename='../escape',expected_sha256=digest)
    with pytest.raises(ValueError):FeatureStore(cfg).put(replace(f,values=f.values.astype('float16')),identity=identity)


def test_interrupted_writer_never_publishes_or_overwrites(case,monkeypatch):
    cfg,identity,f,_,_=case
    store=FeatureStore(cfg)
    original=np.save
    def interrupted(*args,**kwargs):raise OSError('simulated interrupted write')
    monkeypatch.setattr(np,'save',interrupted)
    with pytest.raises(OSError,match='interrupted'):store.put(f,identity=identity)
    directory=next(store.root.iterdir())
    assert not (directory/'COMPLETE').exists()
    with pytest.raises(ValueError,match='incomplete'):store.get(directory.name,expected_identity=identity)
    monkeypatch.setattr(np,'save',original)
    with pytest.raises(FileExistsError):store.put(f,identity=identity)


def test_symlink_outside_root_rejected(case,tmp_path):
    cfg,identity,f,_,digest=case
    store=FeatureStore(cfg);key=store.put(f,identity=identity)
    outside=tmp_path/'outside.npy'
    (store.root/key/'values.npy').rename(outside)
    (store.root/key/'values.npy').symlink_to(outside)
    with pytest.raises(ValueError,match='symlink'):store.get(key,expected_identity=identity)
    external=tmp_path/'external.json';external.write_text('{}')
    (cfg.storage['data_root']/'external.json').symlink_to(external)
    with pytest.raises(ValueError,match='escapes'):load_split(cfg,filename='external.json',expected_sha256=digest)
