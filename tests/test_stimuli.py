"""Tiny local image fixtures; no CUDA discovery, model, checkpoint or forward."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
Image = pytest.importorskip('PIL.Image')
from revision_fmri import StimulusReader, ClassificationTransform
from revision_fmri.config import load_config
from revision_fmri.provenance import array_digest


@pytest.fixture
def images(tmp_path):
    root=tmp_path/'images';root.mkdir()
    rows=[]
    for index,mode in enumerate(('RGB','L','RGBA','P')):
        image=Image.new(mode,(13+index,7),color=1)
        file=root/f'{mode}.png';image.save(file)
        rows.append({'image_id':f'id-{mode}','file':file.name,
                     'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
    payload={'schema_version':1,'dataset_release':'synthetic','images':rows}
    (root/'manifest.json').write_text(json.dumps(payload))
    cfg=load_config('configs/paths.example.yaml')
    cfg=replace(cfg,storage={**cfg.storage,'stimuli_root':root})
    return cfg,payload


def test_order_partial_batch_rgb_and_model_transform_parity(images,monkeypatch):
    cfg,payload=images
    reader=StimulusReader(cfg,manifest='manifest.json')
    def forbidden(): raise AssertionError('No CUDA probing in stimulus preparation')
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    preprocess=ClassificationTransform()
    ids=('id-P','id-L','id-RGBA','id-RGB')
    batches=list(reader.batches(image_ids=ids,batch_size=3,preprocess=preprocess))
    assert [b.image_ids for b in batches]==[ids[:3],ids[3:]]
    assert [b.inputs.shape for b in batches]==[(3,3,224,224),(1,3,224,224)]
    all_values=torch.cat([b.inputs for b in batches])
    for i,image_id in enumerate(ids):
        row=next(r for r in payload['images'] if r['image_id']==image_id)
        with Image.open(cfg.storage['stimuli_root']/row['file']) as image:
            expected=preprocess(image.convert('RGB'))
        torch.testing.assert_close(all_values[i],expected,rtol=0,atol=0)
    for batch in batches:
        assert batch.inputs.device.type=='cpu' and batch.inputs.dtype==torch.float32
        assert batch.provenance['input_sha256']==array_digest(batch.inputs.numpy())
        assert batch.provenance['transform_sha256']==preprocess.sha256
        assert str(cfg.storage['stimuli_root']) not in json.dumps(batch.provenance)
        assert batch.provenance['seed'] is None


def test_portability_and_no_batch_size_effect_on_values(images,tmp_path):
    cfg,_=images
    target=tmp_path/'moved';shutil.copytree(cfg.storage['stimuli_root'],target)
    other=replace(cfg,storage={**cfg.storage,'stimuli_root':target})
    ids=('id-L','id-P','id-RGB')
    prep=ClassificationTransform()
    a=list(StimulusReader(cfg,manifest='manifest.json').batches(image_ids=ids,batch_size=2,preprocess=prep))
    b=list(StimulusReader(other,manifest='manifest.json').batches(image_ids=ids,batch_size=2,preprocess=prep))
    assert [x.provenance for x in a]==[x.provenance for x in b]
    c=list(StimulusReader(cfg,manifest='manifest.json').batches(image_ids=ids,batch_size=1,preprocess=prep))
    torch.testing.assert_close(torch.cat([x.inputs for x in a]),torch.cat([x.inputs for x in c]),rtol=0,atol=0)


@pytest.mark.parametrize('kind',['missing_id','duplicate_id','bad_size','float64','unknown_transform'])
def test_contract_failures(images,kind):
    cfg,_=images;ids=('id-L',);size=1;prep=ClassificationTransform()
    if kind=='missing_id': ids=('id-L','absent')
    elif kind=='duplicate_id': ids=('id-L','id-L')
    elif kind=='bad_size': size=True
    elif kind=='float64': cfg=replace(cfg,compute=replace(cfg.compute,precision='float64'))
    else: prep=lambda image:image
    with pytest.raises(ValueError):
        next(StimulusReader(cfg,manifest='manifest.json').batches(image_ids=ids,batch_size=size,preprocess=prep))


@pytest.mark.parametrize('kind',['corrupt','hash','missing','escape','symlink'])
def test_local_byte_checks(images,tmp_path,kind):
    cfg,payload=images;root=cfg.storage['stimuli_root'];row=payload['images'][0]
    file=root/row['file']
    if kind=='corrupt':
        file.write_bytes(b'not an image');row['sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    elif kind=='hash': file.write_bytes(b'changed')
    elif kind=='missing': file.unlink()
    else:
        outside=tmp_path/'outside.png';shutil.copyfile(file,outside)
        if kind=='escape': row['file']='../outside.png'
        else:
            file.unlink();file.symlink_to(outside)
    (root/'manifest.json').write_text(json.dumps(payload))
    with pytest.raises((ValueError,FileNotFoundError,Image.UnidentifiedImageError)):
        next(StimulusReader(cfg,manifest='manifest.json').batches(image_ids=('id-RGB',),batch_size=1,preprocess=ClassificationTransform()))


def test_manifest_snapshot_and_source_recheck(images):
    cfg,payload=images;root=cfg.storage['stimuli_root']
    reader=StimulusReader(cfg,manifest='manifest.json')
    (root/'manifest.json').write_text('{}')
    kwargs=dict(image_ids=('id-L',),batch_size=1,preprocess=ClassificationTransform())
    next(reader.batches(**kwargs))  # Manifest snapshot, not mutable on-disk metadata.
    (root/'L.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'): next(reader.batches(**kwargs))


def test_duplicate_manifest_ids_fail(images):
    cfg,payload=images;payload['images'].append(payload['images'][0])
    (cfg.storage['stimuli_root']/'manifest.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='unique'): StimulusReader(cfg,manifest='manifest.json')
