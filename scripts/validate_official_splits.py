"""Compare the shared importer against the exact inspected upstream loader."""
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
from revision_fmri import (load_official_split, official_split_inventory,
                           validate_holdout_family, partition_trials)
from revision_fmri.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=load_config(args.config);lock=official_split_inventory()
    source='laion_fmri/splits/__init__.py'
    path=config.storage['data_root']/'official_splits'/lock['commit']/source
    if hashlib.sha256(path.read_bytes()).hexdigest()!=lock['files'][source]['sha256']:
        raise ValueError('Upstream source differs from the inspected pinned file')
    spec=importlib.util.spec_from_file_location('_laion_split_reference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    records={}
    for name in module.list_splits(pool='shared'):
        record=load_official_split(config,name=name,dataset_release='source-parity-validation-only')
        train,test=module.get_train_test_ids(name,pool='shared')
        if tuple(train)!=record.split.train_ids or tuple(test)!=record.split.test_ids:
            raise ValueError(f'Official ID/order mismatch: {name}')
        labels = train + test + [train[0], test[0], 'outside-the-official-pool']
        expected_train, expected_test = module.get_split_masks(labels, name, pool='shared')
        masks = partition_trials(labels, split=record)
        if not (np.array_equal(masks.train, expected_train) and np.array_equal(masks.test, expected_test)):
            raise ValueError(f'Official repeated-trial mask mismatch: {name}')
        records[name]=record
    for family in ('cluster_k5','random'):
        validate_holdout_family([records[f'{family}_{i}'] for i in range(5)])
    print(json.dumps({name:{'train':len(r.split.train_ids),'test':len(r.split.test_ids)}
                      for name,r in records.items()},indent=2))
    print('All official assignments/order and both five-fold families verified')


if __name__=='__main__':main()
