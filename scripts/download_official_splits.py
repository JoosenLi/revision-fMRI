"""Fetch only small hash-locked official split metadata into configured storage."""
import argparse
import hashlib
import urllib.request
from revision_fmri import official_split_inventory
from revision_fmri.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=load_config(args.config);lock=official_split_inventory()
    root=config.storage['data_root']/'official_splits'/lock['commit']
    base=f'https://raw.githubusercontent.com/ViCCo-Group/LAION-fMRI/{lock["commit"]}/'
    for name,expected in lock['files'].items():
        path=root/name
        if not path.resolve().is_relative_to(config.storage['data_root'].resolve()):
            raise ValueError('Destination escapes configured data_root')
        if path.exists():
            data=path.read_bytes()
        else:
            with urllib.request.urlopen(base+name,timeout=30) as stream:
                data=stream.read(expected['size']+1)
        if len(data)!=expected['size'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
            raise ValueError(f'Pinned bytes mismatch: {name}; no existing file overwritten')
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(data)
        print('verified',name)


if __name__=='__main__':main()
