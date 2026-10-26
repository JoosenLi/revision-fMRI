"""Plan/download public sub-01 ses-01 metadata; opt in to its single beta volume.

Uses the documented public S3 route. No raw BOLD, HDF5 duplicate, pretrained
embeddings, stimulus images or license-gated resources are requested.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
import time
import xml.etree.ElementTree as ET

from revision_fmri.config import load_config

ENDPOINT = 'https://laion-fmri.s3.amazonaws.com/'
PREFIX = 'derivatives/glmsingle-tedana/sub-01/ses-01/func/'
PILOT_MASK = 'derivatives/rois/sub-01/laion/sub-01_space-T1w_res-1pt8_label-laion-general_mask.nii.gz'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def open_remote(request, *, timeout):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(1)


def inventory(prefix=PREFIX):
    token = None
    while True:
        query = {'list-type': '2', 'prefix': prefix}
        if token:
            query['continuation-token'] = token
        with open_remote(ENDPOINT+'?'+urllib.parse.urlencode(query), timeout=15) as stream:
            root = ET.fromstring(stream.read())
        for item in root.findall('s:Contents', NS):
            yield {'key': item.findtext('s:Key', namespaces=NS),
                   'size': int(item.findtext('s:Size', namespaces=NS)),
                   'etag': item.findtext('s:ETag', namespaces=NS)}
        token = root.findtext('s:NextContinuationToken', namespaces=NS)
        if not token:
            return


def download_one(root, item):
    key = item['key']
    dest = (root/key).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError('Unsafe remote object key')
    receipt = dest.with_name(dest.name+'.download.json')
    if dest.exists():
        if not receipt.exists():
            raise FileExistsError('Existing file without receipt; will not overwrite')
        saved = json.loads(receipt.read_text())
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if (dest.stat().st_size != item['size'] or saved['etag'] != item['etag']
                or digest.hexdigest() != saved['sha256']):
            raise ValueError('Existing file differs; will not overwrite')
        print('verified', key, flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name+'.part')
    request = urllib.request.Request(ENDPOINT+urllib.parse.quote(key), headers={'If-Match': item['etag']})
    digest, count = hashlib.sha256(), 0
    # Exclusive .part protects concurrent jobs and leaves interrupted data for inspection.
    with partial.open('xb') as output:
        with open_remote(request, timeout=60) as stream:
            while block := stream.read(1024*1024):
                output.write(block)
                digest.update(block)
                count += len(block)
    if count != item['size']:
        raise ValueError('Download size mismatch; partial file retained')
    os.link(partial, dest)  # Atomic exclusive publication; never replace existing data.
    partial.unlink()
    with receipt.open('x') as stream:
        json.dump({**item, 'sha256': digest.hexdigest(), 'source': ENDPOINT+key}, stream, indent=2)
    print('downloaded', key, count, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--download', action='store_true', help='Default is inventory only')
    parser.add_argument('--include-betas', action='store_true', help='Include the roughly 1.16 GB pilot NIfTI')
    parser.add_argument('--include-mask', action='store_true', help='Include laion-general mask for I/O validation, not OTC')
    args = parser.parse_args()
    config = load_config(args.config)
    files = [item for item in inventory() if item['key'].endswith(('.json', '.tsv'))
             or (args.include_betas and item['key'].endswith('_stat-effect_desc-SingletrialBetas_statmap.nii.gz'))]
    if args.include_mask:
        files.extend(item for item in inventory(PILOT_MASK) if item['key'] == PILOT_MASK)
    print(json.dumps({'files': files, 'total_bytes': sum(item['size'] for item in files)}, indent=2), flush=True)
    if args.download:
        for item in files:
            download_one(config.storage['data_root'], item)


if __name__ == '__main__':
    main()
