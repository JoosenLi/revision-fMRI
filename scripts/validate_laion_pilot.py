"""Validate a few pilot responses, explicitly not an OTC scientific evaluation."""
import argparse
from collections import Counter
import json

from revision_fmri import LaionNiftiAdapter
from revision_fmri.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', required=True, help='Manifest filename relative to configured data_root')
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = LaionNiftiAdapter(config, manifest=args.manifest)
    m = adapter.manifest
    if len(m['sessions']) != 1 or m['roi_id'] != 'laion-general-io-pilot':
        raise ValueError('This smoke check is restricted to the named one-session I/O pilot')
    rows = adapter.trials()
    counts = Counter(row.image_id for row in rows)
    labels = tuple(row.image_id for row in rows if counts[row.image_id] == 1)[:3]
    if len(labels) != 3:
        raise ValueError('Pilot needs at least three singly presented labels')
    out = adapter.responses(m['subject_id'], m['roi_id'], labels)
    print(json.dumps({'scope': 'I/O smoke only; not a scientific result',
                      'shape': list(out.values.shape), 'repeat_counts': out.repeat_counts,
                      'finite': True, 'provenance': dict(out.provenance)}, indent=2))


if __name__ == '__main__':
    main()
