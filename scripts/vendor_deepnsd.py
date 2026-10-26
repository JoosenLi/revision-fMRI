"""Rebuild attributed DeepNSD sources from a local pinned checkout; no network.

Whole ridge file is unchanged. RSA/reduction function bodies are exact excerpts;
only module imports are adapted to avoid loading vision models and torch.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

COMMIT = '36f6a34727dccb5eec8fa49364b210a049a388d7'
FILES = {
 'ridge_gcv_mod.py': ('source_code/pressures/ridge_gcv_mod.py', 'ea80dc539534f5f7370a796c3b2d91fbc73b5e0e4c94a78355616e4db7e62659'),
 'rsa.py': ('source_code/pressures/model_opts/mapping_methods.py', 'e659cb187173cea246a93e40cd4b55c4af57e1f4964c2c27d6496f87784d88e2'),
 'reduction.py': ('source_code/pressures/model_opts/feature_reduction.py', '4ed85a715fcc8bdfa73496270addbf2705d6eb97a03906661b71bbc0f87e21bd'),
 'LICENSE': ('LICENSE', '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'),
}
EXCERPTS = {
 'rsa.py': (['compare_rdms'], 'import numpy as np\nfrom warnings import warn\nfrom scipy.stats import pearsonr, spearmanr\n'),
 'reduction.py': (['get_feature_map_filepaths','recursive_delete_if_empty','delete_saved_output',
                   'compute_jl_lemma','get_feature_map_srps'],
                  'import os\nimport numpy as np\nfrom tqdm.auto import tqdm\nfrom sklearn.random_projection import SparseRandomProjection\n'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def vendor(source, destination, sklearn_license):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'repository': 'https://github.com/ColinConwell/DeepNSD', 'commit': COMMIT, 'files': {}}
    for target, (relative, expected) in FILES.items():
        data = (source / relative).read_bytes()
        if digest(data) != expected:
            raise ValueError(f'Unrecognized source bytes: {relative}')
        entry = {'source_path': relative, 'source_sha256': expected, 'adaptation': 'none'}
        if target in EXCERPTS:
            names, imports = EXCERPTS[target]
            text = data.decode()
            lines = text.splitlines(keepends=True)
            nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
            extracts = []
            entry['functions'] = {}
            for name in names:
                n = nodes[name]
                snippet = ''.join(lines[n.lineno-1:n.end_lineno])
                extracts.append(snippet)
                entry['functions'][name] = {'source_lines': [n.lineno, n.end_lineno], 'sha256': digest(snippet.encode())}
            data = ('# Exact function excerpts from DeepNSD; see NOTICE.md and manifest.json.\n'
                    + imports + '\n\n' + '\n\n'.join(extracts)).encode()
            entry['adaptation'] = 'exact function excerpts; minimal module imports'
        (destination / target).write_bytes(data)
        entry['vendored_sha256'] = digest(data)
        manifest['files'][target] = entry
    bsd = sklearn_license.read_bytes()
    (destination / 'SKLEARN_COPYING').write_bytes(bsd)
    manifest['sklearn_license_sha256'] = digest(bsd)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--sklearn-license', type=Path, required=True)
    args = parser.parse_args()
    vendor(args.source_root, args.destination, args.sklearn_license)
