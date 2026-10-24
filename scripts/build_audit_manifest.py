"""Build the publication inventory from local, pinned audit sources; no network/imports."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

SHA = '36f6a34727dccb5eec8fa49364b210a049a388d7'


def literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left) + literal(node.right)
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def assignments(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = literal(node.value)
            except (ValueError, TypeError):
                pass
    return result


def build(sources, destination):
    repo = sources / 'DeepNSD'
    rows = list(csv.DictReader((repo / 'publication/source_data/results_summary.csv').open()))
    contrasts = list(csv.DictReader((repo / 'source_code/pressures/model_contrasts.csv').open()))
    lookup = {(r['model'], r['train_type']): r for r in contrasts}
    custom = repo / 'source_code/pressures/model_opts/model_code/_options.py'
    options = assignments(custom)
    clip = assignments(sources / 'clip_loader.py')['_MODELS']
    task_tree = ast.parse((sources / 'visualpriors_taskonomy.py').read_text())
    task_urls = {}
    for node in task_tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'TASKONOMY_PRETRAINED_WEIGHT_FILES':
            files = ast.literal_eval(node.value.func.value).split()
            task_urls = {f.split('-')[0]: 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + f for f in files}
    ipcl_path = repo / 'source_code/pressures/model_opts/model_code/ipcl_codebase/__init__.py'
    ipcl_files = {}
    for node in ast.parse(ipcl_path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == 'filename':
                    ipcl_files[node.name] = ast.literal_eval(stmt.value)
    models = []
    for row in rows:
        if row['metric'] != 'wrsa':
            continue
        raw = lookup[(row['model'], row['train_task'])]
        family, name = raw['model_source'], row['model']
        location = raw['weights_url'].strip() or None
        status = 'explicit_source_locator' if location else 'unresolved_library_default'
        note = 'Historical dependency revision and checkpoint bytes have not been verified.'
        if row['train_task'] == 'random' or name == 'random_weights':
            location, status = None, 'unresolved_random_state'
        elif family == 'slip':
            location = options['slip_model_weights'][name]
            status = 'explicit_source_url'
            if 'CC12M' in name:
                note += ' Source key says ViT-L; URL says base; generic train_data says YFCC15M.'
        elif family == 'seer':
            location = options['seer_model_weights_root'] + options['seer_model_weights'][name]
            status = 'explicit_source_url'
        elif family == 'open_ipcl':
            fn = options['ipcl_models'][name]
            location = 'https://visionlab-pretrainedmodels.s3.amazonaws.com/project_instancenet/ipcl/' + ipcl_files[fn]
            status = 'explicit_source_url'
        elif family == 'clip':
            location, status = clip[name], 'upstream_registry_url_historical_version_unresolved'
        elif family == 'taskonomy':
            location, status = task_urls[name + '_encoder'], 'upstream_registry_url_historical_version_unresolved'
        elif family == 'bit_expert':
            location = 'https://tfhub.dev/google/experts/bit/r50x1/in21k/' + name.split('-')[-1].lower() + '/1'
            status = 'explicit_source_locator'
        loader = {
            'torchvision': 'get_torchvision_model(model_path, pretrained)',
            'timm': 'get_timm_model(model_name, pretrained)',
            'taskonomy': 'get_taskonomy_encoder(model_name) / random_taskonomy_encoder()',
            'vissl': 'get_vissl_model(model_name)',
            'clip': 'get_clip_model(model_name): clip.load(...).visual',
            'slip': 'get_slip_model(model_name): checkpoint args; strict state_dict; model.visual',
            'open_ipcl': 'get_ipcl_model(model_name): ' + options['ipcl_models'].get(name, ''),
            'seer': 'get_seer_model(model_name)',
            'bit_expert': 'get_bit_expert_model(expertise)',
            'detectron': 'get_detectron_model(model_name, backbone_only=True)',
            'dino': "torch.hub.load('facebookresearch/dino:main', model_name)",
            'yolo': "torch.hub.load('ultralytics/yolov5', model_name, autoshape=False, force_reload=True)",
            'midas': "torch.hub.load('intel-isl/MiDaS', model_name)",
        }[family]
        models.append(dict(model_id=row['model_string'], model=name, family=family,
            architecture=row['architecture'], task=row['train_task'], diet=row['train_data'],
            comparisons={k: v for k, v in row.items() if k.startswith('compare_') and v not in ('', 'NA')},
            loader_reference=loader, checkpoint_locator=location,
            checkpoint_resolution=status, local_sha256=None, extraction_approved=False,
            notes=note, raw_result_metadata=row, raw_contrast_metadata=raw))
    assert len(models) == len({m['model_id'] for m in models}) == 224
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(schema_version=1, deepnsd_commit=SHA,
        audit_date='2026-10-24', models=models), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.sources, args.output)
