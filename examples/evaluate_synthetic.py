"""Exercise the public API on tiny synthetic arrays; no model/data downloads."""
import argparse
import json
import numpy as np
from revision_fmri import (DeepNSDProtocol, EvaluationSplit, FeatureSet,
                           ReproductionPipeline, ResponseSet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Local YAML configuration')
    parser.add_argument('--run-id', help='Optionally save a new configured results JSON')
    args = parser.parse_args()
    rng = np.random.default_rng(23)
    x = rng.normal(size=(24, 8)).astype('float32')
    y = x @ rng.normal(size=(8, 6)) + rng.normal(size=(24, 6))*.1 + 2
    ids = tuple(f'image-{i}' for i in range(24))
    features = [FeatureSet(ids, 'synthetic', 'layer-1', x, 'raw_flattened', {'source': 'synthetic'})]
    responses = ResponseSet(ids, tuple(f'voxel-{i}' for i in range(6)), y,
                            'subject', 'roi', 'space', (3,)*24, {'source': 'synthetic'})
    split = EvaluationSplit(ids[:16], ids[16:], 'synthetic-v1', 'synthetic')
    protocol = DeepNSDProtocol('synthetic-v1', 5, 'Small demonstration only; not a LAION policy')
    pipeline = ReproductionPipeline.from_config(args.config, protocol=protocol)
    report = pipeline.evaluate_model(features=features, responses=responses, split=split)
    print(json.dumps(report.to_dict()['metrics'], indent=2))
    if args.run_id:
        print(pipeline.save_report(report, run_id=args.run_id))


if __name__ == '__main__':
    main()
